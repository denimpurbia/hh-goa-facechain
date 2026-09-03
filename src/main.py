"""Main end-to-end pipeline execution entrypoint for FaceChain Verify."""

import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional

from src.blockchain import BlockchainVerifier
from src.cli import TerminalFormatter, parse_arguments
from src.config import AppConfig, load_config, setup_logger
from src.face_processor import FaceProcessor
from src.hashing import hash_file, hash_metadata
from src.metadata import create_verification_metadata
from src.post_matcher import PostMatcher
from src.reverse_search import get_search_provider
from src.verifier import verify_metadata


def run_pipeline(
    input_image_path: str,
    threshold: Optional[float] = None,
    provider_override: Optional[str] = None,
    output_path: Optional[str] = None,
    skip_blockchain: bool = False,
    use_demo_fixture: bool = False,
) -> int:
    """Run the 8-step end-to-end verification pipeline.

    Args:
        input_image_path: Path to input query face image.
        threshold: Optional override for face match threshold.
        provider_override: Optional search provider name override.
        output_path: Destination path for JSON result.
        skip_blockchain: If True, bypass blockchain storage and verification.
        use_demo_fixture: If True, execute with offline demo fixture.

    Returns:
        Exit code (0 for success, 1 for error or no match).
    """
    config = load_config()
    logger = setup_logger("facechain", config.log_level)
    fmt = TerminalFormatter

    fmt.print_banner()

    match_threshold = threshold if threshold is not None else config.face_match_threshold
    effective_provider = (provider_override or config.search_provider).lower()
    input_path = Path(input_image_path).resolve()

    # Step 1: Loading input image
    fmt.step(1, 8, "Loading input image")
    if not input_path.is_file():
        fmt.failure(f"Input image not found: {input_image_path}")
        fmt.print_footer_error("Input file not found.")
        return 1

    try:
        input_file_hash = hash_file(input_path)
        fmt.success("Image loaded")
        fmt.info("File", input_path.name)
        fmt.info("SHA-256", f"{input_file_hash[:16]}...{input_file_hash[-8:]}")
    except Exception as e:
        fmt.failure(f"Failed to read image: {e}")
        fmt.print_footer_error(str(e))
        return 1

    # Step 2: Detecting face
    fmt.step(2, 8, "Detecting face")
    processor = FaceProcessor()
    try:
        face_result = processor.process_face(input_path)
        fmt.success("Face detected")
        fmt.info("Faces detected", face_result.faces_detected)
        fmt.info("Detection confidence", f"{face_result.detection_score:.2f}")
    except ValueError as e:
        fmt.failure(f"Face detection failed: {e}")
        fmt.print_footer_error("No detectable face found in input image.")
        return 1
    except Exception as e:
        fmt.failure(f"Unexpected error during face detection: {e}")
        fmt.print_footer_error(str(e))
        return 1

    # Step 3: Encoding face
    fmt.step(3, 8, "Encoding face")
    fmt.success("Face embedding generated")
    fmt.info("Embedding dimension", face_result.embedding_dimension)

    # Step 4: Performing reverse image search
    fmt.step(4, 8, "Performing reverse image search")
    display_provider = "Demo Fixture" if use_demo_fixture else effective_provider.capitalize()
    fmt.info("Provider", display_provider)
    print("Searching dynamically...")

    fixture_data = None
    if use_demo_fixture:
        # Provide sample fixture pointing to the same image for demonstration
        fixture_data = [
            {
                "title": "Verified Public Identity Profile & Publication",
                "url": "https://identity-network.org/records/verified-user-2026",
                "source_domain": "identity-network.org",
                "image_url": str(input_path.as_uri()),
                "thumbnail_url": str(input_path.as_uri()),
            }
        ]

    try:
        search_provider = get_search_provider(
            config,
            use_fixture=use_demo_fixture,
            fixture_data=fixture_data,
        )
        candidates = search_provider.search(input_path, max_results=config.max_search_results)
        fmt.success("Search completed")
        fmt.info("Results discovered", len(candidates))
    except RuntimeError as re:
        print("\n" + str(re))
        fmt.print_footer_error("Search configuration required or search request failed.")
        return 1
    except Exception as e:
        fmt.failure(f"Search failed: {e}")
        fmt.print_footer_error(str(e))
        return 1

    if not candidates:
        fmt.failure("Search returned zero candidate results.")
        fmt.print_footer_no_match()
        return 1

    # Step 5: Verifying candidate matches
    fmt.step(5, 8, "Verifying candidate matches")
    matcher = PostMatcher(face_processor=processor, timeout=config.search_timeout)
    match_result = matcher.evaluate_candidates(
        candidates=candidates,
        input_embedding=face_result.embedding,
        threshold=match_threshold,
    )

    if not match_result.is_match or not match_result.best_candidate:
        fmt.failure(f"No candidate passed match threshold ({match_threshold:.2f})")
        if match_result.similarity_score > 0:
            fmt.info("Highest similarity found", f"{match_result.similarity_score:.2f}")
        fmt.print_footer_no_match()
        return 1

    best_cand = match_result.best_candidate
    cand_image_sha256 = match_result.candidate_image_sha256 or input_file_hash

    fmt.info("Candidate", best_cand.title or best_cand.url)
    fmt.info("Source Domain", best_cand.source_domain)
    fmt.info("Similarity", f"{match_result.similarity_score:.2f}")
    fmt.info("Threshold", f"{match_threshold:.2f}")
    fmt.success("VERIFIED FACE MATCH")

    # Step 6: Creating tamper-evident metadata
    fmt.step(6, 8, "Creating tamper-evident metadata")
    metadata = create_verification_metadata(
        source_post_url=best_cand.url,
        source_domain=best_cand.source_domain,
        post_title=best_cand.title,
        candidate_image_url=best_cand.image_url or best_cand.url,
        similarity_score=match_result.similarity_score,
        match_threshold=match_threshold,
        input_image_sha256=input_file_hash,
        candidate_image_sha256=cand_image_sha256,
        face_detection_score=face_result.detection_score,
    )
    metadata_hash = hash_metadata(metadata)
    fmt.success("Metadata created")
    fmt.success("SHA-256 generated")
    fmt.info("Hash", metadata_hash)

    # Step 7: Recording fingerprint on blockchain
    fmt.step(7, 8, "Recording fingerprint on blockchain")
    receipt = None
    blockchain_verifier = None

    if not skip_blockchain:
        try:
            blockchain_verifier = BlockchainVerifier(
                provider_type=config.blockchain_provider,
                rpc_url=config.blockchain_rpc_url,
            )
            fmt.info("Blockchain", "Ethereum Tester (PyEVM Local Testnet)")
            receipt = blockchain_verifier.store_hash(metadata_hash)
            fmt.info("Transaction", receipt.transaction_hash)
            fmt.info("Block Number", receipt.block_number)
            fmt.success("RECORD STORED")
        except Exception as e:
            fmt.failure(f"Blockchain storage failed: {e}")
            fmt.print_footer_error(str(e))
            return 1
    else:
        fmt.info("Status", "Skipped by user flag (--skip-blockchain)")

    # Step 8: Re-verifying record
    fmt.step(8, 8, "Re-verifying record")
    verif_res = None
    if blockchain_verifier is not None:
        verif_res = verify_metadata(
            metadata=metadata,
            blockchain_verifier=blockchain_verifier,
            expected_stored_hash=metadata_hash,
        )
        fmt.info("Stored Hash", verif_res.stored_hash)
        fmt.info("Current Hash", verif_res.current_hash)
        fmt.info("Blockchain Status", "FOUND" if verif_res.blockchain_verified else "NOT FOUND")

        if verif_res.blockchain_verified and not verif_res.tampering_detected:
            fmt.success("VERIFIED")
            print("DATA HAS NOT BEEN TAMPERED WITH")
        else:
            fmt.failure("TAMPERING DETECTED")
            print(verif_res.status_message)
    else:
        fmt.info("Status", "Re-verification skipped (Blockchain disabled)")

    # Save output JSON
    dest_path = Path(output_path or "results/verification_result.json")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    result_json: Dict[str, Any] = {
        "project": "FaceChain Verify",
        "timestamp": metadata["search_timestamp"],
        "face_detection": {
            "faces_detected": face_result.faces_detected,
            "detection_score": face_result.detection_score,
            "embedding_dimension": face_result.embedding_dimension,
        },
        "search": {
            "provider": display_provider.lower(),
            "results_found": len(candidates),
        },
        "match": {
            "found": True,
            "similarity_score": match_result.similarity_score,
            "threshold": match_threshold,
            "source_url": best_cand.url,
            "source_domain": best_cand.source_domain,
            "post_title": best_cand.title,
        },
        "metadata": metadata,
        "blockchain": receipt.to_dict() if receipt else {"status": "skipped"},
        "verification": verif_res.to_dict() if verif_res else {"status": "skipped"},
    }

    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)

    fmt.info("Results saved to", str(dest_path))
    fmt.print_footer_success()
    return 0


def main() -> None:
    """CLI execution wrapper."""
    args = parse_arguments()
    exit_code = run_pipeline(
        input_image_path=args.input,
        threshold=args.threshold,
        provider_override=args.provider,
        output_path=args.output,
        skip_blockchain=args.skip_blockchain,
        use_demo_fixture=args.demo_fixture,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
