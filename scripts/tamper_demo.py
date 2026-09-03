"""Tamper Demonstration Script for FaceChain Verify.

Demonstrates real, cryptographic tamper detection:
1. Creates canonical verification metadata and writes its SHA-256 fingerprint to the Ethereum test chain.
2. Verifies the original metadata against the blockchain (Status: VERIFIED).
3. Modifies a single field in memory (e.g., changing source_post_url).
4. Re-computes the cryptographic SHA-256 fingerprint.
5. Verifies the tampered record against the blockchain (Status: TAMPERING DETECTED).
"""

import copy
import sys
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.blockchain import BlockchainVerifier
from src.cli import TerminalFormatter
from src.hashing import hash_metadata
from src.metadata import create_verification_metadata
from src.verifier import verify_metadata


def run_tamper_demo() -> None:
    """Execute the interactive tamper detection demonstration."""
    fmt = TerminalFormatter
    fmt.print_banner()
    print("DEMO: CRYPTOGRAPHIC TAMPER DETECTION")
    print("=" * 50)

    # 1. Initialize local Ethereum test blockchain
    print("\n[Step 1] Initializing Ethereum Blockchain Test Ledger...")
    blockchain = BlockchainVerifier(provider_type="ethereum_tester", reuse_shared_state=False)
    print("✓ Connected to EthereumTesterProvider")

    # 2. Create legitimate original metadata
    print("\n[Step 2] Creating Genuine Verification Metadata...")
    original_metadata = create_verification_metadata(
        source_post_url="https://twitter.com/ai_researcher/status/1789201948",
        source_domain="twitter.com",
        post_title="Announcing FaceChain Verify at HH Goa 2026",
        candidate_image_url="https://pbs.twimg.com/media/verified_profile.jpg",
        similarity_score=0.8842,
        match_threshold=0.60,
        input_image_sha256="4a3b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b",
        candidate_image_sha256="8f9e0d1c2b3a4f5e6d7c8b9a0f1e2d3c4b5a6f7e8d9c0b1a2f3e4d5c6b7a8f9e",
        search_timestamp="2026-09-02T12:00:00+00:00",
        face_detection_score=0.9921,
    )

    original_hash = hash_metadata(original_metadata)
    print(f"Original Post URL : {original_metadata['source_post_url']}")
    print(f"Original SHA-256  : {original_hash}")

    # 3. Store original hash on blockchain
    print("\n[Step 3] Broadcasting SHA-256 Fingerprint to Ethereum Blockchain...")
    receipt = blockchain.store_hash(original_hash)
    print(f"✓ Transaction Hash: {receipt.transaction_hash}")
    print(f"✓ Block Number    : {receipt.block_number}")
    print("✓ RECORD STORED ON BLOCKCHAIN")

    # 4. Verify original record
    print("\n" + "-" * 50)
    print("VERIFYING ORIGINAL RECORD")
    print("-" * 50)
    original_verification = verify_metadata(
        metadata=original_metadata,
        blockchain_verifier=blockchain,
        expected_stored_hash=original_hash,
    )
    print(f"Stored Hash       : {original_verification.stored_hash}")
    print(f"Current Hash      : {original_verification.current_hash}")
    print(f"Blockchain Status : {'FOUND' if original_verification.blockchain_verified else 'NOT FOUND'}")
    if original_verification.blockchain_verified and not original_verification.tampering_detected:
        print("✓ VERIFIED")
        print("DATA HAS NOT BEEN TAMPERED WITH")
    else:
        print("❌ VERIFICATION FAILED")

    # 5. Tamper with metadata in memory
    print("\n" + "=" * 50)
    print("SIMULATING UNAUTHORIZED DATA TAMPERING")
    print("=" * 50)
    tampered_metadata = copy.deepcopy(original_metadata)
    
    # Tamper with the source URL (e.g., pointing to an attacker-controlled page)
    tampered_metadata["source_post_url"] = "https://tampered-fake-news.example.com/malicious-post/9999"
    tampered_hash = hash_metadata(tampered_metadata)

    print(f"Original Post URL : {original_metadata['source_post_url']}")
    print(f"Tampered Post URL : {tampered_metadata['source_post_url']}")
    print(f"Original SHA-256  : {original_hash}")
    print(f"Tampered SHA-256  : {tampered_hash}")

    # 6. Verify tampered record against original blockchain record
    print("\n" + "-" * 50)
    print("VERIFYING TAMPERED RECORD AGAINST BLOCKCHAIN")
    print("-" * 50)
    tampered_verification = verify_metadata(
        metadata=tampered_metadata,
        blockchain_verifier=blockchain,
        expected_stored_hash=original_hash,
    )
    print(f"Expected Stored Hash : {tampered_verification.stored_hash}")
    print(f"Recalculated Hash   : {tampered_verification.current_hash}")
    print(f"Blockchain Status    : {'FOUND' if tampered_verification.blockchain_verified else 'NOT FOUND'}")

    if tampered_verification.tampering_detected:
        print("❌ TAMPERING DETECTED")
        print(f"Reason: {tampered_verification.status_message}")
    else:
        print("✓ Record matched (unexpected)")

    print("\n" + "=" * 50)
    print("TAMPER DEMONSTRATION COMPLETE: Cryptographic Mismatch Proved")
    print("=" * 50)


if __name__ == "__main__":
    run_tamper_demo()
