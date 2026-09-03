"""Verification metadata creation and canonical serialization.

Produces privacy-safe metadata structures containing cryptographic hashes,
source verification details, and match confidence, without leaking raw biometric vectors.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from src.hashing import canonicalize_json


def create_verification_metadata(
    source_post_url: str,
    source_domain: str,
    post_title: str,
    candidate_image_url: str,
    similarity_score: float,
    match_threshold: float,
    input_image_sha256: str,
    candidate_image_sha256: str,
    search_timestamp: Optional[str] = None,
    face_detection_score: Optional[float] = None,
) -> Dict[str, Any]:
    """Create a standardized verification metadata record.

    Biometric embeddings and raw images are NEVER included.

    Args:
        source_post_url: URL where matching content was found.
        source_domain: Normalized domain name of the source (e.g. twitter.com).
        post_title: Title or headline of the matching public post.
        candidate_image_url: URL of the discovered image.
        similarity_score: Cosine similarity score between face embeddings.
        match_threshold: Threshold used for positive verification.
        input_image_sha256: SHA-256 fingerprint of the input face image.
        candidate_image_sha256: SHA-256 fingerprint of the discovered candidate image.
        search_timestamp: Optional ISO 8601 UTC timestamp string.
        face_detection_score: Optional face detection confidence score.

    Returns:
        Structured dictionary ready for deterministic hashing.
    """
    if search_timestamp is None:
        search_timestamp = datetime.now(timezone.utc).isoformat()

    metadata = {
        "candidate_image_sha256": str(candidate_image_sha256).lower(),
        "candidate_image_url": str(candidate_image_url or ""),
        "input_image_sha256": str(input_image_sha256).lower(),
        "match_threshold": round(float(match_threshold), 4),
        "post_title": str(post_title or ""),
        "search_timestamp": str(search_timestamp),
        "similarity_score": round(float(similarity_score), 4),
        "source_domain": str(source_domain or "").lower(),
        "source_post_url": str(source_post_url or ""),
    }

    if face_detection_score is not None:
        metadata["face_detection_score"] = round(float(face_detection_score), 4)

    return metadata


def canonicalize_metadata(metadata: Dict[str, Any]) -> str:
    """Serialize verification metadata into canonical deterministic JSON.

    Args:
        metadata: Verification metadata dictionary.

    Returns:
        Canonical JSON string.
    """
    return canonicalize_json(metadata)
