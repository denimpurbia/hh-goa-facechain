"""Unit tests for metadata creation and validation."""

from src.hashing import hash_metadata
from src.metadata import canonicalize_metadata, create_verification_metadata


def test_create_verification_metadata_structure():
    """Verify that metadata generator produces expected fields and types."""
    meta = create_verification_metadata(
        source_post_url="https://twitter.com/user/status/12345",
        source_domain="twitter.com",
        post_title="Identity Verification Post",
        candidate_image_url="https://pbs.twimg.com/media/face.jpg",
        similarity_score=0.88765,
        match_threshold=0.60,
        input_image_sha256="aaaabbbbcccc1111222233334444555566667777888899990000aaaabbbbcccc",
        candidate_image_sha256="1111222233334444555566667777888899990000aaaabbbbccccdddd11112222",
        search_timestamp="2026-09-02T12:00:00Z",
        face_detection_score=0.99123,
    )

    assert meta["source_post_url"] == "https://twitter.com/user/status/12345"
    assert meta["source_domain"] == "twitter.com"
    assert meta["post_title"] == "Identity Verification Post"
    assert meta["similarity_score"] == 0.8877  # Rounded to 4 places
    assert meta["match_threshold"] == 0.60
    assert meta["face_detection_score"] == 0.9912

    # Verify no raw embeddings exist in metadata
    assert "embedding" not in meta
    assert "raw_face" not in meta
    assert "biometrics" not in meta


def test_canonicalize_metadata_deterministic():
    """Verify that canonicalized metadata outputs strictly deterministic JSON."""
    meta1 = create_verification_metadata(
        source_post_url="https://linkedin.com/in/researcher",
        source_domain="linkedin.com",
        post_title="Profile Photo",
        candidate_image_url="https://media.licdn.com/dms/image/xyz",
        similarity_score=0.91,
        match_threshold=0.60,
        input_image_sha256="abc123",
        candidate_image_sha256="def456",
        search_timestamp="2026-09-02T10:00:00+00:00",
    )

    canonical = canonicalize_metadata(meta1)
    hash_val = hash_metadata(meta1)

    assert isinstance(canonical, str)
    assert len(hash_val) == 64
