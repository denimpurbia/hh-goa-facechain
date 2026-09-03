"""Unit tests for cryptographic hashing and JSON canonicalization."""

from pathlib import Path
import pytest
from src.hashing import calculate_sha256, canonicalize_json, hash_file, hash_metadata


def test_calculate_sha256_basic():
    """Verify SHA-256 computation against known test vectors."""
    # Known SHA-256 for "hello world"
    expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert calculate_sha256("hello world") == expected
    assert calculate_sha256(b"hello world") == expected


def test_canonicalize_json_key_order_independence():
    """Verify that different dictionary key insertions produce identical canonical JSON."""
    dict_a = {"z": 1, "a": 2, "m": 3}
    dict_b = {"a": 2, "m": 3, "z": 1}
    dict_c = {"m": 3, "z": 1, "a": 2}

    canonical_a = canonicalize_json(dict_a)
    canonical_b = canonicalize_json(dict_b)
    canonical_c = canonicalize_json(dict_c)

    assert canonical_a == canonical_b == canonical_c
    assert canonical_a == '{"a":2,"m":3,"z":1}'


def test_hash_metadata_consistency_and_sensitivity():
    """Verify metadata hashing stability and alteration sensitivity."""
    meta1 = {
        "source_post_url": "https://example.com/post/1",
        "similarity_score": 0.85,
        "match_threshold": 0.60,
    }
    meta1_reordered = {
        "similarity_score": 0.85,
        "match_threshold": 0.60,
        "source_post_url": "https://example.com/post/1",
    }
    meta_altered = {
        "source_post_url": "https://example.com/post/2",
        "similarity_score": 0.85,
        "match_threshold": 0.60,
    }

    hash1 = hash_metadata(meta1)
    hash1_reordered = hash_metadata(meta1_reordered)
    hash_altered = hash_metadata(meta_altered)

    assert len(hash1) == 64
    assert hash1 == hash1_reordered
    assert hash1 != hash_altered


def test_hash_file(tmp_path: Path):
    """Test file hashing utility on temporary content."""
    test_file = tmp_path / "test_data.bin"
    test_file.write_bytes(b"facechain-cryptographic-payload")

    file_hash = hash_file(test_file)
    expected_hash = calculate_sha256(b"facechain-cryptographic-payload")
    assert file_hash == expected_hash

    # Non-existent file should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        hash_file(tmp_path / "non_existent.bin")
