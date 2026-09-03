"""Unit tests for the end-to-end verifier and tampering detection."""

import copy
import pytest
from src.blockchain import BlockchainVerifier
from src.hashing import hash_metadata
from src.metadata import create_verification_metadata
from src.verifier import verify_metadata


@pytest.fixture
def blockchain():
    """Create isolated blockchain test network."""
    return BlockchainVerifier(provider_type="ethereum_tester", reuse_shared_state=False)


def test_verifier_genuine_metadata_success(blockchain):
    """Verify that untampered metadata with matching blockchain record passes verification."""
    meta = create_verification_metadata(
        source_post_url="https://twitter.com/researcher/123",
        source_domain="twitter.com",
        post_title="Original Post",
        candidate_image_url="https://images.example.com/original.jpg",
        similarity_score=0.85,
        match_threshold=0.60,
        input_image_sha256="1111222233334444555566667777888899990000aaaabbbbccccdddd11112222",
        candidate_image_sha256="aaaabbbbcccc1111222233334444555566667777888899990000aaaabbbbcccc",
        search_timestamp="2026-09-02T12:00:00Z",
    )

    data_hash = hash_metadata(meta)
    blockchain.store_hash(data_hash)

    result = verify_metadata(
        metadata=meta,
        blockchain_verifier=blockchain,
        expected_stored_hash=data_hash,
    )

    assert result.blockchain_verified is True
    assert result.tampering_detected is False
    assert result.stored_hash == data_hash
    assert result.current_hash == data_hash
    assert "DATA HAS NOT BEEN TAMPERED WITH" in result.status_message


def test_verifier_tampering_detected_when_content_modified(blockchain):
    """Verify that modifying a single field in metadata triggers tampering detection."""
    original_meta = create_verification_metadata(
        source_post_url="https://twitter.com/researcher/123",
        source_domain="twitter.com",
        post_title="Original Post",
        candidate_image_url="https://images.example.com/original.jpg",
        similarity_score=0.85,
        match_threshold=0.60,
        input_image_sha256="1111222233334444555566667777888899990000aaaabbbbccccdddd11112222",
        candidate_image_sha256="aaaabbbbcccc1111222233334444555566667777888899990000aaaabbbbcccc",
        search_timestamp="2026-09-02T12:00:00Z",
    )

    orig_hash = hash_metadata(original_meta)
    blockchain.store_hash(orig_hash)

    # Tamper with metadata field
    tampered_meta = copy.deepcopy(original_meta)
    tampered_meta["source_post_url"] = "https://tampered-fake-site.com/exploit"

    result = verify_metadata(
        metadata=tampered_meta,
        blockchain_verifier=blockchain,
        expected_stored_hash=orig_hash,
    )

    assert result.blockchain_verified is False
    assert result.tampering_detected is True
    assert result.stored_hash == orig_hash
    assert result.current_hash != orig_hash
    assert "TAMPERING DETECTED" in result.status_message


def test_verifier_unrecorded_metadata_fails(blockchain):
    """Verify that valid metadata not recorded on the blockchain is flagged."""
    unrecorded_meta = create_verification_metadata(
        source_post_url="https://example.com/unrecorded",
        source_domain="example.com",
        post_title="Unrecorded Post",
        candidate_image_url="https://example.com/img.jpg",
        similarity_score=0.75,
        match_threshold=0.60,
        input_image_sha256="1111111111111111111111111111111111111111111111111111111111111111",
        candidate_image_sha256="2222222222222222222222222222222222222222222222222222222222222222",
        search_timestamp="2026-09-02T12:00:00Z",
    )

    result = verify_metadata(
        metadata=unrecorded_meta,
        blockchain_verifier=blockchain,
    )

    assert result.blockchain_verified is False
    assert result.tampering_detected is True
