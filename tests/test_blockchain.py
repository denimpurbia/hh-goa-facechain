"""Unit tests for Ethereum blockchain storage and verification."""

import pytest
from src.blockchain import BlockchainVerifier
from src.hashing import calculate_sha256


@pytest.fixture
def blockchain():
    """Create an isolated test blockchain instance."""
    return BlockchainVerifier(provider_type="ethereum_tester", reuse_shared_state=False)


def test_blockchain_store_and_verify_hash(blockchain):
    """Verify that a 32-byte SHA-256 hash is genuinely stored in transaction data and verified."""
    test_hash = calculate_sha256("test-metadata-fingerprint-payload")
    
    # Store hash on local test chain
    receipt = blockchain.store_hash(test_hash)

    assert receipt.data_hash == test_hash
    assert receipt.transaction_hash.startswith("0x")
    assert receipt.block_number >= 1
    assert receipt.gas_used > 0

    # Independent on-chain query must find the hash
    is_found = blockchain.verify_hash(test_hash)
    assert is_found is True


def test_blockchain_non_existent_hash(blockchain):
    """Verify that a random hash not stored on chain returns False."""
    random_hash = calculate_sha256("completely-unrecorded-data")
    assert blockchain.verify_hash(random_hash) is False


def test_blockchain_invalid_hash_format(blockchain):
    """Verify error handling when an invalid hash format is supplied."""
    with pytest.raises(ValueError):
        blockchain.store_hash("short_hash")


def test_blockchain_get_transaction_details(blockchain):
    """Verify retrieval of mined transaction details."""
    test_hash = calculate_sha256("tx-details-test")
    receipt = blockchain.store_hash(test_hash)

    details = blockchain.get_transaction_details(receipt.transaction_hash)
    assert details is not None
    assert details["transaction_hash"] == receipt.transaction_hash
    assert details["block_number"] == receipt.block_number
    assert details["status"] == 1
