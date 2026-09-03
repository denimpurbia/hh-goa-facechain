"""Verification and tamper-detection engine.

Independently checks canonical metadata against SHA-256 fingerprints
and blockchain transaction logs.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional

from src.blockchain import BlockchainVerifier
from src.hashing import hash_metadata
from src.metadata import canonicalize_metadata

logger = logging.getLogger("facechain.verifier")


@dataclass
class VerificationResult:
    """Structured result of metadata and blockchain integrity verification."""

    stored_hash: str
    current_hash: str
    blockchain_verified: bool
    tampering_detected: bool
    status_message: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "stored_hash": self.stored_hash,
            "current_hash": self.current_hash,
            "blockchain_verified": self.blockchain_verified,
            "tampering_detected": self.tampering_detected,
            "status_message": self.status_message,
        }


def verify_metadata(
    metadata: Dict[str, Any],
    blockchain_verifier: BlockchainVerifier,
    expected_stored_hash: Optional[str] = None,
) -> VerificationResult:
    """Independently re-verify metadata integrity against the blockchain.

    Flow:
        metadata
        -> canonicalize
        -> SHA-256 (current_hash)
        -> check blockchain record

    Args:
        metadata: Metadata dictionary to verify.
        blockchain_verifier: Initialized BlockchainVerifier instance.
        expected_stored_hash: Optional expected original hash for explicit comparison.

    Returns:
        VerificationResult detailing integrity status.
    """
    # 1. Compute current hash from canonicalized metadata
    current_hash = hash_metadata(metadata)
    stored_hash = (expected_stored_hash or current_hash).lower().strip()
    if stored_hash.startswith("0x"):
        stored_hash = stored_hash[2:]

    # 2. Check if the current hash exists on the immutable blockchain ledger
    on_chain = blockchain_verifier.verify_hash(current_hash)

    # 3. Detect tampering
    if expected_stored_hash is not None and stored_hash != current_hash:
        tampering_detected = True
        status_message = (
            f"TAMPERING DETECTED: Metadata has been modified. "
            f"Expected hash '{stored_hash[:12]}...' does not match recomputed hash '{current_hash[:12]}...'."
        )
    elif not on_chain:
        tampering_detected = True
        status_message = (
            f"TAMPERING DETECTED: Hash '{current_hash[:12]}...' was not found on the blockchain ledger."
        )
    else:
        tampering_detected = False
        status_message = "DATA HAS NOT BEEN TAMPERED WITH. Cryptographic hash verified on blockchain."

    blockchain_verified = on_chain and not tampering_detected

    return VerificationResult(
        stored_hash=stored_hash,
        current_hash=current_hash,
        blockchain_verified=blockchain_verified,
        tampering_detected=tampering_detected,
        status_message=status_message,
    )
