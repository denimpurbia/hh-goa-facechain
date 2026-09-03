"""Cryptographic hashing utilities for FaceChain Verify.

Ensures deterministic, canonical SHA-256 fingerprinting for verification metadata
and media files.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Union


def calculate_sha256(data: Union[str, bytes]) -> str:
    """Calculate the SHA-256 hexadecimal digest for raw string or bytes.

    Args:
        data: String (UTF-8 encoded) or bytes to hash.

    Returns:
        Lowercase hexadecimal SHA-256 string (64 characters).
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    hasher = hashlib.sha256()
    hasher.update(data)
    return hasher.hexdigest().lower()


def canonicalize_json(data: Dict[str, Any]) -> str:
    """Deterministically serialize a dictionary into a canonical JSON string.

    Uses sorted keys, no whitespace separators, and ensures strict UTF-8
    compatibility without ASCII escaping.

    Args:
        data: Dictionary to canonicalize.

    Returns:
        Deterministic canonical JSON string.
    """
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def hash_metadata(metadata: Dict[str, Any]) -> str:
    """Generate a SHA-256 fingerprint from a verification metadata dictionary.

    Args:
        metadata: Metadata dictionary.

    Returns:
        64-character lowercase hexadecimal SHA-256 digest.
    """
    canonical_str = canonicalize_json(metadata)
    return calculate_sha256(canonical_str)


def hash_file(file_path: Union[str, Path]) -> str:
    """Calculate SHA-256 hash of a file on disk in streaming chunks.

    Args:
        file_path: Path to the target file.

    Returns:
        64-character lowercase hexadecimal SHA-256 digest.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Target file not found for hashing: {file_path}")

    hasher = hashlib.sha256()
    chunk_size = 64 * 1024  # 64 KB chunks
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest().lower()
