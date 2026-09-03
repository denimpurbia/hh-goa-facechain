"""Blockchain integration module using Web3.py and EthereumTesterProvider.

Creates immutable, tamper-evident records by storing cryptographic SHA-256
metadata fingerprints on an Ethereum-compatible blockchain test network.
Biometric embeddings and raw images are NEVER stored on-chain.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional
from hexbytes import HexBytes
from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider

logger = logging.getLogger("facechain.blockchain")


@dataclass
class BlockchainReceipt:
    """Receipt returned after recording a metadata fingerprint on-chain."""

    data_hash: str
    transaction_hash: str
    block_number: int
    block_timestamp: int
    from_address: str
    to_address: Optional[str]
    gas_used: int
    provider: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize receipt to dictionary."""
        return {
            "provider": self.provider,
            "data_hash": self.data_hash,
            "transaction_hash": self.transaction_hash,
            "block_number": self.block_number,
            "block_timestamp": self.block_timestamp,
            "from_address": self.from_address,
            "to_address": self.to_address,
            "gas_used": self.gas_used,
        }


class BlockchainVerifier:
    """Manages recording and cryptographic re-verification on an Ethereum test chain."""

    # Shared default provider to maintain chain state across module reloads
    _shared_provider: Optional[EthereumTesterProvider] = None

    def __init__(
        self,
        provider_type: str = "ethereum_tester",
        rpc_url: Optional[str] = None,
        reuse_shared_state: bool = True,
    ):
        self.provider_type = provider_type.lower()
        self.rpc_url = rpc_url

        if self.provider_type == "http_rpc" and self.rpc_url:
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
            logger.info(f"Connected to HTTP RPC blockchain provider: {self.rpc_url}")
        else:
            if reuse_shared_state:
                if BlockchainVerifier._shared_provider is None:
                    BlockchainVerifier._shared_provider = EthereumTesterProvider()
                self.w3 = Web3(BlockchainVerifier._shared_provider)
            else:
                self.w3 = Web3(EthereumTesterProvider())
            logger.info("Connected to local EthereumTesterProvider blockchain network.")

        if not self.w3.is_connected():
            raise ConnectionError("Failed to connect to the configured blockchain provider.")

        self.default_account = self.w3.eth.accounts[0] if self.w3.eth.accounts else None

    def store_hash(self, data_hash: str) -> BlockchainReceipt:
        """Record a 32-byte SHA-256 metadata fingerprint on the blockchain.

        Constructs and broadcasts an Ethereum transaction embedding the hex-encoded
        hash into the transaction `data` payload.

        Args:
            data_hash: 64-character hex string representing the SHA-256 fingerprint.

        Returns:
            BlockchainReceipt with block and transaction information.
        """
        clean_hash = data_hash.lower().strip()
        if clean_hash.startswith("0x"):
            clean_hash = clean_hash[2:]

        if len(clean_hash) != 64:
            raise ValueError(f"Invalid SHA-256 hash length ({len(clean_hash)}). Expected 64 hex characters.")

        # Format as HexBytes payload (0x prefixed)
        tx_data = HexBytes("0x" + clean_hash)

        # Build transaction using default funded test account
        tx_params = {
            "from": self.default_account,
            "to": self.w3.eth.accounts[1] if len(self.w3.eth.accounts) > 1 else self.default_account,
            "value": 0,
            "data": tx_data,
            "gas": 100000,
            "gasPrice": self.w3.eth.gas_price if hasattr(self.w3.eth, "gas_price") else 1000000000,
        }

        tx_hash_bytes = self.w3.eth.send_transaction(tx_params)
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash_bytes)

        block = self.w3.eth.get_block(tx_receipt.blockNumber)
        tx_hash_hex = tx_receipt.transactionHash.hex()
        if not tx_hash_hex.startswith("0x"):
            tx_hash_hex = "0x" + tx_hash_hex

        logger.info(f"Recorded hash on blockchain. Tx: {tx_hash_hex}, Block: {tx_receipt.blockNumber}")

        return BlockchainReceipt(
            data_hash=clean_hash,
            transaction_hash=tx_hash_hex,
            block_number=tx_receipt.blockNumber,
            block_timestamp=block.timestamp if hasattr(block, "timestamp") else 0,
            from_address=str(tx_params["from"]),
            to_address=str(tx_params["to"]),
            gas_used=int(tx_receipt.gasUsed),
            provider=self.provider_type,
        )

    def verify_hash(self, data_hash: str) -> bool:
        """Verify whether a given SHA-256 hash exists in any mined transaction on-chain.

        Scans the blockchain blocks and transaction data payloads to independently
        confirm immutable presence without relying on off-chain caches.

        Args:
            data_hash: 64-character hex string to search on-chain.

        Returns:
            True if the exact hash is found on the blockchain, False otherwise.
        """
        clean_hash = data_hash.lower().strip()
        if clean_hash.startswith("0x"):
            clean_hash = clean_hash[2:]

        current_block = self.w3.eth.block_number

        # Scan all mined blocks from 0 to current_block
        for b_num in range(current_block + 1):
            block = self.w3.eth.get_block(b_num, full_transactions=True)
            transactions = block.get("transactions", [])
            for tx in transactions:
                raw_data = tx.get("data") if hasattr(tx, "get") else getattr(tx, "data", None)
                if raw_data is None:
                    raw_data = tx.get("input") if hasattr(tx, "get") else getattr(tx, "input", None)

                if raw_data is None:
                    continue

                if isinstance(raw_data, HexBytes):
                    tx_hex = raw_data.hex().lower()
                elif hasattr(raw_data, "hex"):
                    tx_hex = raw_data.hex().lower()
                else:
                    tx_hex = str(raw_data).lower()

                if tx_hex.startswith("0x"):
                    tx_hex = tx_hex[2:]

                if tx_hex == clean_hash:
                    tx_id = tx.get("hash", "")
                    tx_id_str = tx_id.hex() if hasattr(tx_id, "hex") else str(tx_id)
                    logger.info(f"Hash {clean_hash} verified on-chain in block {b_num} (tx: {tx_id_str})")
                    return True

        logger.warning(f"Hash {clean_hash} NOT found in any blockchain transaction.")
        return False

    def get_transaction_details(self, tx_hash_hex: str) -> Optional[Dict[str, Any]]:
        """Retrieve mined transaction and receipt details by transaction hash."""
        try:
            tx = self.w3.eth.get_transaction(tx_hash_hex)
            receipt = self.w3.eth.get_transaction_receipt(tx_hash_hex)
            block = self.w3.eth.get_block(receipt.blockNumber)
            return {
                "transaction_hash": tx_hash_hex,
                "block_number": receipt.blockNumber,
                "block_timestamp": block.timestamp,
                "from": tx["from"],
                "to": tx.get("to"),
                "gas_used": receipt.gasUsed,
                "status": receipt.status,
                "input_data": tx["input"].hex() if hasattr(tx["input"], "hex") else str(tx["input"]),
            }
        except Exception as e:
            logger.debug(f"Error fetching tx details for {tx_hash_hex}: {e}")
            return None
