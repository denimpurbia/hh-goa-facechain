"""Configuration module for FaceChain Verify.

Loads environment variables, validates settings, provides safe typed defaults,
and securely masks sensitive credentials.
"""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Find and load .env file from project root
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE)


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration dataclass."""

    # Search Configuration
    serpapi_api_key: Optional[str]
    search_provider: str
    search_timeout: int
    max_search_results: int

    # Face Verification Configuration
    face_match_threshold: float

    # Blockchain Configuration
    blockchain_provider: str
    blockchain_rpc_url: Optional[str]

    # Logging & Paths
    log_level: str
    root_dir: Path
    results_dir: Path
    input_dir: Path

    def has_serpapi_key(self) -> bool:
        """Check if a valid SerpApi API key is configured."""
        return bool(self.serpapi_api_key and self.serpapi_api_key.strip())

    def masked_serpapi_key(self) -> str:
        """Return masked API key for safe logging / display."""
        if not self.serpapi_api_key:
            return "<NOT SET>"
        key = self.serpapi_api_key.strip()
        if len(key) <= 8:
            return "***"
        return f"{key[:4]}...{key[-4:]}"


def load_config() -> AppConfig:
    """Load, validate, and return the application configuration."""
    serpapi_key = os.getenv("SERPAPI_API_KEY", "").strip() or None
    search_provider = os.getenv("SEARCH_PROVIDER", "serpapi").strip().lower()

    try:
        search_timeout = int(os.getenv("SEARCH_TIMEOUT", "30"))
    except ValueError:
        search_timeout = 30

    try:
        max_search_results = int(os.getenv("MAX_SEARCH_RESULTS", "10"))
    except ValueError:
        max_search_results = 10

    try:
        threshold_val = float(os.getenv("FACE_MATCH_THRESHOLD", "0.60"))
        # Clamp threshold between 0.0 and 1.0
        face_match_threshold = max(0.0, min(1.0, threshold_val))
    except ValueError:
        face_match_threshold = 0.60

    blockchain_provider = os.getenv("BLOCKCHAIN_PROVIDER", "ethereum_tester").strip().lower()
    blockchain_rpc_url = os.getenv("BLOCKCHAIN_RPC_URL", "").strip() or None
    log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()

    results_dir = ROOT_DIR / "results"
    input_dir = ROOT_DIR / "input"
    results_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    return AppConfig(
        serpapi_api_key=serpapi_key,
        search_provider=search_provider,
        search_timeout=search_timeout,
        max_search_results=max_search_results,
        face_match_threshold=face_match_threshold,
        blockchain_provider=blockchain_provider,
        blockchain_rpc_url=blockchain_rpc_url,
        log_level=log_level,
        root_dir=ROOT_DIR,
        results_dir=results_dir,
        input_dir=input_dir,
    )


def setup_logger(name: str = "facechain", level: Optional[str] = None) -> logging.Logger:
    """Configure and return a structured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    config_level = level or os.getenv("LOG_LEVEL", "INFO").upper()
    numeric_level = getattr(logging, config_level, logging.INFO)
    logger.setLevel(numeric_level)
    return logger
