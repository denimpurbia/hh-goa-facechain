"""Command Line Interface formatting and argument parser for FaceChain Verify."""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Optional

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def parse_arguments() -> argparse.Namespace:
    """Parse and return CLI command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="facechain",
        description="FaceChain Verify: From Face Discovery to Tamper-Evident Verification (HH Goa 2026).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main --input input/face.jpg
  python -m src.main --input input/face.jpg --threshold 0.70
  python -m src.main --input input/face.jpg --demo-fixture
        """,
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to the input query face image (JPG, PNG, WebP).",
    )

    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=None,
        help="Face cosine similarity threshold (default: from .env or 0.60).",
    )

    parser.add_argument(
        "--provider", "-p",
        type=str,
        default=None,
        help="Reverse image search provider (default: serpapi).",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results/verification_result.json",
        help="Path to save the JSON verification results (default: results/verification_result.json).",
    )

    parser.add_argument(
        "--skip-blockchain",
        action="store_true",
        help="Skip recording fingerprint onto the Ethereum blockchain.",
    )

    parser.add_argument(
        "--demo-fixture",
        action="store_true",
        help="Use a clearly-labelled local demo fixture for offline demonstration and testing.",
    )

    return parser.parse_args()


class TerminalFormatter:
    """Formats clean, professional terminal output for the 8-step pipeline."""

    DIVIDER = "=" * 50

    @classmethod
    def print_banner(cls) -> None:
        """Print the project header banner."""
        print(cls.DIVIDER)
        print("FACECHAIN VERIFY")
        print("HH GOA 2026")
        print(cls.DIVIDER)

    @classmethod
    def step(cls, step_num: int, total_steps: int, title: str) -> None:
        """Print step header."""
        print(f"\n[{step_num}/{total_steps}] {title}")

    @classmethod
    def success(cls, message: str) -> None:
        """Print success indicator."""
        try:
            print(f"✓ {message}")
        except UnicodeEncodeError:
            print(f"[OK] {message}")

    @classmethod
    def failure(cls, message: str) -> None:
        """Print failure indicator."""
        try:
            print(f"❌ {message}")
        except UnicodeEncodeError:
            print(f"[FAIL] {message}")

    @classmethod
    def info(cls, key: str, value: Any) -> None:
        """Print indented key-value pair."""
        print(f"  {key}: {value}")

    @classmethod
    def print_footer_success(cls) -> None:
        """Print successful pipeline conclusion."""
        print("\n" + cls.DIVIDER)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print(cls.DIVIDER)

    @classmethod
    def print_footer_no_match(cls) -> None:
        """Print no match conclusion."""
        print("\n" + cls.DIVIDER)
        print("NO VERIFIED MATCH FOUND")
        print(cls.DIVIDER)

    @classmethod
    def print_footer_error(cls, message: str) -> None:
        """Print pipeline termination on error."""
        print("\n" + cls.DIVIDER)
        print(f"PIPELINE TERMINATED: {message}")
        print(cls.DIVIDER)
