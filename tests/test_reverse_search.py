"""Unit tests for reverse image search providers and error handling."""

from pathlib import Path
import pytest
from src.config import AppConfig
from src.reverse_search import (
    DemoFixtureSearchProvider,
    SerpApiProvider,
    get_search_provider,
)


def test_serpapi_missing_key_error(tmp_path):
    """Verify that missing SerpApi key raises clear configuration required message."""
    provider = SerpApiProvider(api_key=None)
    fake_img = tmp_path / "img.jpg"
    fake_img.write_bytes(b"dummy_image_data")

    with pytest.raises(RuntimeError, match="SEARCH CONFIGURATION REQUIRED"):
        provider.search(fake_img)


def test_demo_fixture_provider(tmp_path):
    """Verify demo fixture search returns structured candidates without external calls."""
    fixture_items = [
        {
            "title": "Public Profile Post",
            "url": "https://twitter.com/test/status/1",
            "source_domain": "twitter.com",
            "image_url": "https://images.example.com/face.jpg",
            "thumbnail_url": "https://images.example.com/thumb.jpg",
        }
    ]
    provider = DemoFixtureSearchProvider(fixture_candidates=fixture_items)
    fake_img = tmp_path / "img.jpg"
    fake_img.write_bytes(b"dummy")

    results = provider.search(fake_img, max_results=5)
    assert len(results) == 1
    assert results[0].provider == "demo_fixture"
    assert results[0].source_domain == "twitter.com"
    assert results[0].url == "https://twitter.com/test/status/1"


def test_get_search_provider_factory():
    """Verify provider factory returns appropriate provider instance."""
    config = AppConfig(
        serpapi_api_key="dummy_key",
        search_provider="serpapi",
        search_timeout=10,
        max_search_results=5,
        face_match_threshold=0.60,
        blockchain_provider="ethereum_tester",
        blockchain_rpc_url=None,
        log_level="INFO",
        root_dir=Path("."),
        results_dir=Path("./results"),
        input_dir=Path("./input"),
    )

    serpapi_p = get_search_provider(config, use_fixture=False)
    assert isinstance(serpapi_p, SerpApiProvider)

    fixture_p = get_search_provider(config, use_fixture=True)
    assert isinstance(fixture_p, DemoFixtureSearchProvider)
