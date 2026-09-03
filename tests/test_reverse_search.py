"""Unit tests for reverse image search providers, Google Lens upload flow, and error handling."""

from pathlib import Path
from unittest.mock import MagicMock, patch
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


def test_serpapi_google_lens_mocked_flow(tmp_path):
    """Verify upload_image -> search_with_image_id -> parse_results pipeline with mocks."""
    provider = SerpApiProvider(api_key="mock_key", timeout=10)
    fake_img = tmp_path / "test_face.jpg"
    fake_img.write_bytes(b"mock_image_bytes")

    upload_mock_resp = MagicMock()
    upload_mock_resp.status_code = 200
    upload_mock_resp.json.return_value = {"image_id": "mock_img_id_12345", "status": "success"}

    search_mock_resp = MagicMock()
    search_mock_resp.status_code = 200
    search_mock_resp.json.return_value = {
        "visual_matches": [
            {
                "title": "Public Photo Match",
                "link": "https://twitter.com/example/status/987654",
                "image": "https://example.com/face_highres.jpg",
                "thumbnail": "https://example.com/thumb.jpg",
                "source": "Twitter",
                "position": 1,
            }
        ],
        "exact_matches": [
            {
                "title": "Exact Profile Match",
                "link": "https://linkedin.com/in/example_person",
                "image": "https://example.com/profile.jpg",
                "thumbnail": "https://example.com/profile_thumb.jpg",
                "source": "LinkedIn",
                "position": 1,
            }
        ],
    }

    with patch("requests.post", return_value=upload_mock_resp) as mock_post, \
         patch("requests.get", return_value=search_mock_resp) as mock_get:

        candidates = provider.search(fake_img, max_results=5)

        # Verify upload POST called with SerpApi Image endpoint
        assert mock_post.call_count == 1
        assert "serpapi.com/image" in mock_post.call_args[0][0]
        assert mock_post.call_args[1]["params"]["api_key"] == "mock_key"

        # Verify Google Lens GET called with image_id
        assert mock_get.call_count == 1
        assert mock_get.call_args[1]["params"]["engine"] == "google_lens"
        assert mock_get.call_args[1]["params"]["image_id"] == "mock_img_id_12345"
        assert mock_get.call_args[1]["params"]["api_key"] == "mock_key"

        # Verify parsed candidates
        assert len(candidates) == 2
        assert candidates[0].url == "https://linkedin.com/in/example_person"
        assert candidates[0].source_domain == "linkedin.com"
        assert candidates[0].provider == "google_lens"

        assert candidates[1].url == "https://twitter.com/example/status/987654"
        assert candidates[1].source_domain == "twitter.com"


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
