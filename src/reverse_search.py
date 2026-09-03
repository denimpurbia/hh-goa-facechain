"""Reverse image search module for FaceChain Verify.

Performs genuine dynamic web / social media search via configurable search providers.
Strictly adheres to real dynamic search: never fakes results, never hardcodes URLs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
import requests
from src.config import AppConfig

logger = logging.getLogger("facechain.reverse_search")


@dataclass
class SearchCandidate:
    """Normalized public web/social search result candidate."""

    title: str
    url: str
    source_domain: str
    image_url: Optional[str]
    thumbnail_url: Optional[str]
    provider: str
    raw_rank: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert candidate to dictionary representation."""
        return {
            "title": self.title,
            "url": self.url,
            "source_domain": self.source_domain,
            "image_url": self.image_url,
            "thumbnail_url": self.thumbnail_url,
            "provider": self.provider,
            "raw_rank": self.raw_rank,
        }


class ReverseImageSearchProvider(ABC):
    """Abstract Base Class for reverse image search providers."""

    @abstractmethod
    def search(
        self,
        image_path: Union_Path_Str,
        max_results: int = 10,
    ) -> List[SearchCandidate]:
        """Perform reverse image search given a local image path or URL.

        Args:
            image_path: Path to the query image.
            max_results: Maximum number of search candidates to return.

        Returns:
            List of SearchCandidate objects.
        """
        pass


# Type alias for path/str
Union_Path_Str = Any


class SerpApiProvider(ReverseImageSearchProvider):
    """Real dynamic reverse image search using SerpApi (Google Reverse Image Engine)."""

    SEARCH_ENDPOINT = "https://serpapi.com/search"

    def __init__(self, api_key: Optional[str], timeout: int = 30):
        self.api_key = api_key.strip() if api_key else None
        self.timeout = timeout

    def _extract_domain(self, url: str) -> str:
        """Extract clean domain name from URL."""
        if not url:
            return ""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc or parsed.path.split("/")[0]
            # Strip port and www prefix
            domain = domain.split(":")[0]
            if domain.startswith("www."):
                domain = domain[4:]
            return domain.lower()
        except Exception:
            return ""

    def search(
        self,
        image_path: Union_Path_Str,
        max_results: int = 10,
    ) -> List[SearchCandidate]:
        """Execute live reverse image search with SerpApi.

        Args:
            image_path: Path to the local input face image.
            max_results: Maximum candidates to parse.

        Returns:
            List of parsed and normalized SearchCandidate instances.

        Raises:
            RuntimeError: If API key is missing or search request fails.
        """
        if not self.api_key:
            raise RuntimeError(
                "SEARCH CONFIGURATION REQUIRED:\n"
                "Missing environment variable: SERPAPI_API_KEY\n"
                "Please obtain an API key from https://serpapi.com and add it to your .env file:\n"
                "  SERPAPI_API_KEY=your_actual_serpapi_key\n"
                "Or provide a demo test fixture explicitly with --demo-fixture for offline testing."
            )

        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Input image for reverse search not found: {path}")

        logger.info(f"Uploading image to SerpApi reverse image search: {path.name}")

        # SerpApi google_reverse_image engine supports direct image upload via multipart/form-data
        params = {
            "engine": "google_reverse_image",
            "api_key": self.api_key,
            "no_cache": "true",
        }

        try:
            with open(path, "rb") as f:
                files = {"image": (path.name, f, "image/jpeg")}
                response = requests.post(
                    self.SEARCH_ENDPOINT,
                    params=params,
                    files=files,
                    timeout=self.timeout,
                )

            if response.status_code != 200:
                error_msg = f"SerpApi HTTP {response.status_code}"
                try:
                    err_json = response.json()
                    if "error" in err_json:
                        error_msg += f": {err_json['error']}"
                except Exception:
                    error_msg += f": {response.text[:200]}"
                raise RuntimeError(f"Search request failed: {error_msg}")

            data = response.json()
            return self._parse_serpapi_results(data, max_results)

        except requests.exceptions.Timeout:
            raise RuntimeError(f"Search request timed out after {self.timeout}s.")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error during reverse image search: {str(e)}")

    def _parse_serpapi_results(
        self, data: Dict[str, Any], max_results: int
    ) -> List[SearchCandidate]:
        """Parse raw SerpApi JSON response into structured search candidates."""
        candidates: List[SearchCandidate] = []
        rank = 1

        # Check inline image results / visually similar / organic web results
        # 1. Image results
        image_results = data.get("image_results", [])
        for item in image_results:
            if len(candidates) >= max_results:
                break
            link = item.get("link", "")
            title = item.get("title", "") or item.get("snippet", "")
            image_url = item.get("original", "") or item.get("thumbnail", "")
            thumb_url = item.get("thumbnail", "")

            if link:
                domain = self._extract_domain(link)
                candidates.append(
                    SearchCandidate(
                        title=title,
                        url=link,
                        source_domain=domain,
                        image_url=image_url,
                        thumbnail_url=thumb_url,
                        provider="serpapi",
                        raw_rank=rank,
                    )
                )
                rank += 1

        # 2. Organic web results
        organic = data.get("organic_results", [])
        for item in organic:
            if len(candidates) >= max_results:
                break
            link = item.get("link", "")
            title = item.get("title", "")
            # Check for thumbnail in organic item
            thumb = item.get("thumbnail", "") or item.get("displayed_link", "")

            if link and not any(c.url == link for c in candidates):
                domain = self._extract_domain(link)
                candidates.append(
                    SearchCandidate(
                        title=title,
                        url=link,
                        source_domain=domain,
                        image_url=thumb if (thumb and thumb.startswith("http")) else None,
                        thumbnail_url=thumb if (thumb and thumb.startswith("http")) else None,
                        provider="serpapi",
                        raw_rank=rank,
                    )
                )
                rank += 1

        logger.info(f"Successfully extracted {len(candidates)} dynamic candidates from SerpApi.")
        return candidates


class DemoFixtureSearchProvider(ReverseImageSearchProvider):
    """Explicitly labelled development/demo test provider for offline testing and CI.

    Must ONLY be used when user explicitly invokes --demo-fixture flag.
    Clearly marks candidates with provider='demo_fixture'.
    """

    def __init__(self, fixture_candidates: Optional[List[Dict[str, Any]]] = None):
        self.fixture_candidates = fixture_candidates or []

    def search(
        self,
        image_path: Union_Path_Str,
        max_results: int = 10,
    ) -> List[SearchCandidate]:
        """Return designated demo fixture candidates for offline verification testing."""
        logger.info("Executing Search using explicitly requested demo fixture provider.")
        results: List[SearchCandidate] = []
        for i, item in enumerate(self.fixture_candidates[:max_results], 1):
            results.append(
                SearchCandidate(
                    title=item.get("title", "Demo Public Post"),
                    url=item.get("url", "https://public-web.example.org/profile/post/101"),
                    source_domain=item.get("source_domain", "public-web.example.org"),
                    image_url=item.get("image_url"),
                    thumbnail_url=item.get("thumbnail_url"),
                    provider="demo_fixture",
                    raw_rank=i,
                )
            )
        return results


def get_search_provider(
    config: AppConfig,
    use_fixture: bool = False,
    fixture_data: Optional[List[Dict[str, Any]]] = None,
) -> ReverseImageSearchProvider:
    """Factory function to instantiate the configured search provider.

    Args:
        config: Application configuration.
        use_fixture: If True, explicitly use DemoFixtureSearchProvider.
        fixture_data: Optional fixture list for testing.

    Returns:
        ReverseImageSearchProvider instance.
    """
    if use_fixture:
        return DemoFixtureSearchProvider(fixture_data)

    provider_name = config.search_provider.lower()
    if provider_name == "serpapi":
        return SerpApiProvider(
            api_key=config.serpapi_api_key,
            timeout=config.search_timeout,
        )
    else:
        raise ValueError(f"Unsupported search provider: {config.search_provider}")
