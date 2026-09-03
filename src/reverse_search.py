"""Reverse image search module for FaceChain Verify.

Performs genuine dynamic web / social media search via configurable search providers.
Strictly adheres to real dynamic search: never fakes results, never hardcodes URLs.
Supports local image upload via SerpApi Image API followed by Google Lens search.
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


# Type alias for path/str
Union_Path_Str = Any


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


class SerpApiProvider(ReverseImageSearchProvider):
    """Real dynamic reverse image search using SerpApi Image Upload & Google Lens Engine."""

    UPLOAD_ENDPOINT = "https://serpapi.com/image"
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

    def upload_image(self, image_path: Path) -> str:
        """Upload a local image to SerpApi Image API to obtain a temporary image_id.

        Args:
            image_path: Path to the local input image.

        Returns:
            String image_id returned by SerpApi.

        Raises:
            RuntimeError: If upload fails or image_id is not returned.
        """
        if not self.api_key:
            raise RuntimeError(
                "SEARCH CONFIGURATION REQUIRED:\n"
                "Missing environment variable: SERPAPI_API_KEY\n"
                "Please obtain an API key from https://serpapi.com and add it to your .env file:\n"
                "  SERPAPI_API_KEY=your_actual_serpapi_key\n"
                "Or provide a demo test fixture explicitly with --demo-fixture for offline testing."
            )

        if not image_path.is_file():
            raise FileNotFoundError(f"Input image file not found: {image_path}")

        # Check file size (SerpApi supports up to 500 KB)
        file_size = image_path.stat().st_size
        if file_size > 500 * 1024:
            logger.warning(f"Image size is {file_size / 1024:.1f} KB (SerpApi recommends <= 500 KB).")

        try:
            with open(image_path, "rb") as f:
                files = {"image": (image_path.name, f, "image/jpeg")}
                params = {"api_key": self.api_key}
                response = requests.post(
                    self.UPLOAD_ENDPOINT,
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
                raise RuntimeError(f"Image upload failed: {error_msg}")

            data = response.json()
            image_id = data.get("image_id") or data.get("id")
            if not image_id:
                raise RuntimeError(f"Image upload succeeded but no image_id returned in response: {data}")

            return str(image_id)

        except requests.exceptions.Timeout:
            raise RuntimeError(f"Image upload timed out after {self.timeout}s.")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error during image upload: {str(e)}")

    def search_with_image_id(
        self,
        image_id: str,
        max_results: int = 10,
    ) -> List[SearchCandidate]:
        """Perform Google Lens search using an image_id on SerpApi.

        Args:
            image_id: The image_id returned from SerpApi image upload.
            max_results: Maximum candidates to parse.

        Returns:
            List of parsed SearchCandidate objects.
        """
        params = {
            "engine": "google_lens",
            "image_id": image_id,
            "api_key": self.api_key,
            "no_cache": "true",
        }

        try:
            response = requests.get(
                self.SEARCH_ENDPOINT,
                params=params,
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
                raise RuntimeError(f"Google Lens search failed: {error_msg}")

            data = response.json()
            return self._parse_google_lens_results(data, max_results)

        except requests.exceptions.Timeout:
            raise RuntimeError(f"Google Lens search timed out after {self.timeout}s.")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error during Google Lens search: {str(e)}")

    def _parse_google_lens_results(
        self, data: Dict[str, Any], max_results: int
    ) -> List[SearchCandidate]:
        """Parse raw Google Lens JSON response into structured search candidates."""
        candidates: List[SearchCandidate] = []
        rank = 1

        # Check visual matches and exact matches
        matches = []
        if isinstance(data.get("exact_matches"), list):
            matches.extend(data.get("exact_matches", []))
        if isinstance(data.get("visual_matches"), list):
            matches.extend(data.get("visual_matches", []))

        for item in matches:
            if len(candidates) >= max_results:
                break

            link = item.get("link", "")
            title = item.get("title", "") or item.get("source", "")
            image_url = item.get("image", "") or item.get("thumbnail", "")
            thumbnail_url = item.get("thumbnail", "") or item.get("image", "")
            source = item.get("source", "")
            position = item.get("position", rank)

            if link:
                domain = self._extract_domain(link) or source.lower()
                candidates.append(
                    SearchCandidate(
                        title=title,
                        url=link,
                        source_domain=domain,
                        image_url=image_url if image_url else None,
                        thumbnail_url=thumbnail_url if thumbnail_url else None,
                        provider="google_lens",
                        raw_rank=position if isinstance(position, int) else rank,
                    )
                )
                rank += 1

        logger.info(f"Extracted {len(candidates)} dynamic candidates from Google Lens.")
        return candidates

    def search(
        self,
        image_path: Union_Path_Str,
        max_results: int = 10,
    ) -> List[SearchCandidate]:
        """Execute complete live reverse image search workflow with SerpApi:
        Upload local image -> Obtain image_id -> Query Google Lens -> Parse dynamic candidates.

        Args:
            image_path: Path to the local input face image.
            max_results: Maximum candidates to parse.

        Returns:
            List of parsed and normalized SearchCandidate instances.
        """
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Input image for reverse search not found: {path}")

        print("\n[LIVE SEARCH MODE]")
        print("Uploading local image...")
        image_id = self.upload_image(path)
        print("Image uploaded successfully")

        print("Searching Google Lens...")
        candidates = self.search_with_image_id(image_id, max_results=max_results)
        print(f"Results received: {len(candidates)}")

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
    if provider_name in ("serpapi", "google_lens"):
        return SerpApiProvider(
            api_key=config.serpapi_api_key,
            timeout=config.search_timeout,
        )
    else:
        raise ValueError(f"Unsupported search provider: {config.search_provider}")
