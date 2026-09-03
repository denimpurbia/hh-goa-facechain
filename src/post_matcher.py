"""Post matcher and face similarity verification module.

Downloads discovered candidate public images, extracts facial embeddings,
computes cosine similarity against the query face, and ranks candidates.
"""

from dataclasses import dataclass
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
from PIL import Image

from src.face_processor import FaceProcessor, compute_cosine_similarity
from src.hashing import calculate_sha256
from src.reverse_search import SearchCandidate

logger = logging.getLogger("facechain.post_matcher")

# Known public social and media domains to prioritize
PREFERRED_DOMAINS = {
    "twitter.com",
    "x.com",
    "instagram.com",
    "linkedin.com",
    "facebook.com",
    "youtube.com",
    "tiktok.com",
    "github.com",
    "reddit.com",
    "pinterest.com",
    "medium.com",
    "flickr.com",
    "unsplash.com",
    "wikimedia.org",
    "wikipedia.org",
}


@dataclass
class CandidateEvaluation:
    """Evaluation summary for an individual search candidate."""

    candidate: SearchCandidate
    similarity_score: float
    candidate_image_sha256: str
    is_match: bool
    face_detected: bool
    error_message: Optional[str] = None


@dataclass
class MatchResult:
    """Overall result of candidate matching process."""

    best_candidate: Optional[SearchCandidate]
    similarity_score: float
    threshold: float
    is_match: bool
    candidate_image_sha256: Optional[str]
    candidate_image_bytes: Optional[bytes]
    evaluations: List[CandidateEvaluation]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize match result."""
        return {
            "is_match": self.is_match,
            "similarity_score": round(self.similarity_score, 4),
            "threshold": round(self.threshold, 4),
            "candidate_image_sha256": self.candidate_image_sha256,
            "best_candidate": self.best_candidate.to_dict() if self.best_candidate else None,
            "evaluations_count": len(self.evaluations),
        }


class PostMatcher:
    """Coordinates candidate image downloading, face analysis, and ranking."""

    def __init__(self, face_processor: FaceProcessor, timeout: int = 15, max_download_size_mb: int = 10):
        self.face_processor = face_processor
        self.timeout = timeout
        self.max_download_size = max_download_size_mb * 1024 * 1024
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36 FaceChainVerify/1.0"
            )
        })

    def download_image(self, url: str) -> Optional[bytes]:
        """Safely download candidate image with size limits and error handling.

        Supports HTTP/HTTPS URLs, file:// URIs, and local file paths.

        Args:
            url: Candidate image URL or URI.

        Returns:
            Image bytes if successful, None otherwise.
        """
        if not url:
            return None

        # Support local file path or file:// URI (for offline testing & fixtures)
        if url.startswith("file://"):
            try:
                from urllib.request import url2pathname
                from urllib.parse import urlparse
                parsed_path = url2pathname(urlparse(url).path)
                # On Windows, fix leading slash if present (e.g. /D:/ -> D:/)
                if len(parsed_path) > 3 and parsed_path[0] == "\\" and parsed_path[2] == ":":
                    parsed_path = parsed_path[1:]
                local_p = Path(parsed_path)
                if local_p.is_file():
                    return local_p.read_bytes()
            except Exception as e:
                logger.debug(f"Error loading file URI {url}: {e}")

        if Path(url).is_file():
            try:
                return Path(url).read_bytes()
            except Exception:
                pass

        if not (url.startswith("http://") or url.startswith("https://")):
            return None

        try:
            with self.session.get(url, stream=True, timeout=self.timeout) as resp:
                if resp.status_code != 200:
                    logger.debug(f"Failed to download image from {url}: HTTP {resp.status_code}")
                    return None

                content_type = resp.headers.get("Content-Type", "").lower()
                # Accept common image content types
                if content_type and not any(t in content_type for t in ["image", "octet-stream"]):
                    logger.debug(f"Skipping non-image content type ({content_type}) for {url}")
                    return None

                content = bytearray()
                for chunk in resp.iter_content(chunk_size=32 * 1024):
                    content.extend(chunk)
                    if len(content) > self.max_download_size:
                        logger.warning(f"Image from {url} exceeded max size limit ({self.max_download_size} bytes)")
                        return None

                # Verify image can be parsed by PIL
                try:
                    img = Image.open(io.BytesIO(content))
                    img.verify()
                    return bytes(content)
                except Exception as img_err:
                    logger.debug(f"Corrupt image from {url}: {img_err}")
                    return None

        except requests.exceptions.RequestException as e:
            logger.debug(f"Request error downloading {url}: {e}")
            return None

    def evaluate_candidates(
        self,
        candidates: List[SearchCandidate],
        input_embedding: Any,
        threshold: float = 0.60,
    ) -> MatchResult:
        """Download candidate images, detect faces, compute similarities, and rank.

        Args:
            candidates: List of discovered public search candidates.
            input_embedding: 1D numpy array of the input query face embedding.
            threshold: Cosine similarity threshold for considering a match.

        Returns:
            MatchResult instance containing best verified candidate and all evaluations.
        """
        if not candidates:
            return MatchResult(
                best_candidate=None,
                similarity_score=0.0,
                threshold=threshold,
                is_match=False,
                candidate_image_sha256=None,
                candidate_image_bytes=None,
                evaluations=[],
            )

        # Re-order candidates to slightly prioritize recognized social/media domains
        sorted_candidates = sorted(
            candidates,
            key=lambda c: (
                1 if c.source_domain in PREFERRED_DOMAINS else 0,
                -c.raw_rank,
            ),
            reverse=True,
        )

        evaluations: List[CandidateEvaluation] = []
        best_candidate: Optional[SearchCandidate] = None
        best_similarity = -1.0
        best_image_sha256: Optional[str] = None
        best_image_bytes: Optional[bytes] = None

        for candidate in sorted_candidates:
            target_image_url = candidate.image_url or candidate.thumbnail_url
            if not target_image_url:
                evaluations.append(
                    CandidateEvaluation(
                        candidate=candidate,
                        similarity_score=0.0,
                        candidate_image_sha256="",
                        is_match=False,
                        face_detected=False,
                        error_message="No image or thumbnail URL available",
                    )
                )
                continue

            # Download candidate image
            image_bytes = self.download_image(target_image_url)
            if not image_bytes:
                evaluations.append(
                    CandidateEvaluation(
                        candidate=candidate,
                        similarity_score=0.0,
                        candidate_image_sha256="",
                        is_match=False,
                        face_detected=False,
                        error_message="Image download failed or unsupported image data",
                    )
                )
                continue

            img_sha256 = calculate_sha256(image_bytes)

            try:
                cand_face = self.face_processor.process_face(image_bytes)
                similarity = compute_cosine_similarity(input_embedding, cand_face.embedding)
                is_match = similarity >= threshold

                eval_record = CandidateEvaluation(
                    candidate=candidate,
                    similarity_score=similarity,
                    candidate_image_sha256=img_sha256,
                    is_match=is_match,
                    face_detected=True,
                    error_message=None,
                )
                evaluations.append(eval_record)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_candidate = candidate
                    best_image_sha256 = img_sha256
                    best_image_bytes = image_bytes

            except ValueError as val_err:
                evaluations.append(
                    CandidateEvaluation(
                        candidate=candidate,
                        similarity_score=0.0,
                        candidate_image_sha256=img_sha256,
                        is_match=False,
                        face_detected=False,
                        error_message=f"Face analysis failed: {str(val_err)}",
                    )
                )

        is_match = bool(best_similarity >= threshold and best_candidate is not None)
        final_score = max(0.0, best_similarity) if best_candidate else 0.0

        return MatchResult(
            best_candidate=best_candidate if is_match else (best_candidate if final_score > 0 else None),
            similarity_score=final_score,
            threshold=threshold,
            is_match=is_match,
            candidate_image_sha256=best_image_sha256 if is_match else None,
            candidate_image_bytes=best_image_bytes if is_match else None,
            evaluations=evaluations,
        )
