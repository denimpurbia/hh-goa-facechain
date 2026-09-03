"""Unit tests for PostMatcher strict candidate evaluation and non-face rejection."""

import cv2
import numpy as np
import pytest
from scripts.create_sample_face import create_sample_face_image
from src.face_processor import FaceProcessor
from src.post_matcher import PostMatcher
from src.reverse_search import SearchCandidate


@pytest.fixture
def sample_face_bytes(tmp_path):
    """Generate sample face bytes."""
    path = tmp_path / "face.jpg"
    create_sample_face_image(path)
    return path.read_bytes()


@pytest.fixture
def non_face_image_bytes(tmp_path):
    """Generate non-face noise/graphic image bytes."""
    path = tmp_path / "shirt_graphic.jpg"
    np.random.seed(100)
    noise = np.random.randint(0, 255, (250, 250, 3), dtype=np.uint8)
    cv2.imwrite(str(path), noise)
    return path.read_bytes()


def test_post_matcher_rejects_non_face_candidate(tmp_path, sample_face_bytes, non_face_image_bytes):
    """Verify that a candidate image without a face is rejected with face_detected=False."""
    face_proc = FaceProcessor()
    query_result = face_proc.process_face(sample_face_bytes)

    shirt_path = tmp_path / "shirt.jpg"
    shirt_path.write_bytes(non_face_image_bytes)

    candidate = SearchCandidate(
        title="T-Shirt Product",
        url="https://shop.example.com/product/1",
        source_domain="shop.example.com",
        image_url=str(shirt_path.as_uri()),
        thumbnail_url=str(shirt_path.as_uri()),
        provider="google_lens",
        raw_rank=1,
    )

    matcher = PostMatcher(face_processor=face_proc)
    result = matcher.evaluate_candidates(
        candidates=[candidate],
        input_embedding=query_result.embedding,
        threshold=0.40,
    )

    assert result.is_match is False
    assert result.best_candidate is None
    assert len(result.evaluations) == 1
    assert result.evaluations[0].face_detected is False
    assert "no face detected" in result.evaluations[0].error_message


def test_post_matcher_accepts_matching_face_candidate(tmp_path, sample_face_bytes):
    """Verify that a candidate image with matching face passes threshold."""
    face_proc = FaceProcessor()
    query_result = face_proc.process_face(sample_face_bytes)

    face_cand_path = tmp_path / "matching_face.jpg"
    face_cand_path.write_bytes(sample_face_bytes)

    candidate = SearchCandidate(
        title="Verified Social Profile",
        url="https://twitter.com/user/photo/1",
        source_domain="twitter.com",
        image_url=str(face_cand_path.as_uri()),
        thumbnail_url=str(face_cand_path.as_uri()),
        provider="google_lens",
        raw_rank=1,
    )

    matcher = PostMatcher(face_processor=face_proc)
    result = matcher.evaluate_candidates(
        candidates=[candidate],
        input_embedding=query_result.embedding,
        threshold=0.40,
    )

    assert result.is_match is True
    assert result.best_candidate is not None
    assert result.best_candidate.url == "https://twitter.com/user/photo/1"
    assert result.similarity_score >= 0.99
    assert result.evaluations[0].face_detected is True
