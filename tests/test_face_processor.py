"""Unit tests for FaceProcessor and cosine similarity computation."""

import numpy as np
import pytest
from src.face_processor import FaceProcessor, compute_cosine_similarity
from scripts.create_sample_face import create_sample_face_image


@pytest.fixture
def face_image_path(tmp_path):
    """Generate a valid synthetic face image for testing."""
    img_path = tmp_path / "test_face.jpg"
    create_sample_face_image(img_path)
    return img_path


def test_face_processor_detection_and_embedding(face_image_path):
    """Verify face detection, embedding dimension, and normalization."""
    processor = FaceProcessor()
    result = processor.process_face(face_image_path)

    assert result.faces_detected >= 1
    assert result.detection_score > 0.5
    assert result.embedding_dimension == 512
    assert isinstance(result.embedding, np.ndarray)

    # Embedding should be unit normalized
    norm = np.linalg.norm(result.embedding)
    assert pytest.approx(norm, 0.01) == 1.0


def test_face_processor_no_face_error(tmp_path):
    """Verify that a blank image raises a ValueError (no face detected)."""
    blank_img = tmp_path / "blank.jpg"
    # Create pure black image with no facial features
    import cv2
    cv2.imwrite(str(blank_img), np.zeros((200, 200, 3), dtype=np.uint8))

    processor = FaceProcessor()
    with pytest.raises(ValueError, match="No face detected"):
        processor.process_face(blank_img)


def test_compute_cosine_similarity_identical_vectors():
    """Verify cosine similarity is 1.0 for identical normalized vectors."""
    vec = np.random.randn(512).astype(np.float32)
    vec = vec / np.linalg.norm(vec)

    similarity = compute_cosine_similarity(vec, vec)
    assert pytest.approx(similarity, 0.0001) == 1.0


def test_compute_cosine_similarity_orthogonal_vectors():
    """Verify cosine similarity is 0.0 for orthogonal vectors."""
    vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    similarity = compute_cosine_similarity(vec1, vec2)
    assert pytest.approx(similarity, 0.0001) == 0.0


def test_compute_cosine_similarity_dimension_mismatch():
    """Verify error on mismatched dimensions."""
    vec1 = np.zeros(512)
    vec2 = np.zeros(256)

    with pytest.raises(ValueError, match="dimension mismatch"):
        compute_cosine_similarity(vec1, vec2)
