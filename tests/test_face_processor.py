"""Unit tests for deep FaceProcessor (YuNet + SFace) and cosine similarity computation."""

import cv2
import numpy as np
import pytest
from scripts.create_sample_face import create_sample_face_image
from src.face_processor import (
    FaceProcessor,
    NoFaceDetectedError,
    compute_cosine_similarity,
)


@pytest.fixture
def face_image_path(tmp_path):
    """Generate a valid synthetic face image for testing."""
    img_path = tmp_path / "test_face.jpg"
    create_sample_face_image(img_path)
    return img_path


def test_face_processor_detection_and_embedding(face_image_path):
    """Verify deep face detection, SFace embedding dimension (128-d), and unit normalization."""
    processor = FaceProcessor()
    result = processor.process_face(face_image_path)

    assert result.faces_detected >= 1
    assert result.detection_score > 0.5
    assert result.embedding_dimension == 128
    assert isinstance(result.embedding, np.ndarray)

    # Embedding must be L2 unit normalized
    norm = np.linalg.norm(result.embedding)
    assert pytest.approx(norm, 0.01) == 1.0


def test_face_processor_no_face_error_blank_image(tmp_path):
    """Verify that a blank image raises NoFaceDetectedError."""
    blank_img = tmp_path / "blank.jpg"
    cv2.imwrite(str(blank_img), np.zeros((200, 200, 3), dtype=np.uint8))

    processor = FaceProcessor()
    with pytest.raises(NoFaceDetectedError, match="No face detected"):
        processor.process_face(blank_img)


def test_face_processor_no_face_error_random_graphic(tmp_path):
    """Verify that high-contrast non-face graphics do NOT generate an embedding."""
    graphic_img = tmp_path / "graphic.jpg"
    # Create textured noise with high variance
    np.random.seed(42)
    noise = np.random.randint(0, 255, (250, 250, 3), dtype=np.uint8)
    cv2.imwrite(str(graphic_img), noise)

    processor = FaceProcessor()
    with pytest.raises(NoFaceDetectedError, match="No face detected"):
        processor.process_face(graphic_img)


def test_face_processor_missing_models_raises_error(tmp_path):
    """Verify that missing model files raise a clear configuration RuntimeError."""
    with pytest.raises(RuntimeError, match="DEEP LEARNING FACE MODELS REQUIRED"):
        FaceProcessor(
            yunet_path=tmp_path / "non_existent_yunet.onnx",
            sface_path=tmp_path / "non_existent_sface.onnx",
        )


def test_compute_cosine_similarity_identical_vectors():
    """Verify cosine similarity is 1.0 for identical normalized vectors."""
    vec = np.random.randn(128).astype(np.float32)
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
    vec1 = np.zeros(128)
    vec2 = np.zeros(64)

    with pytest.raises(ValueError, match="dimension mismatch"):
        compute_cosine_similarity(vec1, vec2)
