"""Face detection, validation, primary face selection, and deep biometric embedding extraction.

Uses OpenCV's native deep neural network models:
- YuNet (FaceDetectorYN) for high-accuracy face and landmark detection.
- SFace (FaceRecognizerSF) for 128-dimensional deep facial biometric embeddings.
Strictly rejects non-face images without any heuristic or synthetic contrast fallbacks.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("facechain.face_processor")


class NoFaceDetectedError(ValueError):
    """Raised when no genuine human face is detected in an image."""
    pass


@dataclass
class FaceDetectionResult:
    """Structured result of genuine face detection and deep embedding extraction."""

    faces_detected: int
    detection_score: float
    embedding_dimension: int
    embedding: np.ndarray
    bounding_box: Tuple[int, int, int, int]  # (x, y, w, h)
    landmarks: Optional[np.ndarray] = None  # 5 facial landmarks

    def to_dict(self, include_embedding: bool = False) -> Dict[str, Any]:
        """Convert to dictionary without exposing raw embedding unless explicitly requested."""
        data = {
            "faces_detected": self.faces_detected,
            "detection_score": round(float(self.detection_score), 4),
            "embedding_dimension": self.embedding_dimension,
            "bounding_box": list(self.bounding_box),
        }
        if include_embedding:
            data["embedding"] = self.embedding.tolist()
        return data


class FaceProcessor:
    """Performs genuine deep face detection (YuNet) and biometric feature extraction (SFace)."""

    DEFAULT_YUNET_PATH = Path("models/face_detection_yunet.onnx")
    DEFAULT_SFACE_PATH = Path("models/face_recognition_sface.onnx")

    def __init__(
        self,
        yunet_path: Optional[Union[str, Path]] = None,
        sface_path: Optional[Union[str, Path]] = None,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
    ):
        self.yunet_path = Path(yunet_path) if yunet_path else self.DEFAULT_YUNET_PATH
        self.sface_path = Path(sface_path) if sface_path else self.DEFAULT_SFACE_PATH
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

        self._detector: Optional[Any] = None
        self._recognizer: Optional[Any] = None

        self._init_models()

    def _init_models(self) -> None:
        """Initialize OpenCV FaceDetectorYN and FaceRecognizerSF from ONNX models."""
        missing = []
        if not self.yunet_path.is_file():
            missing.append(f"YuNet model: {self.yunet_path}")
        if not self.sface_path.is_file():
            missing.append(f"SFace model: {self.sface_path}")

        if missing:
            err_msg = (
                "DEEP LEARNING FACE MODELS REQUIRED:\n"
                + "\n".join(f"  ❌ Missing {m}" for m in missing)
                + "\n\nPlease download official OpenCV Zoo models by running:\n"
                "    python scripts/download_models.py\n"
                "Or manually place the ONNX models into the models/ folder:\n"
                "    models/face_detection_yunet.onnx\n"
                "    models/face_recognition_sface.onnx\n"
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        try:
            self._detector = cv2.FaceDetectorYN.create(
                model=str(self.yunet_path.resolve()),
                config="",
                input_size=(320, 320),
                score_threshold=self.score_threshold,
                nms_threshold=self.nms_threshold,
                top_k=5000,
            )
            self._recognizer = cv2.FaceRecognizerSF.create(
                model=str(self.sface_path.resolve()),
                config="",
            )
            logger.info("Initialized YuNet Face Detector and SFace Face Recognizer successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load OpenCV ONNX face models: {e}")

    def load_image(self, image_source: Union[str, Path, bytes, np.ndarray, Image.Image]) -> np.ndarray:
        """Load and validate an image into a standard BGR numpy array.

        Args:
            image_source: File path, raw bytes, OpenCV BGR ndarray, or PIL Image.

        Returns:
            Standard BGR OpenCV image array.

        Raises:
            FileNotFoundError: If path does not exist.
            ValueError: If image format is corrupt or unsupported.
        """
        if isinstance(image_source, np.ndarray):
            if image_source.size == 0:
                raise ValueError("Provided image array is empty.")
            return image_source

        if isinstance(image_source, Image.Image):
            rgb_arr = np.array(image_source.convert("RGB"))
            return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

        if isinstance(image_source, bytes):
            nparr = np.frombuffer(image_source, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image bytes. Corrupt or unsupported format.")
            return img

        # Path-like input
        path = Path(image_source)
        if not path.is_file():
            raise FileNotFoundError(f"Input image file not found: {path}")

        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to read image file: {path}. Unsupported or corrupt format.")
        return img

    def process_face(self, image_source: Union[str, Path, bytes, np.ndarray, Image.Image]) -> FaceDetectionResult:
        """Detect human faces in an image, select the primary face, and return deep embedding.

        Strictly rejects non-face images by raising NoFaceDetectedError.

        Args:
            image_source: Image path, bytes, or ndarray.

        Returns:
            FaceDetectionResult with detection confidence and normalized 128-d embedding.

        Raises:
            NoFaceDetectedError: If no genuine face is detected.
        """
        img_bgr = self.load_image(image_source)
        h, w = img_bgr.shape[:2]

        if h < 20 or w < 20:
            raise NoFaceDetectedError("Image dimensions too small for face detection.")

        # Dynamically set input size for the YuNet detector
        self._detector.setInputSize((w, h))
        ret, faces = self._detector.detect(img_bgr)

        if faces is None or len(faces) == 0:
            raise NoFaceDetectedError("No face detected in the provided image.")

        # Select primary face: largest bounding box area & highest detection score
        # Face layout in YuNet: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rc, y_rc, x_lc, y_lc, score]
        primary_face = max(
            faces,
            key=lambda f: float(max(0, f[2]) * max(0, f[3])) * float(f[14]),
        )

        bbox = (
            int(primary_face[0]),
            int(primary_face[1]),
            int(primary_face[2]),
            int(primary_face[3]),
        )
        score = float(primary_face[14])
        landmarks = primary_face[4:14].copy()

        # Align and crop face using SFace recognizer
        aligned_face = self._recognizer.alignCrop(img_bgr, primary_face)

        # Extract 128-d deep facial biometric feature vector
        raw_feature = self._recognizer.feature(aligned_face)
        embedding = raw_feature.flatten().astype(np.float32)

        # L2 unit normalization for exact cosine similarity comparison
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return FaceDetectionResult(
            faces_detected=len(faces),
            detection_score=score,
            embedding_dimension=len(embedding),
            embedding=embedding,
            bounding_box=bbox,
            landmarks=landmarks,
        )


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity between two face embeddings.

    Args:
        emb1: First embedding vector (1D numpy array).
        emb2: Second embedding vector (1D numpy array).

    Returns:
        Cosine similarity score in range [-1.0, 1.0].
    """
    v1 = np.asarray(emb1, dtype=np.float32).flatten()
    v2 = np.asarray(emb2, dtype=np.float32).flatten()

    if len(v1) != len(v2):
        raise ValueError(
            f"Embedding dimension mismatch: {len(v1)} vs {len(v2)}"
        )

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    dot = np.dot(v1, v2)
    similarity = dot / (norm1 * norm2)
    return float(np.clip(similarity, -1.0, 1.0))
