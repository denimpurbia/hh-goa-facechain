"""Script to generate a sample face image in input/sample_face.jpg for testing."""

from pathlib import Path
import cv2
import numpy as np


def create_sample_face_image(output_path: Path) -> None:
    """Generate a clean synthetic portrait image containing a clear face structure."""
    img = np.full((320, 320, 3), (240, 240, 240), dtype=np.uint8)

    # Face Oval (Head)
    center = (160, 160)
    cv2.ellipse(img, center, (70, 95), 0, 0, 360, (190, 210, 240), -1)  # Skin tone
    cv2.ellipse(img, center, (70, 95), 0, 0, 360, (100, 100, 100), 2)   # Outline

    # Hair
    cv2.ellipse(img, (160, 110), (75, 55), 0, 180, 360, (50, 40, 30), -1)

    # Eyes
    cv2.circle(img, (135, 145), 10, (255, 255, 255), -1)
    cv2.circle(img, (185, 145), 10, (255, 255, 255), -1)
    cv2.circle(img, (135, 145), 5, (80, 50, 30), -1)
    cv2.circle(img, (185, 145), 5, (80, 50, 30), -1)

    # Eyebrows
    cv2.line(img, (120, 130), (148, 132), (50, 40, 30), 3)
    cv2.line(img, (172, 132), (200, 130), (50, 40, 30), 3)

    # Nose
    cv2.line(img, (160, 150), (155, 175), (140, 150, 180), 2)
    cv2.line(img, (155, 175), (165, 175), (140, 150, 180), 2)

    # Mouth (Smile)
    cv2.ellipse(img, (160, 200), (22, 12), 0, 0, 180, (80, 80, 180), -1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"Sample test face image created at: {output_path}")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "input" / "sample_face.jpg"
    create_sample_face_image(out)
