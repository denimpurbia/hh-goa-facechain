"""Download official OpenCV Zoo YuNet face detection and SFace face recognition ONNX models."""

from pathlib import Path
import sys
import urllib.request

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

MODELS = {
    "face_detection_yunet.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def download_models(target_dir: Path = Path("models")) -> None:
    """Download required ONNX models into target directory."""
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading deep learning models into: {target_dir.resolve()}...")
    for model_name, url in MODELS.items():
        dest = target_dir / model_name
        if dest.exists() and dest.stat().st_size > 10000:
            print(f"  [OK] {model_name} already exists ({dest.stat().st_size / 1024:.1f} KB).")
            continue

        print(f"  Downloading {model_name} from {url}...")
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) OpenCV-Model-Downloader"}
            )
            with urllib.request.urlopen(req, timeout=45) as response, open(dest, "wb") as out_file:
                out_file.write(response.read())
            print(f"  [OK] {model_name} downloaded successfully ({dest.stat().st_size / 1024:.1f} KB).")
        except Exception as e:
            print(f"  [ERROR] Failed to download {model_name}: {e}")
            if dest.exists():
                dest.unlink()
            raise


if __name__ == "__main__":
    download_models()
