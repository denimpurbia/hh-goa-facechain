"""Convenience demo script to execute an end-to-end verification run."""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.main import run_pipeline


def main() -> None:
    sample_image = ROOT_DIR / "input" / "sample_face.jpg"
    if not sample_image.exists():
        from scripts.create_sample_face import create_sample_face_image
        create_sample_face_image(sample_image)

    print("Running FaceChain Verify Demo Pipeline with local sample face...")
    exit_code = run_pipeline(
        input_image_path=str(sample_image),
        threshold=0.60,
        output_path="results/verification_result.json",
        use_demo_fixture=True,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
