# FaceChain Verify
> **"From Face Discovery to Tamper-Evident Verification."**  
> *HH Goa 2026 Shortlisting Task 3: Face Identification & Blockchain Verification*

---

## Overview

**FaceChain Verify** is an end-to-end pipeline bridging deep computer vision and Ethereum blockchain technology to identify public web/social content and cryptographically anchor discovery provenance.

Given an input face image, the system:
1. Detects human faces and 5 facial landmarks using **OpenCV YuNet ONNX**.
2. Extracts a **128-dimensional deep facial biometric embedding** using **OpenCV SFace ONNX**.
3. Uploads the local query image to **SerpApi Image API** and searches **Google Lens** dynamically for public web and social posts.
4. Downloads discovered candidate images, validates human face presence, and calculates **cosine similarity**.
5. Rejects non-face images (e.g., shirts, graphics, logos, products) without computing false similarity scores.
6. Constructs canonical verification metadata (no raw biometrics) and computes a **SHA-256 cryptographic fingerprint**.
7. Records the fingerprint on an **Ethereum-compatible blockchain ledger** (`EthereumTesterProvider` / PyEVM).
8. Re-verifies on-chain state and demonstrates instant **cryptographic tamper detection**.

---

## Deep Learning Models & Setup

The project uses official **OpenCV Zoo** ONNX deep learning models:

| Model | File Path | Architecture | Purpose | Size |
| :--- | :--- | :--- | :--- | :--- |
| **YuNet** | `models/face_detection_yunet.onnx` | `cv2.FaceDetectorYN` | Deep face & 5-point landmark detection | 227 KB |
| **SFace** | `models/face_recognition_sface.onnx` | `cv2.FaceRecognizerSF` | 128-d deep facial biometric recognition | 37.8 MB |

### Model Download Command
To download the official ONNX models into `models/`, run:
```powershell
python scripts/download_models.py
```

---

## Installation & Setup

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Download ONNX Models
```powershell
python scripts/download_models.py
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Edit `.env` and set your SerpApi API key:
```ini
SERPAPI_API_KEY=your_actual_serpapi_key_here
FACE_MATCH_THRESHOLD=0.40
SEARCH_PROVIDER=serpapi
SEARCH_TIMEOUT=30
BLOCKCHAIN_PROVIDER=ethereum_tester
```

---

## Running the Project

### Live Web Search & Face Verification
```powershell
python -m src.main --input input/sample_face.jpg
```

### Offline / Local Demo Fixture
```powershell
python -m src.main --input input/sample_face.jpg --demo-fixture
```

### Interactive Tamper Proof Demo
```powershell
python scripts/tamper_demo.py
```

### Run Full Test Suite
```powershell
python -m pytest -v
```

---

## Biometric Matching & Threshold Guide

| Metric | Model | Range | Match Threshold | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Cosine Similarity** | SFace (`cv2.FaceRecognizerSF`) | `[-1.0, 1.0]` | **`0.40`** (Configurable) | Recommended standard benchmark threshold for SFace |

---

## License
MIT License.
