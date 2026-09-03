# FaceChain Verify
> **"From Face Discovery to Tamper-Evident Verification."**  
> *HH Goa 2026 Shortlisting Task 3: Face Identification & Blockchain Verification*

---

## Overview

**FaceChain Verify** is an end-to-end, production-style pipeline bridging computer vision and Ethereum blockchain technology to identify public web/social content and cryptographically anchor discovery provenance.

Given an input face scan, the system detects and normalizes the primary face, extracts a 512-dimensional facial embedding, performs a genuine dynamic reverse-image search across public web and social platforms, downloads and cross-verifies candidate face matches via cosine similarity, constructs canonical verification metadata, computes a deterministic **SHA-256 cryptographic fingerprint**, records this fingerprint on an **Ethereum-compatible test blockchain**, and executes an independent re-verification and **tamper-detection check**.

---

## Key Architecture & Pipeline Flow

```
+-------------------------------------------------------------+
|                      1. Face Scan Input                     |
|                 (JPG / PNG / WebP image file)               |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|         2. Face Detection & Primary Face Selection          |
|          (InsightFace / OpenCV Universal Detector)          |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|               3. 512-D Face Feature Encoding                |
|               (Normalized facial descriptor)                |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|            4. Genuine Reverse Image Web Search              |
|        (Dynamic SerpApi Google Reverse Search API)          |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|             5. Candidate Image Face Verification            |
|       (Download candidate -> Face Embedding -> Cosine Sim)  |
|                 Threshold: Configurable (0.60)              |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|          6. Canonical Tamper-Evident Metadata               |
|      (Deterministic JSON: RFC 8785 UTF-8, sorted keys)      |
|             *NO RAW BIOMETRIC DATA STORED*                  |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|             7. SHA-256 Cryptographic Fingerprint            |
|              (64-character hex metadata digest)             |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|             8. Ethereum Blockchain Recording                |
|      (EthereumTesterProvider / Smart Contract Mapping)      |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|         9. Independent Re-Verification & Tamper Test        |
|          (Stored Hash == Current Hash on Blockchain?)       |
+-------------------------------------------------------------+
```

---

## Features

- 🔍 **Real Face Detection & Encoding**: Detects faces, selects the primary face (area/confidence-based), and extracts 512-dimensional normalized embeddings with support for **InsightFace** and resilient OpenCV DNN/morphological fallbacks.
- 🌐 **100% Genuine Dynamic Search**: Connects to live search engines (SerpApi Google Reverse Image Search) to discover public web and social posts. **Never fakes results, never hardcodes URLs.**
- ⚖️ **Face Similarity Verification**: Safely downloads candidate images, detects candidate faces, calculates cosine similarity, and filters results against a configurable threshold (`FACE_MATCH_THRESHOLD=0.60`).
- 🔐 **Privacy-Preserving Cryptographic Fingerprint**: Strictly avoids storing raw face images or biometric vectors. Assembles canonical metadata (UTC timestamp, candidate URL, domain, similarity score, image SHA-256) and produces a deterministic 32-byte SHA-256 digest.
- ⛓️ **Ethereum Blockchain Storage**: Uses `Web3.py` and `EthereumTesterProvider` (PyEVM) to anchor metadata fingerprints on an Ethereum ledger with block timestamps and transaction hashes.
- 🛡️ **Cryptographic Tamper Detection**: Independently re-computes metadata digests and compares against on-chain transaction records to immediately catch data alterations.
- 🧪 **Comprehensive Test Suite**: 21 unit tests covering hashing, metadata canonicalization, blockchain state queries, face cosine similarity, and tampering detection.

---

## Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ (Tested on 3.10 – 3.14) | Core application platform |
| **Face Detection & Recognition** | InsightFace / ONNX Runtime / OpenCV | Multi-layer face extraction |
| **Image Processing** | OpenCV (`cv2`), Pillow (`PIL`) | Image decoding, resizing, and validation |
| **Search Provider** | SerpApi (Google Reverse Image Engine) | Live reverse image search API |
| **Blockchain** | `web3.py`, `eth-tester`, `py-evm` | Ethereum virtual machine & local testnet |
| **Smart Contracts** | Solidity (`contracts/FaceVerification.sol`) | On-chain mapping & event tracking |
| **Cryptography** | `hashlib` SHA-256, `hexbytes` | Canonical JSON fingerprinting |
| **Configuration** | `python-dotenv` | Type-safe environment management |
| **Testing** | `pytest` | Automated unit & integration tests |

---

## Project Structure

```
hh-goa-facechain/
│
├── input/
│   ├── README.md                      # Input directory guidelines
│   └── sample_face.jpg                # Out-of-the-box portrait test image
│
├── results/
│   ├── .gitkeep
│   └── verification_result.json       # Generated structured JSON output
│
├── contracts/
│   └── FaceVerification.sol           # Solidity smart contract for verification
│
├── src/
│   ├── __init__.py                    # Package metadata
│   ├── config.py                      # Environment loader & safe secrets masking
│   ├── face_processor.py              # Face detector & 512-d feature extractor
│   ├── reverse_search.py              # Search provider interface (SerpApi & Demo)
│   ├── post_matcher.py                # Candidate downloader & cosine similarity matcher
│   ├── metadata.py                    # Privacy-preserving canonical metadata generator
│   ├── hashing.py                     # SHA-256 & canonical JSON serializer
│   ├── blockchain.py                  # Web3.py & EthereumTesterProvider integration
│   ├── verifier.py                    # Re-verification & tamper detection logic
│   ├── cli.py                         # Clean 8-step terminal formatter & CLI args
│   └── main.py                        # End-to-end pipeline runner
│
├── scripts/
│   ├── create_sample_face.py          # Synthetic face generator for testing
│   ├── run_demo.py                    # One-command full pipeline demonstration
│   └── tamper_demo.py                 # Interactive cryptographic tamper proof demo
│
├── tests/
│   ├── __init__.py
│   ├── test_blockchain.py             # Blockchain storage & query tests
│   ├── test_face_processor.py         # Face embedding & cosine similarity tests
│   ├── test_hashing.py                # SHA-256 & canonicalization tests
│   ├── test_metadata.py               # Metadata privacy & format tests
│   ├── test_reverse_search.py         # Search error handling & provider tests
│   └── test_verifier.py               # Cryptographic tamper detection tests
│
├── .env.example                       # Environment configuration template
├── .gitignore                         # Git exclusion rules
├── requirements.txt                   # Dependency specifications
├── README.md                          # Documentation
└── LICENSE                            # MIT License
```

---

## Installation (Windows / Linux / macOS)

### 1. Clone the repository
```powershell
git clone https://github.com/your-username/hh-goa-facechain.git
cd hh-goa-facechain
```

### 2. Create and activate a Python virtual environment
```powershell
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```powershell
pip install -r requirements.txt
```

---

## Configuration

Copy the sample environment file to `.env`:

```powershell
copy .env.example .env
```

Edit `.env` with your settings:

```env
# Get your API key from https://serpapi.com
SERPAPI_API_KEY=your_serpapi_api_key_here

# Face Matching Configuration (0.0 to 1.0)
FACE_MATCH_THRESHOLD=0.60

# Search Provider Configuration
SEARCH_PROVIDER=serpapi
SEARCH_TIMEOUT=30
MAX_SEARCH_RESULTS=10

# Blockchain Configuration
BLOCKCHAIN_PROVIDER=ethereum_tester
BLOCKCHAIN_RPC_URL=

# Logging Level
LOG_LEVEL=INFO
```

> [!IMPORTANT]
> If `SERPAPI_API_KEY` is not provided in `.env`, the pipeline will safely stop at Step 4 and display `SEARCH CONFIGURATION REQUIRED` with setup instructions. It will **never** fake a search result.

---

## How to Run

### 1. Live Dynamic Search Pipeline (Production Mode)
Requires `SERPAPI_API_KEY` in `.env`:
```powershell
python -m src.main --input input/sample_face.jpg
```

**Custom Options:**
```powershell
# Custom match threshold (e.g. 0.75) and custom output path
python -m src.main --input input/my_photo.jpg --threshold 0.75 --output results/my_result.json

# Skip blockchain broadcasting if running purely for face matching
python -m src.main --input input/sample_face.jpg --skip-blockchain
```

### 2. Offline Demo Pipeline (Fixture Mode)
Run the full 8-step pipeline without an external API key:
```powershell
python -m src.main --input input/sample_face.jpg --demo-fixture
# Or use the convenience demo script:
python scripts/run_demo.py
```

### 3. Cryptographic Tamper Demonstration
Run the standalone tamper demonstration script:
```powershell
python scripts/tamper_demo.py
```

---

## Blockchain Implementation Details

### Technology Stack
- **Provider**: `EthereumTesterProvider` (PyEVM backend) via `Web3.py`.
- **Portability**: Runs locally without requiring MetaMask, Infura keys, or real ETH gas tokens.

### Privacy & Data Integrity Guarantees
1. **Zero Raw Biometric Data On-Chain**: Raw images and facial embedding vectors are **NEVER** stored on the blockchain or in public JSON outputs.
2. **Deterministic Metadata Hashing**: Verification records are serialized using canonical JSON (RFC 8785: sorted keys, no extraneous whitespace, UTF-8 encoded).
3. **Transaction Data Anchoring**: The 32-byte SHA-256 metadata fingerprint is embedded directly into the transaction `data` payload mined on the test blockchain.
4. **Smart Contract**: A Solidity contract (`contracts/FaceVerification.sol`) is provided featuring `storeRecord(bytes32 dataHash)` and `verifyRecord(bytes32 dataHash)` for EVM deployment.

---

## Verification & Tamper Detection Flow

### 1. Canonical Verification Metadata Structure
```json
{
  "candidate_image_sha256": "18017e6b68e4923cebf78f5bd88386f03ac18f0b56fb14c0b46f09d024362ebb",
  "candidate_image_url": "https://public-web.example.org/profile.jpg",
  "input_image_sha256": "18017e6b68e4923cebf78f5bd88386f03ac18f0b56fb14c0b46f09d024362ebb",
  "match_threshold": 0.6,
  "post_title": "Verified Public Profile",
  "search_timestamp": "2026-09-02T12:00:00+00:00",
  "similarity_score": 0.9412,
  "source_domain": "twitter.com",
  "source_post_url": "https://twitter.com/user/status/1789201948"
}
```

### 2. Tamper Detection Mechanism
```
Original Metadata -> SHA-256: 0399491c... -> Recorded in Block #1
Re-Verification:
  Stored Hash   : 0399491c...
  Current Hash  : 0399491c...
  Blockchain    : FOUND
  Result        : [OK] DATA HAS NOT BEEN TAMPERED WITH

Tampered Metadata (e.g. Modified URL) -> SHA-256: 3d2e9cdc...
Re-Verification:
  Expected Hash : 0399491c...
  Current Hash  : 3d2e9cdc...
  Blockchain    : NOT FOUND
  Result        : [FAIL] TAMPERING DETECTED
```

---

## Output Verification Result (`results/verification_result.json`)

```json
{
  "project": "FaceChain Verify",
  "timestamp": "2026-09-02T17:32:58.418843+00:00",
  "face_detection": {
    "faces_detected": 1,
    "detection_score": 0.9843,
    "embedding_dimension": 512
  },
  "search": {
    "provider": "serpapi",
    "results_found": 10
  },
  "match": {
    "found": true,
    "similarity_score": 0.9412,
    "threshold": 0.60,
    "source_url": "https://twitter.com/user/status/1789201948",
    "source_domain": "twitter.com",
    "post_title": "Verified Profile"
  },
  "metadata": { ... },
  "blockchain": {
    "provider": "ethereum_tester",
    "data_hash": "bc2a13ef7cbc36fea90f3da7d6f7fbbcccb953bd2cea4f2962c77a98fe712b0f",
    "transaction_hash": "0xac497dc10c83b4e81172f0e052e8cfe37e4426a49f9b39ca333439956a67556e",
    "block_number": 1,
    "gas_used": 22280
  },
  "verification": {
    "stored_hash": "bc2a13ef7cbc36fea90f3da7d6f7fbbcccb953bd2cea4f2962c77a98fe712b0f",
    "current_hash": "bc2a13ef7cbc36fea90f3da7d6f7fbbcccb953bd2cea4f2962c77a98fe712b0f",
    "blockchain_verified": true,
    "tampering_detected": false,
    "status_message": "DATA HAS NOT BEEN TAMPERED WITH. Cryptographic hash verified on blockchain."
  }
}
```

---

## Testing

Run the automated test suite:

```powershell
python -m pytest -v
```

### Test Coverage Summary
- `tests/test_hashing.py`: SHA-256 determinism, key-order independence, file hashing.
- `tests/test_metadata.py`: Canonicalization, field rounding, zero-biometrics validation.
- `tests/test_blockchain.py`: Transaction broadcasting, on-chain scanning, error handling.
- `tests/test_verifier.py`: Authentic record verification, unauthorized tampering detection.
- `tests/test_face_processor.py`: Face detection, 512-d unit normalization, cosine similarity.
- `tests/test_reverse_search.py`: Missing API key detection, provider factory, demo fixtures.

---

## Privacy & Responsible Use

> [!CAUTION]
> **Ethical Guidelines and Compliance:**
> 1. **Authorized Use Only**: Process only images that you own or for which you have explicit, documented consent.
> 2. **No Surveillance or Tracking**: This project must not be used for unauthorized mass identification, tracking, or stalking.
> 3. **Biometric Privacy**: Never commit or broadcast raw biometric vectors or unhashed face images.

---

## Known Limitations

1. **Search Engine Indexing**: Reverse-image search coverage depends on SerpApi and Google's indexing. Brand-new or private posts may not appear in results.
2. **Image Quality & Angles**: Extreme lighting, low resolution, or heavy facial occlusions may reduce detection confidence or cosine similarity scores.
3. **Local Test Blockchain**: The default `ethereum_tester` provider maintains state within the active session. For persistent multi-node verification, configure `BLOCKCHAIN_RPC_URL` in `.env` to point to an Ethereum testnet (e.g. Sepolia) or local Anvil/Geth node.

---

## License

Released under the [MIT License](LICENSE).  
Copyright (c) 2026 FaceChain Verify Contributors.
