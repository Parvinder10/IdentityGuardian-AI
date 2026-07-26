# IdentityGuardian AI: Self-Evolving Multimodal Identity Verification Platform

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

IdentityGuardian AI is a research-grade, production-ready identity verification (KYC) platform. It provides document layout parsing, face matching (ID photo to webcam selfie), forgery checking (face swaps and text alterations), uncertainty calibration, and a continual learning engine to support self-evolving updates.

---

## Technical Features
1. **Multimodal Face Verification**: A Siamese network that extracts embeddings from the ID photo (located by the layout parser) and matches them against a webcam selfie crop using cosine similarity.
2. **Identity Forgery Scan**: Identifies presentation attacks, including face-swap splices and localized text editing.
3. **Calibrated Uncertainty**: Softmax temperature scaling and split conformal prediction intervals mapping biometrics verification margins to avoid false acceptances (FAR).
4. **Self-Healing Fallbacks**: Automatically triggers visual enhancement (LANCZOS super-resolution, contrast scaling) and agent cross-validation checks when text sequence entropy exceeds confidence limits.
5. **Self-Evolving Continual Learning**: Identifies borderline face matches via active learning and computes weight updates using Elastic Weight Consolidation (EWC) diagonal Fisher regularizations to prevent catastrophic forgetting.

---

## Repository Architecture
```
├── configs/                         # Configurations
│   ├── data_config.yaml             # Identity synthesis parameters
│   ├── model_config.yaml            # Hyperparameters for face, forgery, and layout nets
│   └── train_config.yaml            # Continual learning configurations
├── src/                             # Source Package
│   ├── data/                        # ID generation and tampering simulations
│   ├── models/                      # Siamese networks, EWC learner, and document parsers
│   ├── inference/                   # Calibrations, conformal bounds, self-healing engine
│   ├── retrieval/                   # Spatial chunking and Identity Knowledge Graphs
│   ├── evaluation/                  # FAR/FRR metrics and benchmarks
│   └── failure_analysis/            # Integrated Gradients and failure clustering
├── tests/                           # Unit testing suite
├── main.py                          # FastAPI endpoint router
├── Dockerfile                       # Container deployment definition
└── docker-compose.yml               # Service orchestrations (App, Redis, Prometheus)
```

---

## Installation & Setup

### Local Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run pytest suite:
   ```bash
   pytest tests/
   ```

### Docker Deployment
1. Build and boot the microservice stack:
   ```bash
   docker-compose up --build
   ```
2. The interactive API documentation will be available at `http://localhost:8000/docs`.

---

## API Endpoints

### 1. Multimodal Verification
- **Endpoint**: `/verify` (POST)
- **Parameters**: `id_card` (file), `selfie` (file)
- **Response**: Extracted fields, face similarity score, FAR/FRR boundaries, forgery status, and active learning indicators.

### 2. Feature Importance Visualizer
- **Endpoint**: `/explain` (POST)
- **Parameters**: `id_card` (file), `selfie` (file)
- **Response**: Base64 encoded image overlay mapping pixel attributions computed via Integrated Gradients.

### 3. Self-Evolving Training Step
- **Endpoint**: `/continual_train` (POST)
- **Response**: Model rehearsal status, loss, and EWC penalty metrics.
