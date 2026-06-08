# 🧠 NeuroScan AI — Brain MRI Tumor Detection

> **⚠️ Research & Educational Use Only — NOT a medical device — NOT for clinical diagnosis**

A production-style AI platform for brain tumor detection from MRI scans.
Built with PyTorch, FastAPI, and React. Designed for safety, explainability, and responsible AI.

---

## 🏗️ Architecture Overview

```
brain-mri-ai/
├── backend/                    # FastAPI backend
│   ├── main.py                 # App factory + lifespan
│   ├── core/
│   │   ├── config.py           # Pydantic settings
│   │   └── model_manager.py    # Model loading + device selection
│   ├── models/
│   │   ├── classifier.py       # EfficientNet-B0 tumor classifier
│   │   └── validator.py        # MobileNetV3 image validator
│   ├── services/
│   │   ├── validation_service.py  # Quality + type + OOD checks
│   │   ├── prediction_service.py  # MC Dropout inference
│   │   ├── gradcam_service.py     # GradCAM explainability
│   │   └── report_service.py      # Report generation + PDF export
│   └── api/
│       ├── schemas/schemas.py  # Pydantic request/response schemas
│       └── routes/             # upload, validate, predict, gradcam, reports
│
├── frontend/                   # React + Vite + Tailwind
│   └── src/
│       ├── pages/              # Landing, Upload, Results, History, Safety
│       ├── components/         # Layout, UI components
│       └── utils/api.js        # Axios API client
│
├── training/
│   └── scripts/
│       ├── train_classifier.py # EfficientNet-B0 training pipeline
│       └── train_validator.py  # MobileNetV3 training pipeline
│
├── models/                     # Trained model weights (.pth)
├── uploads/                    # Uploaded MRI images
├── outputs/                    # GradCAM visualizations
├── datasets/                   # Training datasets
├── requirements.txt
├── setup.sh
└── README.md
```

---

## 🚀 Quick Start (Apple Silicon M1/M2)

### Automated Setup

```bash
git clone <repo-url>
cd brain-mri-ai
chmod +x setup.sh
./setup.sh
```

### Manual Setup

#### 1. Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

pip install --upgrade pip
```

#### 2. PyTorch for Apple Silicon

```bash
# Apple Silicon (M1/M2) — MPS acceleration
pip install torch torchvision

# CUDA GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
```

#### 3. Python Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Frontend

```bash
cd frontend
npm install
```

---

## 🧬 Dataset Setup

### Tumor Classifier Dataset

Download: [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

Arrange as:
```
datasets/brain_tumor/
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── normal/
└── val/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── normal/
```

### Image Validator Dataset

Collect and arrange:
```
datasets/image_validator/
├── train/
│   ├── brain_mri/      (from tumor dataset)
│   ├── xray/           (NIH ChestX-ray14)
│   ├── blood_cell/     (Blood Cell Images, Kaggle)
│   ├── natural_image/  (Imagenette, fast.ai)
│   └── non_medical/    (random images)
└── val/
    └── ...
```

---

## 🏋️ Training

```bash
source .venv/bin/activate
cd training/scripts

# Train tumor classifier (EfficientNet-B0) — ~30min on M1
python train_classifier.py

# Train image validator (MobileNetV3) — ~15min on M1
python train_validator.py
```

Outputs saved to `models/`:
- `efficientnet_b0_tumor.pth`
- `mobilenet_v3_validator.pth`
- `confusion_matrix.png`
- `training_history.json`

---

## ▶️ Running the Application

### Terminal 1 — Backend

```bash
source .venv/bin/activate
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/api/docs

### Terminal 2 — Frontend

```bash
cd frontend
npm run dev
```

App: http://localhost:5173

---

## 🔬 Analysis Pipeline

```
Upload MRI
    ↓
File format + size check
    ↓
Image quality assessment
  • Blur detection (Laplacian variance)
  • Darkness threshold (mean pixel value)
  • Dimension validation
  • Aspect ratio check
    ↓
Medical image type detection (MobileNetV3)
  • brain_mri ✓ → proceed
  • xray / blood_cell / natural → REJECT
    ↓
Tumor classification (EfficientNet-B0)
    ↓
Monte Carlo Dropout uncertainty (20 passes)
    ↓
OOD detection (entropy + max softmax)
    ↓
GradCAM heatmap generation
    ↓
Structured PDF report
```

---

## 🛡️ Safety Features

| Feature | Implementation |
|---------|---------------|
| Image validation | MobileNetV3-Small classifier (5 classes) |
| Quality checks | Laplacian blur, mean brightness, size/aspect |
| OOD detection | Shannon entropy + max softmax thresholding |
| Uncertainty | Monte Carlo Dropout (20 stochastic passes) |
| Explainability | GradCAM on EfficientNet-B0 last conv block |
| Bias reduction | Class weighting + WeightedRandomSampler |
| Medical disclaimer | On every prediction, every page, every PDF |

---

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | System status |
| `/api/upload/` | POST | Upload MRI image |
| `/api/predict/{file_id}` | POST | Full analysis pipeline |
| `/api/validate/{file_id}` | POST | Validation only |
| `/api/gradcam/{file_id}` | POST | GradCAM only |
| `/api/reports/` | GET | List all reports |
| `/api/reports/{id}` | GET | Get report |
| `/api/reports/{id}/pdf` | GET | Download PDF |
| `/api/reports/{id}` | DELETE | Delete report |

---

## 🍎 Apple Silicon Notes

- **MPS** (Metal Performance Shaders) is auto-detected and used if available
- Use `num_workers=0` in DataLoader on macOS (multiprocessing limitation)
- Mixed precision via `torch.autocast` — enabled on CUDA, native float on MPS
- Expected training throughput: ~80 batches/sec on M1 Pro (batch size 32)

---

## 🔧 Troubleshooting

**Models not found warning:**
```bash
# Check models directory
ls models/
# Models are optional at startup — app works with pretrained backbone weights
# Train models to get tumor-specific weights
```

**MPS not available:**
```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"
# Requires macOS 12.3+ and PyTorch 2.0+
```

**Frontend proxy error:**
```bash
# Ensure backend is running on port 8000 before starting frontend
uvicorn main:app --reload --port 8000
```

**OpenCV import error:**
```bash
pip install opencv-python-headless  # Use headless version on server
```

---

## 📋 Supported Classes

| Class | Description |
|-------|-------------|
| **Glioma** | Glial cell tumor, grades I–IV |
| **Meningioma** | Meningeal tumor, typically benign |
| **Pituitary** | Pituitary adenoma |
| **Normal** | No tumor detected |

---

## ⚠️ Important Disclaimer

This software is provided for **research and educational purposes only**.

- It is NOT a medical device
- It is NOT FDA or CE approved
- It does NOT provide medical advice
- It MUST NOT be used for clinical diagnosis
- Always consult a qualified radiologist or neurologist

---

## 📚 References

- EfficientNet: Tan & Le, 2019 (arXiv:1905.11946)
- GradCAM: Selvaraju et al., 2017 (arXiv:1610.02391)
- MC Dropout: Gal & Ghahramani, 2016 (arXiv:1506.02142)
- MobileNetV3: Howard et al., 2019 (arXiv:1905.02244)
