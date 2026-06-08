#!/usr/bin/env bash
# ============================================================
# Brain MRI AI — Setup Script
# Tested on macOS 13+ (Apple Silicon M1/M2) and Ubuntu 22.04
# ============================================================

set -e  # Exit on any error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_MIN="3.10"

echo ""
echo "╔═══════════════════════════════════════════════╗"
echo "║       NeuroScan AI — Setup Wizard             ║"
echo "║       Brain MRI Tumor Detection System        ║"
echo "╚═══════════════════════════════════════════════╝"
echo ""

# ── Check Python ─────────────────────────────────────────
echo "→ Checking Python version..."
if ! command -v python3 &>/dev/null; then
    echo "❌ Python 3 not found. Install from https://python.org"
    exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Found Python $PY_VERSION"

# ── Check Node ────────────────────────────────────────────
echo "→ Checking Node.js..."
if ! command -v node &>/dev/null; then
    echo "❌ Node.js not found. Install from https://nodejs.org (v18+)"
    exit 1
fi
echo "  Found Node $(node --version)"

# ── Create directories ────────────────────────────────────
echo "→ Creating directories..."
mkdir -p "$PROJECT_DIR"/{models,uploads,outputs,datasets/{brain_tumor/{train,val},image_validator/{train,val}}}

# ── Python virtual environment ────────────────────────────
echo "→ Setting up Python virtual environment..."
cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "  Created .venv"
else
    echo "  .venv already exists"
fi

source .venv/bin/activate
echo "  Activated .venv"

# Upgrade pip
pip install --upgrade pip --quiet

# ── Install PyTorch for Apple Silicon ─────────────────────
echo "→ Installing PyTorch (Apple Silicon MPS support)..."
# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ] && [ "$(uname)" = "Darwin" ]; then
    echo "  Detected Apple Silicon — installing MPS-enabled PyTorch..."
    pip install torch torchvision --quiet
    python3 -c "import torch; print(f'  PyTorch {torch.__version__} | MPS: {torch.backends.mps.is_available()}')"
elif command -v nvidia-smi &>/dev/null; then
    echo "  Detected NVIDIA GPU — installing CUDA PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
else
    echo "  No GPU detected — installing CPU PyTorch..."
    pip install torch torchvision --quiet
fi

# ── Install Python dependencies ───────────────────────────
echo "→ Installing Python dependencies..."
pip install -r requirements.txt --quiet
echo "  ✅ Python dependencies installed"

# ── Frontend Setup ────────────────────────────────────────
echo "→ Installing frontend dependencies..."
cd "$PROJECT_DIR/frontend"
npm install --silent
echo "  ✅ Frontend dependencies installed"

# ── Create .env file ──────────────────────────────────────
cd "$PROJECT_DIR/backend"
if [ ! -f ".env" ]; then
cat > .env << 'EOF'
ENVIRONMENT=development
DEBUG=true
HOST=0.0.0.0
PORT=8000
SECRET_KEY=change-this-in-production
EOF
    echo "→ Created backend/.env"
fi

# ── Done ──────────────────────────────────────────────────
echo ""
echo "╔═══════════════════════════════════════════════╗"
echo "║  ✅ Setup complete!                           ║"
echo "╚═══════════════════════════════════════════════╝"
echo ""
echo "NEXT STEPS:"
echo ""
echo "1. Download brain tumor dataset:"
echo "   https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset"
echo "   → Extract to: datasets/brain_tumor/"
echo ""
echo "2. Train the models:"
echo "   source .venv/bin/activate"
echo "   cd training/scripts"
echo "   python train_classifier.py    # EfficientNet-B0"
echo "   python train_validator.py     # MobileNetV3"
echo ""
echo "3. Start the backend:"
echo "   source .venv/bin/activate"
echo "   cd backend"
echo "   uvicorn main:app --reload --port 8000"
echo ""
echo "4. Start the frontend (new terminal):"
echo "   cd frontend"
echo "   npm run dev"
echo ""
echo "5. Open: http://localhost:5173"
echo ""
echo "⚠  For research use only. Not for medical diagnosis."
echo ""
