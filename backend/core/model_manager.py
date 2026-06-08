"""
Model Manager
=============
Centralized model loading, device management, and lifecycle handling.

Supports:
- Apple Silicon MPS acceleration
- CUDA GPU
- CPU fallback

Models:
- EfficientNet-B0 (tumor classifier)
- MobileNetV3-Small (image validator)
"""

import torch
import torch.nn as nn
from pathlib import Path
from loguru import logger

from backend.core.config import settings
from backend.models.classifier import build_tumor_classifier
from backend.models.validator import build_image_validator


class ModelManager:
    """
    Manages all ML model instances for the application.
    Handles device selection, model loading, and thread-safe inference.
    """

    def __init__(self):
        self.device: torch.device = self._select_device()
        self.classifier: nn.Module | None = None
        self.validator: nn.Module | None = None
        self._initialized: bool = False

    def _select_device(self) -> torch.device:
        """
        Select the best available compute device.
        Priority: MPS (Apple Silicon) → CUDA → CPU
        """
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            logger.info("🍎 Apple Silicon MPS detected — using GPU acceleration")
            return torch.device("mps")
        elif torch.cuda.is_available():
            logger.info(f"🟢 CUDA GPU detected: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        else:
            logger.warning("⚠️  No GPU detected — falling back to CPU")
            return torch.device("cpu")

    async def initialize(self):
        """Load all models from disk. Falls back to pretrained weights if no checkpoint found."""
        logger.info(f"Loading models on device: {self.device}")

        # ── Load Tumor Classifier (EfficientNet-B0) ─────────
        classifier_path = Path(settings.MODELS_DIR) / settings.CLASSIFIER_MODEL_PATH
        self.classifier = build_tumor_classifier(
            num_classes=len(settings.TUMOR_CLASSES),
            pretrained=True,
        )

        if classifier_path.exists():
            logger.info(f"📂 Loading classifier weights: {classifier_path}")
            checkpoint = torch.load(classifier_path, map_location=self.device)
            # Support both raw state_dict and wrapped checkpoint formats
            state = checkpoint.get("model_state_dict", checkpoint)
            self.classifier.load_state_dict(state, strict=False)
        else:
            logger.warning(
                "⚠️  Classifier weights not found — using pretrained backbone only. "
                "Run training script to generate weights."
            )

        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()

        # ── Load Image Validator (MobileNetV3-Small) ────────
        validator_path = Path(settings.MODELS_DIR) / settings.VALIDATOR_MODEL_PATH
        self.validator = build_image_validator(
            num_classes=len(settings.VALID_IMAGE_CLASSES),
            pretrained=True,
        )

        if validator_path.exists():
            logger.info(f"📂 Loading validator weights: {validator_path}")
            checkpoint = torch.load(validator_path, map_location=self.device)
            state = checkpoint.get("model_state_dict", checkpoint)
            self.validator.load_state_dict(state, strict=False)
        else:
            logger.warning("⚠️  Validator weights not found — using pretrained backbone only.")

        self.validator = self.validator.to(self.device)
        self.validator.eval()

        self._initialized = True
        logger.info("✅ Model manager initialized")

    async def cleanup(self):
        """Release model resources on shutdown."""
        self.classifier = None
        self.validator = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("Models released from memory")

    @property
    def is_ready(self) -> bool:
        return self._initialized and self.classifier is not None and self.validator is not None
