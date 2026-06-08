"""
Core Configuration
==================
All application settings via environment variables with safe defaults.
Uses pydantic-settings for type-safe config management.
"""

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Application ────────────────────────────────────────
    APP_NAME: str = "Brain MRI Tumor Detection AI"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    SECRET_KEY: str = "change-this-in-production-brain-mri-secret"

    # ── CORS ────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ]

    # ── File Paths ──────────────────────────────────────────
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    MODELS_DIR: str = str(Path(__file__).resolve().parent.parent.parent / "models")
    UPLOADS_DIR: str = str(Path(__file__).resolve().parent.parent.parent / "uploads")
    OUTPUTS_DIR: str = str(Path(__file__).resolve().parent.parent.parent / "outputs")

    # ── Model Config ────────────────────────────────────────
    # Main tumor classifier
    CLASSIFIER_MODEL_PATH: str = "efficientnet_b0_tumor.pth"
    # Image validity classifier (MRI vs non-MRI)
    VALIDATOR_MODEL_PATH: str = "mobilenet_v3_validator.pth"

    # ── Prediction Config ───────────────────────────────────
    TUMOR_CLASSES: List[str] = ["glioma", "meningioma", "pituitary", "normal"]
    VALID_IMAGE_CLASSES: List[str] = [
        "brain_mri", "xray", "blood_cell", "natural_image", "non_medical"
    ]
    IMAGE_SIZE: int = 224

    # ── Safety Thresholds ───────────────────────────────────
    # Minimum confidence to report a prediction (below = uncertain)
    MIN_CONFIDENCE_THRESHOLD: float = 0.60
    # Maximum entropy allowed (above = OOD / uncertain)
    MAX_ENTROPY_THRESHOLD: float = 1.2
    # Minimum accepted image dimension (pixels)
    MIN_IMAGE_DIMENSION: int = 64
    # Maximum accepted image dimension (pixels)
    MAX_IMAGE_DIMENSION: int = 4096
    # Accepted aspect ratio range
    MIN_ASPECT_RATIO: float = 0.5
    MAX_ASPECT_RATIO: float = 2.0
    # Blur detection threshold (Laplacian variance)
    BLUR_THRESHOLD: float = 80.0
    # Dark image threshold (mean pixel value, 0-255)
    DARK_THRESHOLD: float = 30.0

    # ── Upload Config ───────────────────────────────────────
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".dcm"]

    # ── Monte Carlo Dropout ─────────────────────────────────
    # Number of forward passes for uncertainty estimation
    MC_DROPOUT_PASSES: int = 20

    class Config:
        env_file = ".env"
        case_sensitive = True


# Singleton settings instance
settings = Settings()
