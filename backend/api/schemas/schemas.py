"""
API Schemas — Enhanced
=======================
Tambahan: DicomMetadata, SecondOpinion, AuditEntry, ComparisonResult
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


# ── Enums ─────────────────────────────────────────────────────

class TumorClass(str, Enum):
    GLIOMA     = "glioma"
    MENINGIOMA = "meningioma"
    PITUITARY  = "pituitary"
    NORMAL     = "normal"


class ImageValidityStatus(str, Enum):
    VALID_BRAIN_MRI        = "valid_brain_mri"
    INVALID_WRONG_MODALITY = "invalid_wrong_modality"
    INVALID_NOT_MEDICAL    = "invalid_not_medical"
    INVALID_LOW_QUALITY    = "invalid_low_quality"
    INVALID_CORRUPTED      = "invalid_corrupted"
    UNCERTAIN              = "uncertain"


class UncertaintyLevel(str, Enum):
    LOW      = "low"
    MODERATE = "moderate"
    HIGH     = "high"


class RiskLevel(str, Enum):
    LOW      = "low"
    MODERATE = "moderate"
    HIGH     = "high"


# ── Quality ───────────────────────────────────────────────────

class ImageQualityReport(BaseModel):
    is_acceptable:          bool
    width:                  int
    height:                 int
    aspect_ratio:           float
    blur_score:             float
    mean_brightness:        float
    is_blurry:              bool
    is_too_dark:            bool
    is_too_small:           bool
    has_invalid_aspect_ratio: bool
    rejection_reason:       Optional[str] = None


# ── Validation ────────────────────────────────────────────────

class ValidationResult(BaseModel):
    is_valid_brain_mri: bool
    status:             ImageValidityStatus
    detected_type:      str
    type_confidence:    float = Field(ge=0.0, le=1.0)
    quality_report:     Optional[ImageQualityReport] = None
    rejection_message:  Optional[str] = None


# ── OOD ───────────────────────────────────────────────────────

class OODResult(BaseModel):
    is_in_distribution:     bool
    entropy:                float
    max_softmax_confidence: float
    uncertainty_level:      UncertaintyLevel
    ood_message:            Optional[str] = None


# ── Prediction ────────────────────────────────────────────────

class ClassProbability(BaseModel):
    class_name:  str
    probability: float = Field(ge=0.0, le=1.0)
    percentage:  float = Field(ge=0.0, le=100.0)


class UncertaintyEstimate(BaseModel):
    mean_confidence:          float
    std_confidence:           float
    confidence_interval_low:  float
    confidence_interval_high: float
    uncertainty_level:        UncertaintyLevel
    mc_passes:                int


class PredictionResult(BaseModel):
    prediction:          TumorClass
    confidence:          float = Field(ge=0.0, le=1.0)
    class_probabilities: List[ClassProbability]
    uncertainty:         UncertaintyEstimate
    ood_result:          OODResult
    inference_time_ms:   float
    risk_level:          RiskLevel
    clinical_notes:      str


# ── GradCAM ───────────────────────────────────────────────────

class GradCAMResult(BaseModel):
    original_image_url: str
    heatmap_url:        str
    overlay_url:        str
    attention_region:   str
    explanation:        str


# ── DICOM Metadata ────────────────────────────────────────────

class DicomMetadata(BaseModel):
    patient_id:        Optional[str] = None
    study_date:        Optional[str] = None
    modality:          Optional[str] = None
    manufacturer:      Optional[str] = None
    scanner_model:     Optional[str] = None
    field_strength:    Optional[str] = None
    slice_thickness:   Optional[str] = None
    repetition_time:   Optional[str] = None
    echo_time:         Optional[str] = None
    study_description: Optional[str] = None
    series_description: Optional[str] = None
    pixel_spacing:     Optional[str] = None
    rows:              Optional[str] = None
    columns:           Optional[str] = None


# ── Second Opinion ────────────────────────────────────────────

class SecondOpinionResult(BaseModel):
    prediction:       TumorClass
    confidence:       float
    agrees_with_primary: bool
    disagreement_note:   Optional[str] = None
    inference_time_ms:   float


# ── Comparison ────────────────────────────────────────────────

class ComparisonResult(BaseModel):
    scan_a_report_id: str
    scan_b_report_id: str
    scan_a_prediction: str
    scan_b_prediction: str
    scan_a_confidence: float
    scan_b_confidence: float
    changed:           bool
    change_note:       Optional[str] = None


# ── Full Report ───────────────────────────────────────────────

class FullAnalysisReport(BaseModel):
    report_id:      str
    image_filename: str
    timestamp:      datetime
    validation:     ValidationResult
    prediction:     Optional[PredictionResult] = None
    gradcam:        Optional[GradCAMResult]    = None
    dicom_metadata: Optional[DicomMetadata]   = None
    second_opinion: Optional[SecondOpinionResult] = None
    model_version:  str = "1.0.0"
    disclaimer:     str = (
        "AI-Assisted Analysis Only. This report is generated by an AI system "
        "and is NOT a substitute for professional medical diagnosis. "
        "Always consult a qualified radiologist or neurologist."
    )


# ── Upload ────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    file_id:    str
    filename:   str
    file_path:  str
    size_bytes: int
    is_dicom:   bool = False
    message:    str = "Image uploaded successfully"


# ── Audit ─────────────────────────────────────────────────────

class AuditEntry(BaseModel):
    id:               str
    timestamp:        str
    event:            str
    file_id:          str
    filename:         str
    result:           Optional[str]   = None
    confidence:       Optional[float] = None
    risk_level:       Optional[str]   = None
    rejected:         bool = False
    rejection_reason: Optional[str]   = None
    inference_ms:     Optional[float] = None
    cached:           bool = False


class AuditStats(BaseModel):
    total_analyses:     int
    total_predictions:  int
    total_rejections:   int
    rejection_rate:     float
    class_distribution: Dict[str, int]
    avg_confidence:     float


# ── Error ─────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error:  str
    detail: Optional[str] = None
    code:   Optional[str] = None