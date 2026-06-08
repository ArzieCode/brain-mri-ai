"""
Image Validation Service — Enhanced
=====================================
6-layer validation pipeline for brain MRI images.
"""

import io
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from loguru import logger
from torchvision import transforms
from scipy import stats as scipy_stats

from backend.core.config import settings
from backend.api.schemas.schemas import (
    ImageQualityReport,
    ValidationResult,
    ImageValidityStatus,
    OODResult,
    UncertaintyLevel,
)

# ── Constants ─────────────────────────────────────────────────────────────────

MRI_BRIGHTNESS_MAX  = 200
MRI_STD_MIN         = 25.0
MRI_STD_MAX         = 110.0
MRI_KURTOSIS_MIN    = -0.5
MRI_ENTROPY_MIN     = 5.5
MRI_DARK_RATIO_MIN  = 0.20
MRI_DARK_RATIO_MAX  = 0.90
MRI_BRIGHT_BLOB_MIN = 0.05
MRI_BRIGHT_BLOB_MAX = 0.75
OOD_MIN_CONFIDENCE  = 0.55
OOD_MAX_ENTROPY     = 1.10
DL_REJECT_MIN_CONF  = 0.50   # harus >= 50% yakin baru ditolak
DL_ACCEPT_MIN_CONF  = 0.40   # minimum untuk diterima sebagai brain_mri

VALIDATION_TRANSFORM = transforms.Compose([
    transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _to_gray(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)


# ── Statistical Checker ───────────────────────────────────────────────────────

class MRIStatisticalChecker:

    def check(self, gray: np.ndarray) -> tuple[bool, str | None]:
        flat = gray.flatten().astype(np.float32)
        mean_brightness = float(gray.mean())
        std_brightness  = float(gray.std())

        if mean_brightness > MRI_BRIGHTNESS_MAX:
            return False, f"Image too bright for MRI (mean={mean_brightness:.1f})"
        if std_brightness < MRI_STD_MIN:
            return False, f"Image lacks tissue contrast (std={std_brightness:.1f})"
        if std_brightness > MRI_STD_MAX:
            return False, f"Contrast too high — likely a natural photo (std={std_brightness:.1f})"

        dark_ratio = float((gray < 15).sum()) / flat.size
        if dark_ratio < MRI_DARK_RATIO_MIN:
            return False, f"Insufficient dark background ({dark_ratio:.2%})"
        if dark_ratio > MRI_DARK_RATIO_MAX:
            return False, f"Image nearly blank (dark ratio {dark_ratio:.2%})"

        bright_ratio = float((gray > 50).sum()) / flat.size
        if bright_ratio < MRI_BRIGHT_BLOB_MIN:
            return False, "Too little bright tissue detected"
        if bright_ratio > MRI_BRIGHT_BLOB_MAX:
            return False, f"Too much bright area ({bright_ratio:.2%}) — likely a photograph"

        kurt = float(scipy_stats.kurtosis(flat))
        if kurt < MRI_KURTOSIS_MIN:
            return False, f"Pixel distribution inconsistent with MRI (kurtosis={kurt:.2f})"

        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist / (hist.sum() + 1e-10)
        entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))
        if entropy < MRI_ENTROPY_MIN:
            return False, f"Low image texture entropy ({entropy:.2f}) — too uniform"

        # Extra: reject jika gambar terlalu berwarna (foto selfie/cermin)
        return True, None


# ── Anatomical Checker ────────────────────────────────────────────────────────

class MRIAnatomicalChecker:

    def check(self, gray: np.ndarray) -> tuple[bool, str | None]:
        h, w = gray.shape
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, "No tissue region detected"

        largest    = max(contours, key=cv2.contourArea)
        area_ratio = cv2.contourArea(largest) / (h * w)

        if area_ratio < 0.05:
            return False, f"Brain region too small ({area_ratio:.1%})"
        if area_ratio > 0.85:
            return False, f"Bright region fills image ({area_ratio:.1%})"

        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            if not (0.20 * w < cx < 0.80 * w and 0.15 * h < cy < 0.85 * h):
                return False, f"Brain centroid off-center (cx={cx/w:.2f}, cy={cy/h:.2f})"

        left  = gray[:, :w // 2]
        right = np.fliplr(gray[:, w // 2:])
        mw    = min(left.shape[1], right.shape[1])
        sym   = np.abs(left[:, :mw].astype(float) - right[:, :mw].astype(float)).mean()

        if sym < 5.0:
            return False, f"Image perfectly symmetric ({sym:.2f}) — possibly synthetic"
        if sym > 100.0:
            return False, f"Left-right asymmetry too extreme ({sym:.2f})"

        return True, None


# ── Color Saturation Checker (deteksi foto/selfie/cermin) ────────────────────

class ColorSaturationChecker:
    """
    Foto selfie, cermin, atau gambar berwarna punya saturasi tinggi.
    Brain MRI hampir selalu grayscale atau low-saturation.
    """

    def check(self, pil_img: Image.Image) -> tuple[bool, str | None]:
        hsv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1].astype(float)
        mean_sat = saturation.mean()
        high_sat_ratio = float((saturation > 50).sum()) / saturation.size

        if mean_sat > 60:
            return False, (
                f"Image has high color saturation (mean={mean_sat:.1f}) — "
                "brain MRI scans are grayscale or low-saturation"
            )
        if high_sat_ratio > 0.30:
            return False, (
                f"Too many colorful pixels ({high_sat_ratio:.1%}) — "
                "this does not appear to be a medical scan"
            )
        return True, None


# ── Main Validation Service ───────────────────────────────────────────────────

class ImageValidationService:

    def __init__(self, validator_model, device: torch.device):
        self.validator      = validator_model
        self.device         = device
        self._stat_checker  = MRIStatisticalChecker()
        self._anat_checker  = MRIAnatomicalChecker()
        self._color_checker = ColorSaturationChecker()

    def validate_file_format(self, filename: str, file_size_bytes: int) -> str | None:
        suffix = filename.lower().split(".")[-1]
        if f".{suffix}" not in settings.ALLOWED_EXTENSIONS:
            return f"Unsupported format: .{suffix}"
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if file_size_bytes > max_bytes:
            return f"File too large ({file_size_bytes / 1024 / 1024:.1f} MB)"
        return None

    def assess_image_quality(self, image_bytes: bytes) -> ImageQualityReport:
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            return ImageQualityReport(
                is_acceptable=False, width=0, height=0, aspect_ratio=0,
                blur_score=0, mean_brightness=0, is_blurry=False,
                is_too_dark=False, is_too_small=False,
                has_invalid_aspect_ratio=False,
                rejection_reason=f"Cannot open image: {e}",
            )

        w, h         = pil_img.size
        aspect_ratio = w / h if h else 0
        gray         = _to_gray(pil_img)
        blur_score   = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_bright  = float(gray.mean())
        is_too_dark  = mean_bright < settings.DARK_THRESHOLD
        is_too_small = min(w, h) < settings.MIN_IMAGE_DIMENSION
        has_bad_asp  = not (settings.MIN_ASPECT_RATIO <= aspect_ratio <= settings.MAX_ASPECT_RATIO)
        is_blurry    = False
        rejection    = None

        if is_too_small:
            rejection = f"Image too small ({w}x{h})"
        elif has_bad_asp:
            rejection = f"Bad aspect ratio ({aspect_ratio:.2f})"
        elif is_too_dark:
            rejection = "Image too dark"

        return ImageQualityReport(
            is_acceptable=rejection is None,
            width=w, height=h, aspect_ratio=round(aspect_ratio, 3),
            blur_score=round(blur_score, 2), mean_brightness=round(mean_bright, 2),
            is_blurry=is_blurry, is_too_dark=is_too_dark,
            is_too_small=is_too_small, has_invalid_aspect_ratio=has_bad_asp,
            rejection_reason=rejection,
        )

    def assess_mri_statistics(self, image_bytes: bytes) -> tuple[bool, str | None]:
        try:
            gray = _to_gray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        except Exception as e:
            return False, f"Cannot decode image: {e}"
        return self._stat_checker.check(gray)

    def assess_mri_anatomy(self, image_bytes: bytes) -> tuple[bool, str | None]:
        try:
            gray = _to_gray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        except Exception as e:
            return False, f"Cannot decode image: {e}"
        return self._anat_checker.check(gray)

    def assess_color_saturation(self, image_bytes: bytes) -> tuple[bool, str | None]:
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            return False, f"Cannot decode image: {e}"
        return self._color_checker.check(pil_img)

    @torch.no_grad()
    def classify_image_type(self, image_bytes: bytes) -> tuple[str, float]:
        if self.validator is None:
            logger.warning("[VALIDATOR] No DL validator — skipping DL check")
            return "brain_mri", 0.0

        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            return "non_medical", 0.0

        tensor   = VALIDATION_TRANSFORM(pil_img).unsqueeze(0).to(self.device)
        logits   = self.validator(tensor)
        probs    = F.softmax(logits, dim=1)[0]
        idx      = int(probs.argmax())
        detected = settings.VALID_IMAGE_CLASSES[idx]
        conf     = float(probs[idx])

        logger.debug(
            f"[CLASSIFIER] {detected} ({conf:.3f}) | "
            f"{[f'{c}={p:.3f}' for c, p in zip(settings.VALID_IMAGE_CLASSES, probs.tolist())]}"
        )
        return detected, conf

    def compute_entropy(self, probs: torch.Tensor) -> float:
        return float(-(probs * torch.log(probs + 1e-10)).sum())

    def assess_ood(self, probs: torch.Tensor) -> OODResult:
        max_conf = float(probs.max())
        entropy  = self.compute_entropy(probs)
        is_in    = max_conf >= OOD_MIN_CONFIDENCE and entropy <= OOD_MAX_ENTROPY

        if max_conf > 0.80:
            level = UncertaintyLevel.LOW
        elif max_conf > 0.60:
            level = UncertaintyLevel.MODERATE
        else:
            level = UncertaintyLevel.HIGH

        msg = None
        if not is_in:
            msg = (
                f"High entropy ({entropy:.3f}) or low confidence ({max_conf:.3f}) — "
                "image may not match training distribution."
            )

        return OODResult(
            is_in_distribution=is_in,
            entropy=round(entropy, 4),
            max_softmax_confidence=round(max_conf, 4),
            uncertainty_level=level,
            ood_message=msg,
        )

    async def validate(self, image_bytes: bytes, filename: str, file_size: int) -> ValidationResult:

        # Layer 1: Format
        fmt_err = self.validate_file_format(filename, file_size)
        if fmt_err:
            logger.warning(f"[VALIDATE] L1 FAIL: {fmt_err}")
            return ValidationResult(
                is_valid_brain_mri=False, status=ImageValidityStatus.INVALID_CORRUPTED,
                detected_type="unknown", type_confidence=0.0,
                quality_report=None, rejection_message=fmt_err,
            )

        # Layer 2: Quality
        quality = self.assess_image_quality(image_bytes)
        if not quality.is_acceptable:
            logger.warning(f"[VALIDATE] L2 FAIL: {quality.rejection_reason}")
            return ValidationResult(
                is_valid_brain_mri=False, status=ImageValidityStatus.INVALID_LOW_QUALITY,
                detected_type="unknown", type_confidence=0.0,
                quality_report=quality, rejection_message=quality.rejection_reason,
            )

        # Layer 3: Color saturation (deteksi foto/selfie/cermin)
        color_ok, color_reason = self.assess_color_saturation(image_bytes)
        if not color_ok:
            logger.warning(f"[VALIDATE] L3 FAIL (color): {color_reason}")
            return ValidationResult(
                is_valid_brain_mri=False, status=ImageValidityStatus.INVALID_NOT_MEDICAL,
                detected_type="natural_image", type_confidence=0.0,
                quality_report=quality, rejection_message=color_reason,
            )

        # Layer 4: Statistical MRI fingerprint
        stat_ok, stat_reason = self.assess_mri_statistics(image_bytes)
        if not stat_ok:
            logger.warning(f"[VALIDATE] L4 FAIL (stats): {stat_reason}")
            return ValidationResult(
                is_valid_brain_mri=False, status=ImageValidityStatus.INVALID_NOT_MEDICAL,
                detected_type="non_mri", type_confidence=0.0,
                quality_report=quality,
                rejection_message=f"Pixel statistics inconsistent with brain MRI: {stat_reason}",
            )

        # Layer 5: Anatomical plausibility
        anat_ok, anat_reason = self.assess_mri_anatomy(image_bytes)
        if not anat_ok:
            logger.warning(f"[VALIDATE] L5 FAIL (anatomy): {anat_reason}")
            return ValidationResult(
                is_valid_brain_mri=False, status=ImageValidityStatus.INVALID_NOT_MEDICAL,
                detected_type="non_brain_mri", type_confidence=0.0,
                quality_report=quality,
                rejection_message=f"Anatomical check failed: {anat_reason}",
            )

        # Layer 6: DL classifier
        detected_type, conf = self.classify_image_type(image_bytes)

        if conf > 0.0:
            if detected_type != "brain_mri" and conf >= DL_REJECT_MIN_CONF:
                logger.warning(f"[VALIDATE] L6 FAIL (DL confident): {detected_type} ({conf:.3f})")
                return ValidationResult(
                    is_valid_brain_mri=False, status=ImageValidityStatus.INVALID_NOT_MEDICAL,
                    detected_type=detected_type, type_confidence=round(conf, 4),
                    quality_report=quality,
                    rejection_message=(
                        f"DL classifier identified image as '{detected_type}' "
                        f"(confidence {conf:.1%}), not brain MRI"
                    ),
                )
            elif detected_type != "brain_mri" and conf < DL_REJECT_MIN_CONF:
                logger.warning(f"[VALIDATE] L6 WARN (DL uncertain): {detected_type} ({conf:.3f}) — continuing")
            elif detected_type == "brain_mri" and conf < DL_ACCEPT_MIN_CONF:
                return ValidationResult(
                    is_valid_brain_mri=False, status=ImageValidityStatus.UNCERTAIN,
                    detected_type=detected_type, type_confidence=round(conf, 4),
                    quality_report=quality,
                    rejection_message=f"DL classifier uncertain ({conf:.1%}). Please upload a clearer scan.",
                )

        logger.info(f"[VALIDATE] All layers passed — brain_mri confirmed (DL conf={conf:.3f})")
        return ValidationResult(
            is_valid_brain_mri=True, status=ImageValidityStatus.VALID_BRAIN_MRI,
            detected_type="brain_mri", type_confidence=round(conf, 4),
            quality_report=quality, rejection_message=None,
        )