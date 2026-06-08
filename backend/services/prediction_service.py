"""
Prediction Service
==================
Core inference pipeline for brain tumor classification.
"""

import time
import io
from typing import List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from loguru import logger
from torchvision import transforms

from backend.core.config import settings
from backend.services.validation_service import ImageValidationService
from backend.api.schemas.schemas import (
    PredictionResult, ClassProbability, UncertaintyEstimate,
    UncertaintyLevel, TumorClass, RiskLevel,
)

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

CLINICAL_NOTES = {
    TumorClass.GLIOMA: (
        "Gliomas originate from glial cells and represent the most common type of "
        "primary brain tumor. They vary widely in grade (I-IV), with grade IV "
        "(glioblastoma) being the most aggressive. Always confirm with a radiologist."
    ),
    TumorClass.MENINGIOMA: (
        "Meningiomas arise from the meninges and are typically benign (grade I). "
        "They often appear as well-defined, homogeneously enhancing masses. "
        "Many are incidentally found and may only require observation."
    ),
    TumorClass.PITUITARY: (
        "Pituitary tumors (adenomas) arise from the pituitary gland. "
        "Microadenomas (<10mm) may cause hormonal symptoms. "
        "Most are benign and treatable. Clinical correlation is essential."
    ),
    TumorClass.NORMAL: (
        "No tumor pattern detected in this scan. Normal MRI findings include "
        "symmetric brain structures and no abnormal signal enhancement. "
        "A normal AI result still warrants clinical correlation."
    ),
}


class PredictionService:

    def __init__(
        self,
        classifier_model,
        validation_service: ImageValidationService,
        device: torch.device,
    ):
        self.classifier         = classifier_model
        self.device             = device
        self.validation_service = validation_service

    def _preprocess(self, image_bytes: bytes) -> torch.Tensor:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor  = INFERENCE_TRANSFORM(pil_img)
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def _single_forward(self, tensor: torch.Tensor) -> np.ndarray:
        self.classifier.eval()
        logits = self.classifier(tensor)
        return F.softmax(logits, dim=1)[0].cpu().numpy()

    def _mc_dropout_passes(
        self, tensor: torch.Tensor, n_passes: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.classifier.enable_mc_dropout()
        all_probs = []
        with torch.no_grad():
            for _ in range(n_passes):
                logits = self.classifier(tensor)
                probs  = F.softmax(logits, dim=1)[0].cpu().numpy()
                all_probs.append(probs)
        self.classifier.disable_mc_dropout()
        arr = np.array(all_probs)
        return arr.mean(axis=0), arr.std(axis=0)

    def _build_class_probabilities(self, probs: np.ndarray) -> List[ClassProbability]:
        result = [
            ClassProbability(
                class_name=settings.TUMOR_CLASSES[i],
                probability=round(float(probs[i]), 4),
                percentage=round(float(probs[i]) * 100, 2),
            )
            for i in range(len(probs))
        ]
        return sorted(result, key=lambda x: x.probability, reverse=True)

    def _build_uncertainty(
        self, mean_probs: np.ndarray, std_probs: np.ndarray
    ) -> UncertaintyEstimate:
        pred_idx  = int(mean_probs.argmax())
        mean_conf = float(mean_probs[pred_idx])
        std_conf  = float(std_probs[pred_idx])
        ci_low    = max(0.0, mean_conf - 2 * std_conf)
        ci_high   = min(1.0, mean_conf + 2 * std_conf)

        if std_conf < 0.05:
            level = UncertaintyLevel.LOW
        elif std_conf < 0.15:
            level = UncertaintyLevel.MODERATE
        else:
            level = UncertaintyLevel.HIGH

        return UncertaintyEstimate(
            mean_confidence=round(mean_conf, 4),
            std_confidence=round(std_conf, 4),
            confidence_interval_low=round(ci_low, 4),
            confidence_interval_high=round(ci_high, 4),
            uncertainty_level=level,
            mc_passes=settings.MC_DROPOUT_PASSES,
        )

    def _determine_risk(
        self, prediction: TumorClass, confidence: float, uncertainty: UncertaintyEstimate
    ) -> RiskLevel:
        if prediction == TumorClass.NORMAL:
            return RiskLevel.MODERATE if uncertainty.uncertainty_level == UncertaintyLevel.HIGH else RiskLevel.LOW
        if confidence > 0.85 and uncertainty.uncertainty_level == UncertaintyLevel.LOW:
            return RiskLevel.HIGH
        elif confidence > 0.65 or uncertainty.uncertainty_level == UncertaintyLevel.MODERATE:
            return RiskLevel.MODERATE
        return RiskLevel.LOW

    async def predict(
        self,
        image_bytes: bytes,
        filename: str = "upload.jpg",
        file_size: int = 0,
    ) -> PredictionResult:
        start = time.time()

        # Validasi dulu
        validation = await self.validation_service.validate(image_bytes, filename, file_size)
        if not validation.is_valid_brain_mri:
            raise ValueError(f"Image rejected: {validation.rejection_message}")

        tensor = self._preprocess(image_bytes)

        # MC Dropout
        mean_probs, std_probs = self._mc_dropout_passes(tensor, settings.MC_DROPOUT_PASSES)

        # OOD
        probs_tensor = torch.tensor(mean_probs, dtype=torch.float32)
        ood_result   = self.validation_service.assess_ood(probs_tensor)

        pred_idx    = int(mean_probs.argmax())
        prediction  = TumorClass(settings.TUMOR_CLASSES[pred_idx])
        confidence  = float(mean_probs[pred_idx])
        class_probs = self._build_class_probabilities(mean_probs)
        uncertainty = self._build_uncertainty(mean_probs, std_probs)
        risk_level  = self._determine_risk(prediction, confidence, uncertainty)
        elapsed_ms  = (time.time() - start) * 1000

        logger.info(
            f"[PREDICT] {prediction.value} | conf={confidence:.3f} "
            f"+/-{float(std_probs[pred_idx]):.3f} | risk={risk_level.value} | {elapsed_ms:.1f}ms"
        )

        return PredictionResult(
            prediction=prediction,
            confidence=round(confidence, 4),
            class_probabilities=class_probs,
            uncertainty=uncertainty,
            ood_result=ood_result,
            inference_time_ms=round(elapsed_ms, 2),
            risk_level=risk_level,
            clinical_notes=CLINICAL_NOTES[prediction],
        )