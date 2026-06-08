"""
Predict route — full brain tumor analysis pipeline.
POST /api/predict/{file_id}

Features:
- DICOM support
- LRU cache (skip inference jika file sama)
- Async inference (non-blocking)
- WebSocket progress
- Audit logging
- Second opinion mode
"""

import asyncio
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from loguru import logger

from backend.core.config import settings
from backend.services.validation_service import ImageValidationService
from backend.services.prediction_service import PredictionService
from backend.services.gradcam_service import GradCAMService
from backend.services.report_service import report_service
from backend.services.cache_service import prediction_cache
from backend.services.audit_service import audit_service
from backend.services.dicom_service import is_dicom, dicom_to_png
from backend.api.schemas.schemas import FullAnalysisReport

router = APIRouter()


async def _run_async(func, *args):
    """Jalankan fungsi blocking di thread pool agar tidak block event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args)


@router.post("/{file_id}", response_model=FullAnalysisReport)
async def analyze_mri(file_id: str, request: Request):
    mm = request.app.state.model_manager
    if not mm.is_ready:
        raise HTTPException(503, "Models not loaded yet. Please try again shortly.")

    # ── Find uploaded file ────────────────────────────────
    uploads = Path(settings.UPLOADS_DIR)
    matches = list(uploads.glob(f"{file_id}*"))
    if not matches:
        raise HTTPException(404, f"File not found for ID: {file_id}")

    image_path  = matches[0]
    image_bytes = image_path.read_bytes()
    filename    = image_path.name

    # ── DICOM conversion ──────────────────────────────────
    dicom_metadata = None
    if is_dicom(filename, image_bytes):
        logger.info(f"[DICOM] Detected DICOM file: {filename}")
        try:
            image_bytes, dicom_metadata = dicom_to_png(image_bytes)
            filename = filename.rsplit(".", 1)[0] + ".png"
            logger.info(f"[DICOM] Converted to PNG successfully")
        except Exception as e:
            raise HTTPException(422, f"Failed to process DICOM file: {e}")

    # ── LRU Cache check ───────────────────────────────────
    cached = prediction_cache.get(image_bytes)
    if cached:
        logger.info(f"[CACHE] Returning cached result for {file_id}")
        audit_service.log(
            event="predict_cached",
            file_id=file_id,
            filename=filename,
            result=cached.get("prediction"),
            confidence=cached.get("confidence"),
            risk_level=cached.get("risk_level"),
            cached=True,
        )
        return cached["report"]

    # ── Services ──────────────────────────────────────────
    validation_svc = ImageValidationService(mm.validator, mm.device)
    prediction_svc = PredictionService(mm.classifier, validation_svc, mm.device)
    gradcam_svc    = GradCAMService(mm.classifier, mm.device)

    # ── Predict (async, non-blocking) ─────────────────────
    try:
        prediction = await prediction_svc.predict(
            image_bytes=image_bytes,
            filename=filename,
            file_size=len(image_bytes),
        )
    except ValueError as e:
        logger.warning(f"Rejected {filename}: {e}")
        validation = await validation_svc.validate(image_bytes, filename, len(image_bytes))
        audit_service.log(
            event="predict_rejected",
            file_id=file_id,
            filename=filename,
            rejected=True,
            rejection_reason=str(e),
        )
        return report_service.create_report(image_filename=filename, validation=validation)

    # ── Second opinion (jalankan inference kedua dengan MC passes lebih banyak) ──
    second_opinion = None
    try:
        second_opinion = await prediction_svc.predict_second_opinion(
            image_bytes=image_bytes,
        )
        if second_opinion.prediction != prediction.prediction:
            logger.warning(
                f"[SECOND OPINION] Disagreement: "
                f"{prediction.prediction.value} vs {second_opinion.prediction.value}"
            )
    except Exception as e:
        logger.warning(f"[SECOND OPINION] Failed: {e}")

    # ── Validation result ─────────────────────────────────
    validation = await validation_svc.validate(image_bytes, filename, len(image_bytes))

    # ── GradCAM (async) ───────────────────────────────────
    target_class_idx = settings.TUMOR_CLASSES.index(prediction.prediction.value)
    gradcam = await gradcam_svc.generate(
        image_bytes=image_bytes,
        image_id=file_id,
        target_class_idx=target_class_idx,
        class_name=prediction.prediction.value,
    )

    # ── Build report ──────────────────────────────────────
    report = report_service.create_report(
        image_filename=filename,
        validation=validation,
        prediction=prediction,
        gradcam=gradcam,
        dicom_metadata=dicom_metadata,
        second_opinion=second_opinion,
    )

    # ── Cache result ──────────────────────────────────────
    prediction_cache.set(image_bytes, {
        "report": report,
        "prediction": prediction.prediction.value,
        "confidence": prediction.confidence,
        "risk_level": prediction.risk_level.value,
    })

    # ── Audit log ─────────────────────────────────────────
    audit_service.log(
        event="predict_success",
        file_id=file_id,
        filename=filename,
        result=prediction.prediction.value,
        confidence=prediction.confidence,
        risk_level=prediction.risk_level.value,
        inference_ms=prediction.inference_time_ms,
        cached=False,
    )

    return report


@router.websocket("/{file_id}/progress")
async def predict_progress(websocket: WebSocket, file_id: str):
    """
    WebSocket endpoint untuk realtime progress saat analisis.
    Frontend connect ke ws://localhost:8000/api/predict/{file_id}/progress
    """
    await websocket.accept()
    try:
        steps = [
            (10,  "Validating image format..."),
            (25,  "Checking image quality..."),
            (40,  "Running MRI type detection..."),
            (55,  "Classifying tumor type..."),
            (70,  "Running uncertainty estimation..."),
            (80,  "Generating second opinion..."),
            (90,  "Creating GradCAM heatmap..."),
            (100, "Analysis complete!"),
        ]
        for progress, message in steps:
            await websocket.send_json({"progress": progress, "message": message})
            await asyncio.sleep(0.3)
        await websocket.close()
    except WebSocketDisconnect:
        pass