"""Validate route — standalone image validation without prediction."""

from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from backend.core.config import settings
from backend.services.validation_service import ImageValidationService
from backend.api.schemas.schemas import ValidationResult

router = APIRouter()


@router.post("/{file_id}", response_model=ValidationResult)
async def validate_image(file_id: str, request: Request):
    mm = request.app.state.model_manager
    if not mm.is_ready:
        raise HTTPException(503, "Models not ready")

    uploads = Path(settings.UPLOADS_DIR)
    matches = list(uploads.glob(f"{file_id}*"))
    if not matches:
        raise HTTPException(404, f"File not found: {file_id}")

    image_path = matches[0]
    image_bytes = image_path.read_bytes()

    svc = ImageValidationService(mm.validator, mm.device)
    return await svc.validate(image_bytes, image_path.name, len(image_bytes))
