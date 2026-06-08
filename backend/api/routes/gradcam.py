"""GradCAM route — generate explanations for a given file + class."""

from pathlib import Path
from fastapi import APIRouter, HTTPException, Request, Query
from backend.core.config import settings
from backend.services.gradcam_service import GradCAMService
from backend.api.schemas.schemas import GradCAMResult

router = APIRouter()


@router.post("/{file_id}", response_model=GradCAMResult)
async def generate_gradcam(
    file_id: str,
    request: Request,
    class_idx: int = Query(0, ge=0, le=3, description="Target class index (0=glioma,1=meningioma,2=pituitary,3=normal)"),
):
    mm = request.app.state.model_manager
    if not mm.is_ready:
        raise HTTPException(503, "Models not ready")

    uploads = Path(settings.UPLOADS_DIR)
    matches = list(uploads.glob(f"{file_id}*"))
    if not matches:
        raise HTTPException(404, f"File not found: {file_id}")

    image_bytes = matches[0].read_bytes()
    class_name = settings.TUMOR_CLASSES[class_idx]

    svc = GradCAMService(mm.classifier, mm.device)
    return await svc.generate(image_bytes, file_id, class_idx, class_name)
