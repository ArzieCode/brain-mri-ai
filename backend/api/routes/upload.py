"""Upload route — accepts MRI image files."""

import uuid
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from loguru import logger

from backend.core.config import settings
from backend.api.schemas.schemas import UploadResponse, ErrorResponse

router = APIRouter()


@router.post("/", response_model=UploadResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    """
    Accept an image upload and save it to the uploads directory.
    Returns a file_id for subsequent prediction requests.
    """
    file_id = str(uuid.uuid4())
    suffix = Path(file.filename or "image.jpg").suffix.lower()
    save_name = f"{file_id}{suffix}"
    save_path = Path(settings.UPLOADS_DIR) / save_name

    # Read file into memory first to check size
    content = await file.read()
    size_bytes = len(content)

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if size_bytes > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_bytes / 1024 / 1024:.1f} MB). Max: {settings.MAX_UPLOAD_SIZE_MB} MB",
        )

    # Save to disk
    with open(save_path, "wb") as f:
        f.write(content)

    logger.info(f"Uploaded: {save_name} ({size_bytes} bytes)")

    return UploadResponse(
        file_id=file_id,
        filename=file.filename or save_name,
        file_path=f"/uploads/{save_name}",
        size_bytes=size_bytes,
    )
