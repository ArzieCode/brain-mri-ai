"""Health check route."""

from fastapi import APIRouter, Request
import torch

router = APIRouter()


@router.get("/health")
async def health(request: Request):
    mm = getattr(request.app.state, "model_manager", None)
    device = str(mm.device) if mm else "unknown"
    return {
        "status": "ok",
        "models_ready": mm.is_ready if mm else False,
        "device": device,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }
