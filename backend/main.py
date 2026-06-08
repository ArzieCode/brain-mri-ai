"""
Brain MRI Tumor Detection - FastAPI Application Entry Point
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from backend.core.config import settings
from backend.core.model_manager import ModelManager
from backend.api.routes import upload, predict, validate, gradcam, reports, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Brain MRI AI starting up...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    app.state.model_manager = ModelManager()
    await app.state.model_manager.initialize()
    logger.info("All models loaded successfully")
    yield
    logger.info("Shutting down Brain MRI AI...")
    await app.state.model_manager.cleanup()


def create_application() -> FastAPI:
    application = FastAPI(
        title="Brain MRI Tumor Detection API",
        description=(
            "AI-powered brain tumor detection from MRI scans. "
            "For research and educational purposes only. "
            "NOT a replacement for medical diagnosis."
        ),
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        process_time = (time.time() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
        return response

    Path(settings.UPLOADS_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)

    application.mount("/uploads", StaticFiles(directory=settings.UPLOADS_DIR), name="uploads")
    application.mount("/outputs", StaticFiles(directory=settings.OUTPUTS_DIR), name="outputs")

    application.include_router(health.router,   prefix="/api",          tags=["Health"])
    application.include_router(upload.router,   prefix="/api/upload",   tags=["Upload"])
    application.include_router(validate.router, prefix="/api/validate", tags=["Validation"])
    application.include_router(predict.router,  prefix="/api/predict",  tags=["Prediction"])
    application.include_router(gradcam.router,  prefix="/api/gradcam",  tags=["GradCAM"])
    application.include_router(reports.router,  prefix="/api/reports",  tags=["Reports"])

    return application


app = create_application()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )