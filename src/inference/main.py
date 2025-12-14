import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

from domain.config import INFERENCE_MODEL_MANAGER_MODE
from fastapi import FastAPI

from inference.core.model_manager import ModelManager, ModelManagerMode
from inference.v1.endpoints import inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage the model manager's lifecycle with the application startup and shutdown events."""
    # Load the model manager on startup
    mode = cast(ModelManagerMode, INFERENCE_MODEL_MANAGER_MODE)
    model_manager = ModelManager(mode=mode)
    app.state.model_manager = model_manager
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(inference.router, prefix="/api/v1", tags=["Inference"])


@app.get("/health", status_code=200)
async def health_check() -> dict[str, str]:
    """Health check endpoint for Docker and other monitoring systems."""
    return {"status": "ok"}
