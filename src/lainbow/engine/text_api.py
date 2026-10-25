"""Tiny HTTP endpoint that embeds text queries for the server."""

import logging
import threading
from collections.abc import Mapping
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException

from lainbow.engine.protocol import TextEmbeddingIn, TextEmbeddingOut

logger = logging.getLogger(__name__)


def create_app(models: Mapping[str, Any]) -> FastAPI:
    """Build the app serving every loaded model that has ``embed_text``."""
    text_models = {k: m for k, m in models.items() if callable(getattr(m, "embed_text", None))}
    app = FastAPI(title="Lainbow engine text endpoint")

    @app.post("/text-embedding")
    def text_embedding(body: TextEmbeddingIn) -> TextEmbeddingOut:
        model = text_models.get(body.model)
        if model is None:
            raise HTTPException(status_code=404, detail=f"unknown text model {body.model!r}")
        return TextEmbeddingOut(embedding=[float(x) for x in model.embed_text(body.text)])

    return app


def serve_in_thread(app: FastAPI, port: int) -> threading.Thread:
    """Run the app with uvicorn in a daemon thread."""
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="text-api", daemon=True)
    thread.start()
    logger.info("text-embedding endpoint listening on port %d", port)
    return thread
