import asyncio
import importlib

import httpx
import numpy as np
import pytest
from fastapi import FastAPI

from domain import config
from inference.v1.endpoints import inference


class FakeModelManager:
    def run_text_embedding(self, model_name: str, text: str) -> np.ndarray:
        if model_name == "mert":
            raise NotImplementedError(f"{model_name} does not support text embedding")
        return np.array([0.1, 0.2, 0.3], dtype=np.float32)


def test_text_embedding_endpoint_success() -> None:
    """Tests 200 OK with expected list embedding."""
    app = FastAPI()
    app.include_router(inference.router, prefix="/api/v1")
    app.state.model_manager = FakeModelManager()

    async def _run_test() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/text-embedding/clap",
                json={"text": "ambient piano music"},
            )
            assert response.status_code == 200
            assert response.json() == {"embedding": pytest.approx([0.1, 0.2, 0.3])}

    asyncio.run(_run_test())


def test_text_embedding_endpoint_unknown_model() -> None:
    """Tests 404 Not Found for an unknown model."""
    app = FastAPI()
    app.include_router(inference.router, prefix="/api/v1")
    app.state.model_manager = FakeModelManager()

    async def _run_test() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/text-embedding/unknown_model",
                json={"text": "ambient piano music"},
            )
            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    asyncio.run(_run_test())


def test_text_embedding_endpoint_not_implemented() -> None:
    """Tests 400 Bad Request when the model raises NotImplementedError."""
    app = FastAPI()
    app.include_router(inference.router, prefix="/api/v1")
    app.state.model_manager = FakeModelManager()

    async def _run_test() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/text-embedding/mert",
                json={"text": "ambient piano music"},
            )
            assert response.status_code == 400
            assert "does not support text embedding" in response.json()["detail"]

    asyncio.run(_run_test())


def test_text_embedding_endpoint_empty_text() -> None:
    """Tests 422 Unprocessable Entity for empty text."""
    app = FastAPI()
    app.include_router(inference.router, prefix="/api/v1")
    app.state.model_manager = FakeModelManager()

    async def _run_test() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/text-embedding/clap",
                json={"text": ""},
            )
            assert response.status_code == 422

    asyncio.run(_run_test())


def test_text_embedding_endpoint_text_too_long() -> None:
    """Tests 422 Unprocessable Entity for text exceeding max length 1000."""
    app = FastAPI()
    app.include_router(inference.router, prefix="/api/v1")
    app.state.model_manager = FakeModelManager()

    async def _run_test() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/text-embedding/clap",
                json={"text": "a" * 1001},
            )
            assert response.status_code == 422

    asyncio.run(_run_test())


def test_inference_pinned_models_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that config.INFERENCE_PINNED_MODELS parses comma-separated lists and drops empties."""
    try:
        # Default when unset
        monkeypatch.delenv("INFERENCE_PINNED_MODELS", raising=False)
        importlib.reload(config)
        assert config.INFERENCE_PINNED_MODELS == ["muq_mulan"]

        # Explicit empty string
        monkeypatch.setenv("INFERENCE_PINNED_MODELS", "")
        importlib.reload(config)
        assert config.INFERENCE_PINNED_MODELS == []

        # Whitespace-only string
        monkeypatch.setenv("INFERENCE_PINNED_MODELS", "   ,   ")
        importlib.reload(config)
        assert config.INFERENCE_PINNED_MODELS == []

        # Multiple valid models with extra whitespace
        monkeypatch.setenv("INFERENCE_PINNED_MODELS", " muq_mulan , , clap ,   ")
        importlib.reload(config)
        assert config.INFERENCE_PINNED_MODELS == ["muq_mulan", "clap"]
    finally:
        monkeypatch.undo()
        importlib.reload(config)
