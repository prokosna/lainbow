import asyncio
import importlib
import io
import time
from unittest.mock import patch

import httpx
import numpy as np
import pytest
from fastapi import FastAPI

from domain import config
from inference.core.model_manager import ModelManager
from inference.v1.endpoints import inference


class FakeModelManager:
    def run_inference(self, model_name: str, input_data: np.ndarray) -> np.ndarray:
        time.sleep(1.0)
        return np.zeros((1, 4), dtype=np.float32)


def test_inference_endpoint_does_not_block_health() -> None:
    """Tests that long-running inference in threadpool does not block /health route."""
    app = FastAPI()
    app.include_router(inference.router, prefix="/api/v1")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.state.model_manager = FakeModelManager()

    async def _run_test() -> None:
        buf = io.BytesIO()
        np.save(buf, np.zeros((1, 10), dtype=np.float32))
        files = {"file": ("audio.npy", buf.getvalue(), "application/octet-stream")}

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            inference_task = asyncio.create_task(client.post("/api/v1/inference/mert", files=files))

            # Allow the inference request to be dispatched and enter threadpool sleep
            await asyncio.sleep(0.05)

            t_health_start = time.perf_counter()
            health_response = await client.get("/health")
            health_duration = time.perf_counter() - t_health_start

            assert health_response.status_code == 200
            assert health_response.json() == {"status": "ok"}
            assert health_duration < 0.5

            inference_response = await inference_task
            assert inference_response.status_code == 200

            with io.BytesIO(inference_response.content) as resp_buf:
                embedding = np.load(resp_buf)

            np.testing.assert_array_equal(embedding, np.zeros((1, 4), dtype=np.float32))

    asyncio.run(_run_test())


def test_inference_load_models_on_startup_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that config.INFERENCE_LOAD_MODELS_ON_STARTUP drops empty entries."""
    try:
        # Unset / empty string
        monkeypatch.setenv("INFERENCE_MODELS_TO_LOAD_ON_STARTUP", "")
        importlib.reload(config)
        assert config.INFERENCE_LOAD_MODELS_ON_STARTUP == []

        # Whitespace-only string
        monkeypatch.setenv("INFERENCE_MODELS_TO_LOAD_ON_STARTUP", "   ,   ")
        importlib.reload(config)
        assert config.INFERENCE_LOAD_MODELS_ON_STARTUP == []

        # Mixed valid, empty, and whitespace entries
        monkeypatch.setenv("INFERENCE_MODELS_TO_LOAD_ON_STARTUP", " mert , , clap ,   ")
        importlib.reload(config)
        assert config.INFERENCE_LOAD_MODELS_ON_STARTUP == ["mert", "clap"]
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_model_manager_startup_skips_empty_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that ModelManager skips empty names on startup without warning."""
    try:
        monkeypatch.setattr(config, "INFERENCE_LOAD_MODELS_ON_STARTUP", ["", "mert", "   "])
        with patch.object(ModelManager, "_load_model") as mock_load:
            manager = ModelManager(mode="multi")
            assert "mert" in manager._models
            assert "" not in manager._models
            mock_load.assert_called_once_with("mert")
    finally:
        monkeypatch.undo()
