from collections.abc import Generator
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import Session

from api.session_manager_wrap import get_db_session
from api.v1.endpoints import batch
from domain import config
from domain.messages import AnalyzeSongTaskPayload


def test_analyze_defaults_with_monkeypatched_config(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "ANALYZE_MODELS", ["muq_mulan", "acoustic_features"])

    mock_send = MagicMock()
    monkeypatch.setattr(batch.celery_app, "send_task", mock_send)

    # 1. Calling batch.run_analysis with a default RunAnalysisRequest() sends payload with monkeypatched models
    task_result = batch.run_analysis(db=db, request=batch.RunAnalysisRequest())
    assert mock_send.call_count == 1
    call_args, call_kwargs = mock_send.call_args
    assert call_args[0] == "worker.tasks.analyze_song_task"
    assert call_kwargs["queue"] == "cpu_queue"
    assert call_kwargs["kwargs"]["task_id"] == str(task_result.id)
    assert call_kwargs["kwargs"]["payload"]["models"] == ["muq_mulan", "acoustic_features"]

    # 2. Calling with an explicit RunAnalysisRequest(models=["clap"]) sends ["clap"]
    mock_send.reset_mock()
    task_result_explicit = batch.run_analysis(
        db=db, request=batch.RunAnalysisRequest(models=["clap"])
    )
    assert mock_send.call_count == 1
    _, call_kwargs_explicit = mock_send.call_args
    assert call_kwargs_explicit["kwargs"]["task_id"] == str(task_result_explicit.id)
    assert call_kwargs_explicit["kwargs"]["payload"]["models"] == ["clap"]

    # 3. AnalyzeSongTaskPayload default models equals monkeypatched value
    payload = AnalyzeSongTaskPayload(task_id=uuid4())
    assert payload.models == ["muq_mulan", "acoustic_features"]


def test_analyze_defaults_http_endpoint_without_body(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "ANALYZE_MODELS", ["muq_mulan", "acoustic_features"])

    mock_send = MagicMock()
    monkeypatch.setattr(batch.celery_app, "send_task", mock_send)

    app = FastAPI()
    app.include_router(batch.router, prefix="/api/v1/batch")

    def override_get_db_session() -> Generator[Session, None, None]:
        yield db

    app.dependency_overrides[get_db_session] = override_get_db_session
    client = TestClient(app)

    # Calling POST /api/v1/batch/analyze/run without a body (as Sola MPD does)
    resp = client.post("/api/v1/batch/analyze/run")
    assert resp.status_code == 200
    assert mock_send.call_count == 1
    _, call_kwargs = mock_send.call_args
    assert call_kwargs["kwargs"]["payload"]["models"] == ["muq_mulan", "acoustic_features"]

    # Calling with explicit models body overrides defaults
    mock_send.reset_mock()
    resp_explicit = client.post("/api/v1/batch/analyze/run", json={"models": ["clap"]})
    assert resp_explicit.status_code == 200
    assert mock_send.call_count == 1
    _, call_kwargs_explicit = mock_send.call_args
    assert call_kwargs_explicit["kwargs"]["payload"]["models"] == ["clap"]


def test_analyze_defaults_unpatched() -> None:
    # Invariant I1: when LAINBOW_ANALYZE_MODELS is unset, defaults are ["muq", "muq_mulan"]
    req = batch.RunAnalysisRequest()
    assert req.models == ["muq", "muq_mulan"]
    payload = AnalyzeSongTaskPayload(task_id=uuid4())
    assert payload.models == ["muq", "muq_mulan"]
