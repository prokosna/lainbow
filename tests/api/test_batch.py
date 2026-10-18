from unittest.mock import MagicMock

import pytest
from sqlmodel import Session

from api.v1.endpoints import batch


def test_batch_run_scan_calls_send_task(db: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_send = MagicMock()
    monkeypatch.setattr(batch.celery_app, "send_task", mock_send)

    task_result = batch.run_scan(db=db)

    assert mock_send.call_count == 1
    call_args, call_kwargs = mock_send.call_args
    assert call_args[0] == "worker.tasks.scan_library_task"
    assert call_kwargs["queue"] == "cpu_queue"
    assert call_kwargs["kwargs"]["task_id"] == str(task_result.id)
    assert call_kwargs["kwargs"]["payload"]["task_id"] == task_result.id
    assert call_kwargs["kwargs"]["payload"]["target_path"] == "."


def test_batch_run_vacuum_calls_send_task(db: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_send = MagicMock()
    monkeypatch.setattr(batch.celery_app, "send_task", mock_send)

    task_result = batch.run_vacuum(db=db)

    assert mock_send.call_count == 1
    call_args, call_kwargs = mock_send.call_args
    assert call_args[0] == "worker.tasks.vacuum_library_task"
    assert call_kwargs["queue"] == "cpu_queue"
    assert call_kwargs["kwargs"]["task_id"] == str(task_result.id)
    assert call_kwargs["kwargs"]["payload"]["task_id"] == task_result.id


def test_batch_run_analysis_calls_send_task(db: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_send = MagicMock()
    monkeypatch.setattr(batch.celery_app, "send_task", mock_send)

    task_result = batch.run_analysis(db=db, request=batch.RunAnalysisRequest())

    assert mock_send.call_count == 1
    call_args, call_kwargs = mock_send.call_args
    assert call_args[0] == "worker.tasks.analyze_song_task"
    assert call_kwargs["queue"] == "cpu_queue"
    assert call_kwargs["kwargs"]["task_id"] == str(task_result.id)
    assert call_kwargs["kwargs"]["payload"]["task_id"] == task_result.id
    assert "models" in call_kwargs["kwargs"]["payload"]
