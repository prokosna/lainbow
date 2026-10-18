"""WI-M1-01 (F2): no task may be left PENDING or RUNNING after its work has ended."""

import time
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlmodel import Session

from api.v1.endpoints import batch
from domain import task_manager
from domain.messages import AnalyzeSongTaskPayload, ScanTaskPayload
from domain.schemas import TaskStatus
from worker import tasks


def _status(db: Session, task_id: Any) -> TaskStatus:
    db.expire_all()
    task = task_manager.get_task(db, task_id)
    assert task is not None
    return task.status


def test_recent_running_task_still_blocks(db: Session) -> None:
    running = task_manager.create_task(db, ScanTaskPayload.NAME)
    task_manager.update_task_status(db, running.id, TaskStatus.RUNNING)
    new = task_manager.create_task(db, ScanTaskPayload.NAME)

    tasks.scan_library_task(str(new.id), {"task_id": new.id, "target_path": "."})

    assert _status(db, new.id) == TaskStatus.SKIPPED
    assert _status(db, running.id) == TaskStatus.RUNNING


def test_analyze_failure_before_dispatch_marks_failure(
    db: Session, make_song: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_song("a/song.wav")
    monkeypatch.setattr(tasks, "chord", MagicMock(side_effect=RuntimeError("broker down")))
    task = task_manager.create_task(db, AnalyzeSongTaskPayload.NAME)

    tasks.analyze_song_task(str(task.id), {"task_id": task.id, "models": ["muq"]})

    assert _status(db, task.id) == TaskStatus.FAILURE


def test_embedding_subtask_never_raises_when_progress_recording_fails(
    db: Session,
    make_song: Any,
    fake_vector_store: Any,
    fake_inference: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    song = make_song("a/song.wav")
    task = task_manager.create_task(db, AnalyzeSongTaskPayload.NAME)
    task_manager.update_task_status(
        db, task.id, TaskStatus.RUNNING, details={"total_count": 1, "processed_count": 0}
    )
    monkeypatch.setattr(tasks, "record_progress", MagicMock(side_effect=RuntimeError("db gone")))

    # A raising header task would keep the chord callback from ever running
    result = tasks.generate_song_embedding(
        task_id=str(task.id), start_time=time.time(), file_paths=[song], model_name="muq"
    )

    assert result is False


def test_enqueue_failure_marks_task_failure(db: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        batch.scan_library_task, "delay", MagicMock(side_effect=RuntimeError("broker down"))
    )

    with pytest.raises(HTTPException) as exc_info:
        batch.run_scan(db=db)

    assert exc_info.value.status_code == 500
    created = task_manager.get_latest_task_by_name(db, ScanTaskPayload.NAME, excluding_id=uuid4())
    assert created is not None
    assert _status(db, created.id) == TaskStatus.FAILURE
