import time
from collections.abc import Callable
from typing import Any

import pytest
from sqlmodel import Session

import worker.acoustic_features
from domain import task_manager
from domain.schemas import TaskStatus
from worker import tasks


def test_acoustic_extraction_alone_does_not_complete_task(
    db: Session,
    make_song: Callable[..., str],
) -> None:
    song1 = make_song("artist/album/song1.wav")
    song2 = make_song("artist/album/song2.wav")
    file_paths = [song1, song2]

    task = task_manager.create_task(db, "Analyze Task")
    task_manager.update_task_status(
        db,
        task.id,
        TaskStatus.RUNNING,
        details={"total_count": 2, "processed_count": 0},
    )

    start_time = time.time()
    result = tasks.extract_acoustic_features(str(task.id), start_time, file_paths)
    assert result is True

    db.expire_all()
    refreshed = task_manager.get_task(db, task.id)
    assert refreshed is not None
    assert refreshed.status == TaskStatus.RUNNING
    assert refreshed.details is not None
    assert refreshed.details.get("processed_count") == 0


def test_acoustic_vectorization_completes_task(
    db: Session,
    make_song: Callable[..., str],
    fake_vector_store: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        worker.acoustic_features,
        "calculate_features_stats",
        lambda: worker.acoustic_features.SongFeaturesStats(),
    )

    song1 = make_song("artist/album/song1.wav")
    song2 = make_song("artist/album/song2.wav")
    file_paths = [song1, song2]

    task = task_manager.create_task(db, "Analyze Task")
    task_manager.update_task_status(
        db,
        task.id,
        TaskStatus.RUNNING,
        details={"total_count": 2, "processed_count": 0},
    )

    start_time = time.time()
    res_extract = tasks.extract_acoustic_features(str(task.id), start_time, file_paths)
    assert res_extract is True

    res_vectorize = tasks.vectorize_song_features([], str(task.id), start_time, file_paths)
    assert res_vectorize is True

    db.expire_all()
    refreshed = task_manager.get_task(db, task.id)
    assert refreshed is not None
    assert refreshed.status == TaskStatus.SUCCESS
    assert refreshed.progress == 100


def test_record_progress_never_exceeds_100(db: Session) -> None:
    task = task_manager.create_task(db, "Test Progress Task")
    task_manager.update_task_status(
        db,
        task.id,
        TaskStatus.RUNNING,
        details={"total_count": 2, "processed_count": 0},
    )

    start_time = time.time()
    tasks.record_progress(task_id=str(task.id), start_time=start_time, processed_count=2)
    db.expire_all()
    refreshed_1 = task_manager.get_task(db, task.id)
    assert refreshed_1 is not None
    assert refreshed_1.progress == 100
    assert refreshed_1.status == TaskStatus.SUCCESS

    tasks.record_progress(task_id=str(task.id), start_time=start_time, processed_count=2)
    db.expire_all()
    refreshed_2 = task_manager.get_task(db, task.id)
    assert refreshed_2 is not None
    assert refreshed_2.progress == 100
    assert refreshed_2.status == TaskStatus.SUCCESS
