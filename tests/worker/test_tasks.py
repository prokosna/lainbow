import datetime
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlmodel import Session, col, select, text

from domain import task_manager
from domain.embedding_models import MODEL_SPECS
from domain.messages import AnalyzeSongTaskPayload, ScanTaskPayload
from domain.schemas import EmbeddingModel, SongEmbedding, Status, TaskStatus
from worker import tasks


def test_skip_when_same_task_already_running(db: Session) -> None:
    task1 = task_manager.create_task(db, ScanTaskPayload.NAME)
    task_manager.update_task_status(db, task1.id, TaskStatus.RUNNING)

    task2 = task_manager.create_task(db, ScanTaskPayload.NAME)
    assert task2.status == TaskStatus.PENDING

    tasks.scan_library_task(str(task2.id), {"task_id": task2.id, "target_path": "."})

    db.expire_all()
    refreshed_task2 = task_manager.get_task(db, task2.id)
    assert refreshed_task2 is not None
    assert refreshed_task2.status == TaskStatus.SKIPPED


def test_record_progress_accumulates_and_completes(db: Session) -> None:
    task = task_manager.create_task(db, "Test Progress Task")
    task_manager.update_task_status(
        db,
        task.id,
        TaskStatus.RUNNING,
        details={"total_count": 4, "processed_count": 0},
    )

    start_time = time.time()

    tasks.record_progress(task_id=str(task.id), start_time=start_time, processed_count=2)
    db.expire_all()
    refreshed_1 = task_manager.get_task(db, task.id)
    assert refreshed_1 is not None
    assert refreshed_1.progress == 50
    assert refreshed_1.status == TaskStatus.RUNNING
    assert refreshed_1.details == {"total_count": 4, "processed_count": 2}

    tasks.record_progress(task_id=str(task.id), start_time=start_time, processed_count=2)
    db.expire_all()
    refreshed_2 = task_manager.get_task(db, task.id)
    assert refreshed_2 is not None
    assert refreshed_2.progress == 100
    assert refreshed_2.status == TaskStatus.SUCCESS
    assert refreshed_2.details == {"total_count": 4, "processed_count": 4}


def test_generate_song_embedding_success(
    db: Session,
    make_song: Callable[..., str],
    fake_vector_store: dict[str, dict[str, list[float]]],
    fake_inference: Any,
) -> None:
    song1 = make_song("artist/album/song1.wav")
    song2 = make_song("artist/album/song2.wav")
    file_paths = [song1, song2]

    task = task_manager.create_task(db, AnalyzeSongTaskPayload.NAME)
    task_manager.update_task_status(
        db,
        task.id,
        TaskStatus.RUNNING,
        details={"total_count": 2, "processed_count": 0},
    )

    start_time = time.time()
    result = tasks.generate_song_embedding(
        task_id=str(task.id),
        start_time=start_time,
        file_paths=file_paths,
        model_name="muq",
    )
    assert result is True

    embeddings = db.exec(
        select(SongEmbedding).where(col(SongEmbedding.file_path).in_(file_paths))
    ).all()
    assert len(embeddings) == 2
    for emb in embeddings:
        assert emb.model_name == EmbeddingModel.MUQ
        assert emb.status == Status.COMPLETED
        assert emb.dimension == 1024
        assert emb.milvus_collection_name == MODEL_SPECS[EmbeddingModel.MUQ].collection_name

    muq_collection = MODEL_SPECS[EmbeddingModel.MUQ].collection_name
    assert muq_collection in fake_vector_store
    assert len(fake_vector_store[muq_collection]) == 2
    for fp in file_paths:
        assert fp in fake_vector_store[muq_collection]
        assert len(fake_vector_store[muq_collection][fp]) == 1024


def test_partial_failure_does_not_fail_whole_chunk(
    db: Session,
    music_root: Path,
    make_song: Callable[..., str],
    fake_vector_store: dict[str, dict[str, list[float]]],
    fake_inference: Any,
) -> None:
    song1 = make_song("artist/album/song1.wav")
    song2 = make_song("artist/album/song2.wav")
    file_paths = [song1, song2]

    (music_root / song1).unlink()

    task = task_manager.create_task(db, AnalyzeSongTaskPayload.NAME)
    task_manager.update_task_status(
        db,
        task.id,
        TaskStatus.RUNNING,
        details={"total_count": 2, "processed_count": 0},
    )

    start_time = time.time()
    tasks.generate_song_embedding(
        task_id=str(task.id),
        start_time=start_time,
        file_paths=file_paths,
        model_name="muq",
    )

    emb2 = db.exec(
        select(SongEmbedding).where(
            SongEmbedding.file_path == song2,
            SongEmbedding.model_name == EmbeddingModel.MUQ,
        )
    ).one_or_none()
    assert emb2 is not None
    assert emb2.status == Status.COMPLETED


def test_stale_running_task_does_not_block_forever(
    db: Session,
    music_root: Path,
) -> None:
    task1 = task_manager.create_task(db, ScanTaskPayload.NAME)
    task_manager.update_task_status(db, task1.id, TaskStatus.RUNNING)

    past = datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=2)
    db.exec(
        text("UPDATE taskresult SET updated_at = :val WHERE id = :id").params(
            val=past.strftime("%Y-%m-%d %H:%M:%S"), id=task1.id.hex
        )
    )
    db.commit()

    task2 = task_manager.create_task(db, ScanTaskPayload.NAME)

    tasks.scan_library_task(str(task2.id), {"task_id": task2.id, "target_path": "."})

    db.expire_all()
    refreshed_task2 = task_manager.get_task(db, task2.id)
    assert refreshed_task2 is not None
    assert refreshed_task2.status != TaskStatus.SKIPPED

    # The stale task is closed out rather than left RUNNING forever
    refreshed_task1 = task_manager.get_task(db, task1.id)
    assert refreshed_task1 is not None
    assert refreshed_task1.status == TaskStatus.FAILURE
