import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from sqlmodel import Session, select

import worker.acoustic_features
import worker.embeddings
from domain import task_manager
from domain.embedding_models import MODEL_SPECS
from domain.messages import AnalyzeSongTaskPayload
from domain.schemas import EmbeddingModel, SongEmbedding, Status, TaskStatus
from worker import tasks


def test_missing_song_gets_failed_embedding_and_no_vector(
    db: Session,
    music_root: Path,
    make_song: Callable[..., str],
    fake_vector_store: dict[str, dict[str, list[float]]],
    fake_inference: Any,
) -> None:
    """When a song file fails to load during batch embedding generation,

    it should receive a FAILED SongEmbedding and no vector in the vector store,
    while valid songs are completed and stored.
    """
    song1 = make_song("artist/album/song1.wav")
    song2 = make_song("artist/album/song2.wav")
    file_paths = [song1, song2]

    # Delete song1 from disk so librosa load fails
    (music_root / song1).unlink()

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

    muq_collection = MODEL_SPECS[EmbeddingModel.MUQ].collection_name

    # song1: missing, FAILED status, dimension is None, no vector
    emb1 = db.exec(
        select(SongEmbedding).where(
            SongEmbedding.file_path == song1,
            SongEmbedding.model_name == EmbeddingModel.MUQ,
        )
    ).one_or_none()
    assert emb1 is not None
    assert emb1.status == Status.FAILED
    assert emb1.dimension is None
    assert emb1.milvus_collection_name == muq_collection
    assert song1 not in fake_vector_store.get(muq_collection, {})

    # song2: present, COMPLETED status, vector stored
    emb2 = db.exec(
        select(SongEmbedding).where(
            SongEmbedding.file_path == song2,
            SongEmbedding.model_name == EmbeddingModel.MUQ,
        )
    ).one_or_none()
    assert emb2 is not None
    assert emb2.status == Status.COMPLETED
    assert emb2.dimension == 1024
    assert song2 in fake_vector_store.get(muq_collection, {})


def test_processed_count_advances_by_full_chunk_size_on_partial_failure(
    db: Session,
    music_root: Path,
    make_song: Callable[..., str],
    fake_vector_store: dict[str, dict[str, list[float]]],
    fake_inference: Any,
) -> None:
    """Task processed_count should advance by len(file_paths) even if songs fail."""
    song1 = make_song("artist/album/song1.wav")
    song2 = make_song("artist/album/song2.wav")
    song3 = make_song("artist/album/song3.wav")
    file_paths = [song1, song2, song3]

    (music_root / song1).unlink()

    task = task_manager.create_task(db, AnalyzeSongTaskPayload.NAME)
    task_manager.update_task_status(
        db,
        task.id,
        TaskStatus.RUNNING,
        details={"total_count": 3, "processed_count": 0},
    )

    start_time = time.time()
    tasks.generate_song_embedding(
        task_id=str(task.id),
        start_time=start_time,
        file_paths=file_paths,
        model_name="muq",
    )

    db.expire_all()
    refreshed_task = task_manager.get_task(db, task.id)
    assert refreshed_task is not None
    assert refreshed_task.details is not None
    assert refreshed_task.details["processed_count"] == 3
    assert refreshed_task.progress == 100
    assert refreshed_task.status == TaskStatus.SUCCESS


def test_select_songs_for_model_retry_failed_behavior(
    db: Session,
    make_song: Callable[..., str],
) -> None:
    """_select_songs_for_model selects unanalyzed songs, and only selects FAILED songs

    when retry_failed is True.
    """
    song_new = make_song("artist/album/new.wav")
    song_completed = make_song("artist/album/completed.wav")
    song_failed = make_song("artist/album/failed.wav")

    # Add COMPLETED embedding for song_completed
    db.add(
        SongEmbedding(
            file_path=song_completed,
            model_name=EmbeddingModel.MUQ,
            status=Status.COMPLETED,
            dimension=1024,
        )
    )
    # Add FAILED embedding for song_failed
    db.add(
        SongEmbedding(
            file_path=song_failed,
            model_name=EmbeddingModel.MUQ,
            status=Status.FAILED,
            dimension=None,
        )
    )
    db.commit()

    # With retry_failed=False: only song_new is selected
    selected_no_retry = tasks._select_songs_for_model(db, EmbeddingModel.MUQ, retry_failed=False)
    selected_paths_no_retry = {s.file_path for s in selected_no_retry}
    assert song_new in selected_paths_no_retry
    assert song_completed not in selected_paths_no_retry
    assert song_failed not in selected_paths_no_retry

    # With retry_failed=True: song_new and song_failed are selected, but not song_completed
    selected_retry = tasks._select_songs_for_model(db, EmbeddingModel.MUQ, retry_failed=True)
    selected_paths_retry = {s.file_path for s in selected_retry}
    assert song_new in selected_paths_retry
    assert song_failed in selected_paths_retry
    assert song_completed not in selected_paths_retry


def test_acoustic_extraction_failure_gets_failed_row_and_advances_progress(
    db: Session,
    make_song: Callable[..., str],
    fake_vector_store: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A song whose acoustic extraction failed (no SongFeatures row) gets a FAILED

    SongEmbedding(model_name=ACOUSTIC_FEATURES) after vectorize_song_features,
    and progress is recorded for the full chunk.
    """
    monkeypatch.setattr(
        worker.acoustic_features,
        "calculate_features_stats",
        lambda: worker.acoustic_features.SongFeaturesStats(),
    )

    song_failed = make_song("artist/album/failed_extract.wav")
    song_ok = make_song("artist/album/ok_extract.wav")
    file_paths = [song_failed, song_ok]

    task = task_manager.create_task(db, "Analyze Task")
    task_manager.update_task_status(
        db,
        task.id,
        TaskStatus.RUNNING,
        details={"total_count": 2, "processed_count": 0},
    )

    start_time = time.time()
    # Extract only for song_ok; song_failed extraction simulated as failed (no features created)
    res_extract = tasks.extract_acoustic_features(str(task.id), start_time, [song_ok])
    assert res_extract is True

    # Run vectorization for both songs
    res_vectorize = tasks.vectorize_song_features([], str(task.id), start_time, file_paths)
    assert res_vectorize is True

    # Verify song_failed has FAILED SongEmbedding and no vector
    failed_emb = db.exec(
        select(SongEmbedding).where(
            SongEmbedding.file_path == song_failed,
            SongEmbedding.model_name == EmbeddingModel.ACOUSTIC_FEATURES,
        )
    ).one_or_none()
    assert failed_emb is not None
    assert failed_emb.status == Status.FAILED
    assert failed_emb.dimension is None
    assert failed_emb.milvus_collection_name == worker.acoustic_features.COLLECTION_NAME
    assert song_failed not in fake_vector_store.get(worker.acoustic_features.COLLECTION_NAME, {})

    # Verify song_ok has COMPLETED SongEmbedding and vector
    ok_emb = db.exec(
        select(SongEmbedding).where(
            SongEmbedding.file_path == song_ok,
            SongEmbedding.model_name == EmbeddingModel.ACOUSTIC_FEATURES,
        )
    ).one_or_none()
    assert ok_emb is not None
    assert ok_emb.status == Status.COMPLETED
    assert ok_emb.dimension == worker.acoustic_features.EMBEDDING_DIMENSION
    assert song_ok in fake_vector_store.get(worker.acoustic_features.COLLECTION_NAME, {})

    # Task processed_count advances by full chunk (2)
    db.expire_all()
    refreshed = task_manager.get_task(db, task.id)
    assert refreshed is not None
    assert refreshed.details is not None
    assert refreshed.details.get("processed_count") == 2
    assert refreshed.progress == 100
    assert refreshed.status == TaskStatus.SUCCESS


def test_inference_failure_returns_false_and_writes_no_embedding_rows(
    db: Session,
    make_song: Callable[..., str],
    fake_inference: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When inference fails for muq, generate_song_embedding returns False

    and writes no SongEmbedding rows (neither COMPLETED nor FAILED).
    """
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

    def raise_for_muq(model_name: str, audio_data: Any) -> Any:
        if model_name == "muq":
            raise RuntimeError("Inference server error for muq")
        return fake_inference(model_name, audio_data)

    monkeypatch.setattr(worker.embeddings, "run_inference", raise_for_muq)

    start_time = time.time()
    result = tasks.generate_song_embedding(
        task_id=str(task.id),
        start_time=start_time,
        file_paths=file_paths,
        model_name="muq",
    )
    assert result is False

    # Verify no SongEmbedding rows are written (neither COMPLETED nor FAILED)
    embedding_rows = db.exec(select(SongEmbedding)).all()
    assert len(embedding_rows) == 0
