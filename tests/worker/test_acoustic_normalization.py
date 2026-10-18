import time
from typing import Any

import numpy as np
import pytest
from sqlmodel import Session, select

import worker.acoustic_features
from domain import task_manager, vector_store_utils
from domain.messages import AnalyzeSongTaskPayload
from domain.schemas import (
    EmbeddingModel,
    Song,
    SongEmbedding,
    SongFeatures,
    Status,
    TaskStatus,
)
from worker import tasks
from worker.acoustic_features import FEATURE_ORDER, FeatureStat, SongFeaturesStats


def _make_features(file_path: str, val: float = 1.0) -> SongFeatures:
    kwargs: dict[str, Any] = {
        "file_path": file_path,
        "status": Status.COMPLETED,
    }
    for col_name in FEATURE_ORDER:
        kwargs[col_name] = val
    return SongFeatures(**kwargs)


def test_reupsert_all_completed_features_with_new_stats(
    db: Session,
    fake_vector_store: dict[str, dict[str, list[float]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two songs already vectorized in an earlier run plus one new song:

    after vectorize_song_features for the new song only, all three vectors in
    fake_vector_store were written in this call with the stats it returned.
    Only the new song gets a new SongEmbedding row.
    """
    song1 = "artist/album/song1.wav"
    song2 = "artist/album/song2.wav"
    song3 = "artist/album/song3.wav"

    # Add songs 1 and 2
    db.add(Song(file_path=song1, title="Song 1"))
    db.add(Song(file_path=song2, title="Song 2"))
    sf1 = _make_features(song1, 10.0)
    sf2 = _make_features(song2, 20.0)
    db.add(sf1)
    db.add(sf2)
    db.commit()

    stats_run1 = SongFeaturesStats(
        bpm=FeatureStat(mean=100.0, std=10.0),
        **{name: FeatureStat(mean=1.0, std=2.0) for name in FEATURE_ORDER if name != "bpm"},
    )
    stats_run2 = SongFeaturesStats(
        bpm=FeatureStat(mean=150.0, std=30.0),
        **{name: FeatureStat(mean=5.0, std=10.0) for name in FEATURE_ORDER if name != "bpm"},
    )

    current_stats = stats_run1
    monkeypatch.setattr(worker.acoustic_features, "calculate_features_stats", lambda: current_stats)

    task1 = task_manager.create_task(db, "Analyze Task 1")
    task_manager.update_task_status(
        db, task1.id, TaskStatus.RUNNING, details={"total_count": 2, "processed_count": 0}
    )
    start_time = time.time()
    res1 = tasks.vectorize_song_features([], str(task1.id), start_time, [song1, song2])
    assert res1 is True

    col_name = worker.acoustic_features.COLLECTION_NAME
    # Verify song1 and song2 vectors match run 1 stats
    vec1_run1 = worker.acoustic_features.create_feature_vector_data(sf1, stats_run1)[
        "feature_vector"
    ]
    vec2_run1 = worker.acoustic_features.create_feature_vector_data(sf2, stats_run1)[
        "feature_vector"
    ]
    assert np.allclose(fake_vector_store[col_name][song1], vec1_run1)
    assert np.allclose(fake_vector_store[col_name][song2], vec2_run1)
    assert song3 not in fake_vector_store[col_name]

    # Verify embeddings for run 1
    emb1_run1 = db.exec(
        select(SongEmbedding).where(
            SongEmbedding.file_path == song1,
            SongEmbedding.model_name == EmbeddingModel.ACOUSTIC_FEATURES,
        )
    ).one()
    emb2_run1 = db.exec(
        select(SongEmbedding).where(
            SongEmbedding.file_path == song2,
            SongEmbedding.model_name == EmbeddingModel.ACOUSTIC_FEATURES,
        )
    ).one()
    emb1_created_at = emb1_run1.created_at
    emb2_created_at = emb2_run1.created_at

    # Add song 3
    db.add(Song(file_path=song3, title="Song 3"))
    sf3 = _make_features(song3, 30.0)
    db.add(sf3)
    db.commit()

    # Second run with different stats, called ONLY for the new song
    current_stats = stats_run2
    task2 = task_manager.create_task(db, "Analyze Task 2")
    task_manager.update_task_status(
        db, task2.id, TaskStatus.RUNNING, details={"total_count": 1, "processed_count": 0}
    )
    res2 = tasks.vectorize_song_features([], str(task2.id), start_time, [song3])
    assert res2 is True

    # Assert all three vectors in fake_vector_store were written with the second stats
    vec1_run2 = worker.acoustic_features.create_feature_vector_data(sf1, stats_run2)[
        "feature_vector"
    ]
    vec2_run2 = worker.acoustic_features.create_feature_vector_data(sf2, stats_run2)[
        "feature_vector"
    ]
    vec3_run2 = worker.acoustic_features.create_feature_vector_data(sf3, stats_run2)[
        "feature_vector"
    ]

    # Verify vectors differ between run1 and run2
    assert not np.allclose(vec1_run1, vec1_run2)
    assert not np.allclose(vec2_run1, vec2_run2)

    # Verify all three vectors in vector store now match run 2 stats
    assert np.allclose(fake_vector_store[col_name][song1], vec1_run2)
    assert np.allclose(fake_vector_store[col_name][song2], vec2_run2)
    assert np.allclose(fake_vector_store[col_name][song3], vec3_run2)

    # Only the new song gets a new SongEmbedding row
    all_embeddings = db.exec(
        select(SongEmbedding).where(SongEmbedding.model_name == EmbeddingModel.ACOUSTIC_FEATURES)
    ).all()
    assert len(all_embeddings) == 3
    emb3 = db.exec(
        select(SongEmbedding).where(
            SongEmbedding.file_path == song3,
            SongEmbedding.model_name == EmbeddingModel.ACOUSTIC_FEATURES,
        )
    ).one()
    assert emb3.status == Status.COMPLETED
    assert emb3.dimension == worker.acoustic_features.EMBEDDING_DIMENSION

    # Song 1 and 2 embeddings were untouched
    db.refresh(emb1_run1)
    db.refresh(emb2_run1)
    assert emb1_run1.created_at == emb1_created_at
    assert emb2_run1.created_at == emb2_created_at


def test_vectorize_song_features_pagination_with_2500_songs(
    db: Session,
    fake_vector_store: dict[str, dict[str, list[float]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With 2,500 SongFeatures rows, upsert_vectors is called 3 times (pages of 1000)."""
    stats = SongFeaturesStats()
    monkeypatch.setattr(worker.acoustic_features, "calculate_features_stats", lambda: stats)

    # Insert 2,500 Song and SongFeatures rows
    songs = [Song(file_path=f"songs/song_{i}.wav", title=f"Song {i}") for i in range(2500)]
    features = [_make_features(f"songs/song_{i}.wav", float(i % 100)) for i in range(2500)]
    db.add_all(songs)
    db.add_all(features)
    db.commit()

    upsert_batches: list[list[dict[str, Any]]] = []
    original_upsert = vector_store_utils.upsert_vectors

    def wrapping_upsert(collection_name: str, data: list[dict[str, Any]]) -> None:
        upsert_batches.append(data)
        original_upsert(collection_name, data)

    monkeypatch.setattr(vector_store_utils, "upsert_vectors", wrapping_upsert)

    task = task_manager.create_task(db, "Analyze Task")
    task_manager.update_task_status(
        db, task.id, TaskStatus.RUNNING, details={"total_count": 1, "processed_count": 0}
    )
    start_time = time.time()

    # Call vectorize_song_features for one new song only
    result = tasks.vectorize_song_features([], str(task.id), start_time, ["songs/song_0.wav"])
    assert result is True

    # upsert_vectors should be called 3 times with pages of 1000, 1000, 500
    assert len(upsert_batches) == 3
    assert [len(batch) for batch in upsert_batches] == [1000, 1000, 500]
    assert sum(len(batch) for batch in upsert_batches) == 2500

    # Only the new song gets a SongEmbedding row
    embeddings = db.exec(
        select(SongEmbedding).where(SongEmbedding.model_name == EmbeddingModel.ACOUSTIC_FEATURES)
    ).all()
    assert len(embeddings) == 1
    assert embeddings[0].file_path == "songs/song_0.wav"
    assert embeddings[0].status == Status.COMPLETED


def test_vectorize_song_features_target_chunking_with_2500_songs(
    db: Session,
    fake_vector_store: dict[str, dict[str, list[float]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With 2,500 target paths, targets are processed in chunks of 1000 and every target gets one SongEmbedding."""
    stats = SongFeaturesStats()
    monkeypatch.setattr(worker.acoustic_features, "calculate_features_stats", lambda: stats)

    # Insert 2,500 Song and SongFeatures rows directly
    target_paths = [f"songs/song_{i}.wav" for i in range(2500)]
    songs = [Song(file_path=p, title=f"Song {i}") for i, p in enumerate(target_paths)]
    features = [_make_features(p, float(i % 100)) for i, p in enumerate(target_paths)]
    db.add_all(songs)
    db.add_all(features)
    db.commit()

    chunk_calls: list[tuple[int, int]] = []
    chunk_lengths: list[int] = []
    original_chunks = tasks._chunks

    def spy_chunks(seq: Any, size: int) -> Any:
        chunk_calls.append((len(seq), size))
        for chunk in original_chunks(seq, size):
            chunk_lengths.append(len(chunk))
            yield chunk

    monkeypatch.setattr(tasks, "_chunks", spy_chunks)

    task = task_manager.create_task(db, "Analyze Task")
    task_manager.update_task_status(
        db, task.id, TaskStatus.RUNNING, details={"total_count": 2500, "processed_count": 0}
    )
    start_time = time.time()

    result = tasks.vectorize_song_features([], str(task.id), start_time, target_paths)
    assert result is True

    # Prove chunking: _chunks was called with size 1000 on the 2,500 paths,
    # yielding chunks of sizes [1000, 1000, 500].
    assert chunk_calls == [(2500, 1000)]
    assert chunk_lengths == [1000, 1000, 500]

    # Every target gets exactly one SongEmbedding row
    embeddings = db.exec(
        select(SongEmbedding).where(SongEmbedding.model_name == EmbeddingModel.ACOUSTIC_FEATURES)
    ).all()
    assert len(embeddings) == 2500
    assert {e.file_path for e in embeddings} == set(target_paths)
    assert all(e.status == Status.COMPLETED for e in embeddings)
    assert all(e.dimension == worker.acoustic_features.EMBEDDING_DIMENSION for e in embeddings)


def test_analyze_song_task_chains_single_vectorize_task_for_acoustic(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """analyze_song_task should chain extraction group to a single vectorize task."""
    for i in range(15):
        db.add(Song(file_path=f"songs/song_{i}.wav", title=f"Song {i}"))
    db.commit()

    captured_groups: list[Any] = []
    monkeypatch.setattr(tasks, "TASK_CHUNK_SIZE", 5)

    def fake_chord(header: Any) -> Any:
        captured_groups.extend(header)
        return lambda callback: None

    monkeypatch.setattr(tasks, "chord", fake_chord)

    task = task_manager.create_task(db, AnalyzeSongTaskPayload.NAME)
    tasks.analyze_song_task(
        str(task.id),
        {"task_id": task.id, "models": ["acoustic_features"]},
    )

    assert len(captured_groups) == 1
    acoustic_canvas = captured_groups[0]

    # In Celery, chain(group, task) is optimized as a chord canvas:
    # with extraction tasks in header and the single vectorize_song_features task as body
    if acoustic_canvas.name == "celery.chord":
        header = acoustic_canvas.kwargs["header"]
        body = acoustic_canvas.kwargs["body"]
        # 15 songs with chunk size 5 -> 3 extract tasks in header
        assert len(header) == 3
        for extract_task in header:
            assert extract_task.task == "worker.tasks.extract_acoustic_features"
        # Body is a single vectorize_song_features task with all 15 file paths
        assert body.task == "worker.tasks.vectorize_song_features"
        assert len(body.kwargs["file_paths"]) == 15
    else:
        extract_group = acoustic_canvas.tasks[0]
        assert len(extract_group.tasks) == 3
        vectorize_task = acoustic_canvas.tasks[1]
        assert vectorize_task.task == "worker.tasks.vectorize_song_features"
        assert len(vectorize_task.kwargs["file_paths"]) == 15
