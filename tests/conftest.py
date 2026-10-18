from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest
import soundfile as sf
from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

import worker.embeddings
from domain import config, session_manager, vector_store_utils
from domain.embedding_models import MODEL_SPECS
from domain.schemas import Song


@pytest.fixture
def db_engine() -> Generator[Engine, None, None]:
    """Provides an in-memory SQLite engine with all domain schemas created."""
    import domain.schemas  # noqa: F401

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    yield engine
    SQLModel.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def db(db_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> Generator[Session, None, None]:
    """
    Patches SessionFactory so that db_session_context uses the test database,
    and yields a Session for test setup and assertions.
    """
    factory = sessionmaker(
        bind=db_engine,
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
    )
    monkeypatch.setattr(session_manager, "SessionFactory", factory)
    monkeypatch.setattr(session_manager, "engine", db_engine)
    with factory() as session:
        yield session


@pytest.fixture
def fake_vector_store(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, dict[str, list[float]]]:
    """
    In-memory vector store holding {collection_name: {file_path: list[float]}}.
    Patches vector_store_utils functions to read/write from this store.
    """
    store: dict[str, dict[str, list[float]]] = {}

    def fake_create_collection_if_not_exists(
        collection_name: str,
        dimension: int,
    ) -> None:
        if collection_name not in store:
            store[collection_name] = {}

    def fake_upsert_vectors(
        collection_name: str,
        data: list[dict[str, Any]],
    ) -> None:
        if collection_name not in store:
            store[collection_name] = {}
        for row in data:
            file_path = row.get("file_path")
            if not isinstance(file_path, str) or not file_path:
                continue
            vector = row.get("embedding", row.get("feature_vector"))
            if vector is None:
                continue
            if hasattr(vector, "tolist"):
                vector = vector.tolist()
            store[collection_name][file_path] = [float(x) for x in vector]

    def fake_delete_vectors(
        collection_name: str,
        ids: list[str],
    ) -> None:
        col = store.get(collection_name)
        if col is not None:
            for file_path in ids:
                col.pop(file_path, None)

    def fake_query_vector(
        collection_name: str,
        vector_id: str,
    ) -> list[float] | None:
        vec = store.get(collection_name, {}).get(vector_id)
        return list(vec) if vec is not None else None

    def fake_search_vectors(
        collection_name: str,
        query_vector: list[float],
        limit: int,
    ) -> list[dict[str, Any]]:
        col = store.get(collection_name, {})
        results: list[dict[str, Any]] = []
        q = np.array(query_vector, dtype=np.float32)
        norm_q = np.linalg.norm(q)
        for fp, vec in col.items():
            v = np.array(vec, dtype=np.float32)
            norm_v = np.linalg.norm(v)
            sim = float(np.dot(q, v) / (norm_q * norm_v)) if (norm_q > 0 and norm_v > 0) else 0.0
            results.append({"file_path": fp, "distance": sim})
        results.sort(key=lambda x: x["distance"], reverse=True)
        return results[:limit]

    monkeypatch.setattr(
        vector_store_utils,
        "create_collection_if_not_exists",
        fake_create_collection_if_not_exists,
    )
    monkeypatch.setattr(vector_store_utils, "upsert_vectors", fake_upsert_vectors)
    monkeypatch.setattr(vector_store_utils, "delete_vectors", fake_delete_vectors)
    monkeypatch.setattr(vector_store_utils, "query_vector", fake_query_vector)
    monkeypatch.setattr(vector_store_utils, "search_vectors", fake_search_vectors)

    monkeypatch.setattr(worker.embeddings, "_ready_collections", set())

    return store


class InferenceCall(NamedTuple):
    model_name: str
    audio_data: Any


class FakeInference:
    """Fake inference runner recording calls and returning constant vectors."""

    def __init__(self) -> None:
        self.calls: list[InferenceCall] = []

    def __call__(self, model_name: str, audio_data: Any) -> np.ndarray:
        self.calls.append(InferenceCall(model_name=model_name, audio_data=audio_data))
        dimensions = {spec.name.value: spec.dimension for spec in MODEL_SPECS.values()}
        dim = dimensions.get(model_name)
        if dim is None:
            raise ValueError(f"Unknown model_name for fake inference: {model_name}")
        batch_size = audio_data.shape[0] if hasattr(audio_data, "shape") else len(audio_data)
        return np.ones((batch_size, dim), dtype=np.float32)

    @property
    def call_count(self) -> int:
        return len(self.calls)


@pytest.fixture
def fake_inference(monkeypatch: pytest.MonkeyPatch) -> FakeInference:
    fake = FakeInference()
    monkeypatch.setattr(worker.embeddings, "run_inference", fake)
    return fake


@pytest.fixture
def music_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    music_dir = tmp_path / "music"
    music_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "MUSIC_NAS_ROOT_DIR", str(music_dir))
    return music_dir


@pytest.fixture
def make_song(music_root: Path, db: Session) -> Callable[..., str]:
    def _make_song(relative_path: str, seconds: float = 3.0) -> str:
        full_path = music_root / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        sample_rate = 24000
        total_samples = int(sample_rate * seconds)
        t = np.linspace(0, seconds, total_samples, endpoint=False, dtype=np.float32)
        audio_data = 0.5 * np.sin(2 * np.pi * 440.0 * t)
        sf.write(str(full_path), audio_data, sample_rate)

        song = Song(
            file_path=relative_path,
            title=Path(relative_path).stem,
            duration_seconds=int(seconds),
        )
        db.add(song)
        db.commit()
        return relative_path

    return _make_song
