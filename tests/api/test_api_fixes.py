from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import Session

from api.session_manager_wrap import get_db_session
from api.v1.endpoints import songs, stats
from domain import vector_store_utils
from domain.schemas import EmbeddingModel, Song, SongEmbedding, Status


def test_stats_counts_only_completed(db: Session) -> None:
    song1 = Song(file_path="song1.flac", title="Song 1")
    song2 = Song(file_path="song2.flac", title="Song 2")
    song3 = Song(file_path="song3.flac", title="Song 3")
    db.add(song1)
    db.add(song2)
    db.add(song3)
    db.commit()

    emb_completed = SongEmbedding(
        file_path="song1.flac",
        model_name=EmbeddingModel.MUQ,
        status=Status.COMPLETED,
    )
    emb_failed = SongEmbedding(
        file_path="song2.flac",
        model_name=EmbeddingModel.MUQ,
        status=Status.FAILED,
    )
    emb_pending = SongEmbedding(
        file_path="song3.flac",
        model_name=EmbeddingModel.MUQ,
        status=Status.PENDING,
    )
    db.add(emb_completed)
    db.add(emb_failed)
    db.add(emb_pending)
    db.commit()

    db_stats = stats.get_db_stats(db=db)
    assert db_stats.songs_with_muq == 1
    assert db_stats.total_songs == 3


def test_similar_rejects_out_of_range_limit(db: Session) -> None:
    app = FastAPI()
    app.include_router(songs.router, prefix="/api/v1/songs")

    def override_get_db_session() -> Generator[Session, None, None]:
        yield db

    app.dependency_overrides[get_db_session] = override_get_db_session

    client = TestClient(app)

    resp_0 = client.get("/api/v1/songs/a.flac/similar?limit=0")
    assert resp_0.status_code == 422

    resp_201 = client.get("/api/v1/songs/a.flac/similar?limit=201")
    assert resp_201.status_code == 422


def test_qdrant_client_is_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_qdrant_cls = MagicMock()
    monkeypatch.setattr(vector_store_utils, "QdrantClient", mock_qdrant_cls)
    monkeypatch.setattr(vector_store_utils, "_client", None)

    client1 = vector_store_utils._get_client()
    client2 = vector_store_utils._get_client()

    assert client1 is client2
    assert mock_qdrant_cls.call_count == 1


def test_create_db_and_tables_raises_if_no_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.core import db_manager
    from domain import session_manager

    monkeypatch.setattr(session_manager, "engine", None)
    with pytest.raises(RuntimeError, match="Database engine is not initialized"):
        db_manager.create_db_and_tables()
