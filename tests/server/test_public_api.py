"""Tests for the public API: similar songs, text search, analysis, stats, analyze/run."""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import Engine, create_engine, insert
from sqlalchemy.orm import Session, sessionmaker

from lainbow.core import models
from lainbow.core.settings import ServerSettings
from lainbow.db.migrate import upgrade_to_head
from lainbow.db.schema import Analysis, AnalysisDetail, Track
from lainbow.db.session import make_session_factory
from lainbow.db.vectors import ensure_vector_store, vector_table
from lainbow.server.app import create_app
from lainbow.server.profiles import activate_profiles
from lainbow.server.text_embedding import TextEmbedder, TextEmbeddingUnavailableError

pytestmark = pytest.mark.postgres

ACOUSTIC = models.MODELS["acoustic"]
MULAN = models.MODELS["muq_mulan"]
OTHER_PROFILE = "r0-old"


class FakeEmbedder(TextEmbedder):
    """Returns a fixed vector, or fails, and records calls."""

    def __init__(self, vector: list[float] | None) -> None:
        super().__init__("http://unused")
        self.vector = vector
        self.calls: list[tuple[str, str]] = []

    def embed(self, model: str, text: str) -> list[float]:
        self.calls.append((model, text))
        if self.vector is None:
            raise TextEmbeddingUnavailableError("down")
        return self.vector


def vec(dim: int, *values: float) -> list[float]:
    return [*values, *([0.0] * (dim - len(values)))]


@pytest.fixture
def engine(clean_db: str) -> Iterator[Engine]:
    upgrade_to_head(clean_db)
    eng = create_engine(clean_db)
    yield eng
    eng.dispose()


@pytest.fixture
def factory(engine: Engine) -> sessionmaker[Session]:
    return make_session_factory(engine)


def make_settings(*model_names: str) -> ServerSettings:
    return ServerSettings(
        database_url=SecretStr("postgresql://unused/x"),
        models=list(model_names),
        scan_interval_min=0,
    )


def profile_of(model: str, factory: sessionmaker[Session], settings: ServerSettings) -> str:
    return models.MODELS[model].profile(settings.method_for(model))


def add_track(session: Session, path: str, **kw: str) -> int:
    now = datetime.now(UTC)
    track = Track(path=path, duration_sec=100.0, mtime=now, size_bytes=1, seen_at=now, **kw)
    session.add(track)
    session.flush()
    return track.id


def add_vector(session: Session, model: str, profile: str, track_id: int, v: list[float]) -> None:
    session.execute(
        insert(vector_table(model)).values(track_id=track_id, profile=profile, embedding=v)
    )


def add_analysis(
    session: Session, track_id: int, model: str, profile: str, state: str, **kw: object
) -> None:
    session.add(Analysis(track_id=track_id, model=model, profile=profile, state=state, **kw))


def test_similar_order_exclusion_and_encoding(factory: sessionmaker[Session]) -> None:
    settings = make_settings("acoustic")
    profile = profile_of("acoustic", factory, settings)
    d = ACOUSTIC.dimension
    paths = ["Artist/Al bum/01 a b.flac", "x/near.flac", "x/mid.flac", "x/far.flac"]
    vectors = [vec(d, 1, 0), vec(d, 1, 0.1), vec(d, 1, 1), vec(d, 0, 1)]
    with factory() as session:
        activate_profiles(factory, settings)
        for path, v in zip(paths, vectors, strict=True):
            tid = add_track(session, path, title=path)
            add_vector(session, "acoustic", profile, tid, v)
        session.commit()
    with TestClient(create_app(settings, session_factory=factory)) as client:
        url = f"/api/v1/songs/{quote(paths[0], safe='')}/similar"
        resp = client.get(url, params={"model_name": "acoustic", "limit": 10})
        assert resp.status_code == 200
        songs = resp.json()["songs"]
        assert [s["file_path"] for s in songs] == paths[1:]
        assert songs[0]["score"] > songs[1]["score"] > songs[2]["score"]
        assert songs[0]["title"] == "x/near.flac"
        limited = client.get(url, params={"model_name": "acoustic", "limit": 1}).json()
        assert len(limited["songs"]) == 1


def test_similar_errors(factory: sessionmaker[Session]) -> None:
    settings = make_settings("acoustic", "muq_mulan")
    profile = profile_of("acoustic", factory, settings)
    with factory() as session:
        activate_profiles(factory, settings)
        ensure_vector_store(session.connection(), "acoustic", OTHER_PROFILE)
        no_vec = add_track(session, "a/none.flac")
        other = add_track(session, "a/other.flac")
        add_vector(session, "acoustic", OTHER_PROFILE, other, vec(ACOUSTIC.dimension, 1))
        ok = add_track(session, "a/ok.flac")
        add_vector(session, "acoustic", profile, ok, vec(ACOUSTIC.dimension, 1))
        assert no_vec != other
        session.commit()
    with TestClient(create_app(settings, session_factory=factory)) as client:

        def similar(path: str, model: str) -> tuple[int, str]:
            r = client.get(f"/api/v1/songs/{quote(path, safe='')}/similar?model_name={model}")
            return r.status_code, str(r.json().get("detail"))

        assert similar("a/none.flac", "acoustic") == (
            404,
            "a/none.flac has not been analyzed with acoustic yet",
        )
        assert similar("a/other.flac", "acoustic")[0] == 404
        assert similar("a/missing.flac", "acoustic")[0] == 404
        assert similar("a/ok.flac", "mert")[0] == 422  # inactive
        assert similar("a/ok.flac", "nonexistent")[0] == 422


def test_similar_requires_similar_purpose(
    factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings("acoustic")
    textonly = {**models.MODELS, "acoustic": _replace_purposes(ACOUSTIC, {models.Purpose.TEXT})}
    monkeypatch.setattr("lainbow.core.models.MODELS", textonly)
    with TestClient(create_app(settings, session_factory=factory)) as client:
        resp = client.get("/api/v1/songs/a.flac/similar?model_name=acoustic")
        assert resp.status_code == 422


def _replace_purposes(spec: models.ModelSpec, purposes: set[models.Purpose]) -> models.ModelSpec:
    import dataclasses

    return dataclasses.replace(spec, purposes=frozenset(purposes))


def test_search(factory: sessionmaker[Session]) -> None:
    settings = make_settings("muq", "muq_mulan")
    profile = profile_of("muq_mulan", factory, settings)
    d = MULAN.dimension
    with factory() as session:
        activate_profiles(factory, settings)
        for path, v in (("s/a.flac", vec(d, 1, 0)), ("s/b.flac", vec(d, 1, 1))):
            add_vector(session, "muq_mulan", profile, add_track(session, path), v)
        session.commit()
    embedder = FakeEmbedder(vec(d, 1, 0.2))
    with TestClient(create_app(settings, session_factory=factory, text_embedder=embedder)) as c:
        resp = c.get("/api/v1/songs/search", params={"q": "calm piano", "model_name": "muq_mulan"})
        assert resp.status_code == 200
        assert [s["file_path"] for s in resp.json()["songs"]] == ["s/a.flac", "s/b.flac"]
        assert embedder.calls == [("muq_mulan", "calm piano")]
        # "search" is not taken as a file path
        assert c.get("/api/v1/songs/search").status_code == 422
        # muq has no TEXT purpose
        bad = c.get("/api/v1/songs/search", params={"q": "x", "model_name": "muq"})
        assert bad.status_code == 422
        assert (
            c.get("/api/v1/songs/search", params={"q": "x", "model_name": "clap"}).status_code
            == 422
        )


def test_search_embedder_unavailable(factory: sessionmaker[Session]) -> None:
    settings = make_settings("muq_mulan")
    with TestClient(
        create_app(settings, session_factory=factory, text_embedder=FakeEmbedder(None))
    ) as c:
        resp = c.get("/api/v1/songs/search", params={"q": "x"})
        assert resp.status_code == 503
        assert resp.json()["detail"] == (
            "text search needs the engine next to the server, which is not reachable"
        )


def test_analysis(factory: sessionmaker[Session]) -> None:
    settings = make_settings("muq", "acoustic")
    muq = profile_of("muq", factory, settings)
    aco = profile_of("acoustic", factory, settings)
    with factory() as session:
        activate_profiles(factory, settings)
        tid = add_track(session, "p/q.flac", title="T", artist="A", genre="G")
        add_analysis(session, tid, "muq", muq, "done")
        session.add(AnalysisDetail(track_id=tid, model="muq", profile=muq, details={"k": 1}))
        add_analysis(
            session, tid, "acoustic", aco, "leased", lease_until=datetime.now(UTC) + timedelta(1)
        )
        add_track(session, "p/pending.flac")
        session.commit()
    with TestClient(create_app(settings, session_factory=factory)) as c:
        body = c.get(f"/api/v1/songs/{quote('p/q.flac', safe='')}/analysis").json()
        assert body["file_path"] == "p/q.flac"
        assert body["title"] == "T"
        assert body["duration_sec"] == 100.0
        assert body["analyses"]["muq"] == {"profile": muq, "state": "done", "details": {"k": 1}}
        assert body["analyses"]["acoustic"] == {"profile": aco, "state": "leased", "details": None}
        pend = c.get("/api/v1/songs/p%2Fpending.flac/analysis").json()
        assert pend["analyses"]["muq"]["state"] == "pending"
        assert c.get("/api/v1/songs/nope.flac/analysis").status_code == 404


def test_stats(factory: sessionmaker[Session]) -> None:
    settings = make_settings("muq", "acoustic")
    muq = profile_of("muq", factory, settings)
    aco = profile_of("acoustic", factory, settings)
    with factory() as session:
        activate_profiles(factory, settings)
        ids = [add_track(session, f"t/{i}.flac") for i in range(4)]
        add_analysis(session, ids[0], "muq", muq, "done")
        add_analysis(session, ids[1], "muq", muq, "done")
        add_analysis(session, ids[2], "muq", muq, "failed")
        add_analysis(session, ids[0], "acoustic", aco, "done")
        add_analysis(session, ids[0], "muq", "r0-old", "done")  # another profile: not counted
        session.commit()
    with TestClient(create_app(settings, session_factory=factory)) as c:
        body = c.get("/api/v1/stats").json()
        assert body["total_songs"] == 4
        assert body["songs_with_muq"] == 2
        assert body["songs_with_acoustic"] == 1
        assert body["songs_with_clap"] == 0
        for model in models.MODELS:
            assert f"songs_with_{model}" in body
        assert body["pending_tasks"] == 0
        assert body["running_tasks"] == 0
        assert body["analysis"]["muq"] == {
            "profile": muq,
            "done": 2,
            "failed": 1,
            "leased": 0,
            "pending": 1,
        }
        assert set(body["analysis"]) == {"muq", "acoustic"}


def test_analyze_run_retries_failed(factory: sessionmaker[Session]) -> None:
    settings = make_settings("muq", "acoustic")
    muq = profile_of("muq", factory, settings)
    with factory() as session:
        activate_profiles(factory, settings)
        tid = add_track(session, "f/x.flac")
        add_analysis(session, tid, "muq", muq, "failed", error="boom")
        session.commit()
    with TestClient(create_app(settings, session_factory=factory)) as c:
        resp = c.post("/api/v1/batch/analyze/run")
        assert resp.status_code == 200
        run = resp.json()
        assert run["kind"] == "analyze"
        for _ in range(100):
            got = c.get(f"/api/v1/batch/tasks/{run['id']}").json()
            if got["state"] in ("succeeded", "failed"):
                break
            time.sleep(0.05)
        assert got["state"] == "succeeded", got
        assert got["detail"] == {"muq": 1, "acoustic": 0}
        stats = c.get("/api/v1/stats").json()
        assert stats["analysis"]["muq"]["failed"] == 0
        assert stats["analysis"]["muq"]["pending"] == 1
