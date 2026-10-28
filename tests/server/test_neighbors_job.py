"""Tests for background neighbor stats refresh job and trigger conditions."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from lainbow.core.method import AnalysisMethod
from lainbow.core.models import MODELS, ModelSpec, Purpose
from lainbow.db import neighbors as db_neighbors
from lainbow.db.migrate import upgrade_to_head
from lainbow.db.schema import NeighborProfile, NeighborStat, Track
from lainbow.db.session import make_session_factory
from lainbow.db.vectors import ensure_vector_store, vector_table
from lainbow.db.work import Task
from lainbow.server.neighbors import make_neighbors_job, needs_refresh

pytestmark = pytest.mark.postgres


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> str:
    """Register a fake low-dimensional model for neighbor refresh testing."""
    model_name = "fake_srv_nb"
    spec = ModelSpec(
        id=model_name,
        dimension=4,
        sample_rate=16000,
        clip_sec=10.0,
        revision=1,
        description="Fake model for server neighbor tests",
        short_name="FakeSrvNB",
        purposes=frozenset({Purpose.SIMILAR}),
        default_method=AnalysisMethod(max_sec=180, drop_tail=False),
        memory_gb=1.0,
    )
    fake_models = {str(k): v for k, v in MODELS.items()}
    fake_models[model_name] = spec
    monkeypatch.setattr("lainbow.core.models.MODELS", fake_models)
    return model_name


@pytest.fixture
def db_engine(clean_db: str, fake_model: str) -> Iterator[Engine]:
    upgrade_to_head(clean_db)
    eng = create_engine(clean_db)
    with eng.begin() as conn:
        ensure_vector_store(conn, fake_model, "p1")
    yield eng
    eng.dispose()


@pytest.fixture
def factory(db_engine: Engine) -> sessionmaker[Session]:
    return make_session_factory(db_engine)


def add_track(session: Session, path: str) -> int:
    now = datetime.now(UTC)
    track = Track(path=path, duration_sec=120.0, mtime=now, size_bytes=1000, seen_at=now)
    session.add(track)
    session.flush()
    return track.id


def add_vector(session: Session, model: str, profile: str, track_id: int, vec: list[float]) -> None:
    tbl = vector_table(model)
    session.execute(tbl.insert().values(track_id=track_id, profile=profile, embedding=vec))


def test_needs_refresh_and_job_execution(
    factory: sessionmaker[Session], fake_model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Lower MIN_VECTORS threshold for fast testing
    monkeypatch.setattr(db_neighbors, "MIN_VECTORS", 3)

    profile = "p1"
    tasks = (Task(model=fake_model, profile=profile),)

    # 1. Initially no tracks/vectors -> under MIN_VECTORS -> False
    assert not needs_refresh(factory, tasks, k=2)

    # 2. Add 10 tracks with vectors (>= MIN_VECTORS 3, but no neighbor stats)
    with factory() as session:
        for i in range(10):
            tid = add_track(session, f"/track{i}.flac")
            add_vector(session, fake_model, profile, tid, [float(i % 4 == j) for j in range(4)])
        session.commit()

    # needs_refresh is now True
    assert needs_refresh(factory, tasks, k=2)

    # If k <= 0, needs_refresh is False
    assert not needs_refresh(factory, tasks, k=0)

    # 3. Execute make_neighbors_job
    job = make_neighbors_job(factory, tasks, k=2)
    result = job()

    assert fake_model in result
    assert result[fake_model]["profile"] == profile
    assert result[fake_model]["kind"] == "full"
    assert result[fake_model]["computed"] == 10

    # 4. Verify DB was populated
    with factory() as session:
        prof = session.get(NeighborProfile, (fake_model, profile))
        assert prof is not None
        assert prof.k == 2
        assert prof.vectors == 10

        stats = session.query(NeighborStat).filter_by(model=fake_model, profile=profile).all()
        assert len(stats) == 10

    # 5. Calling needs_refresh again returns False
    assert not needs_refresh(factory, tasks, k=2)

    # 6. Adding an 11th track creates missing stats (11 < 10 * 1.25) -> needs_refresh becomes True
    with factory() as session:
        tid11 = add_track(session, "/track11.flac")
        add_vector(session, fake_model, profile, tid11, [0.5, 0.5, 0.0, 0.0])
        session.commit()

    assert needs_refresh(factory, tasks, k=2)

    # Running job updates only the missing track
    result_missing = job()
    assert result_missing[fake_model]["kind"] == "missing"
    assert result_missing[fake_model]["computed"] == 1

    # needs_refresh is False again
    assert not needs_refresh(factory, tasks, k=2)
