"""Tests for neighbor statistics computation and management (lainbow.db.neighbors)."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, create_engine, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from lainbow.core.method import AnalysisMethod
from lainbow.core.models import MODELS, ModelSpec, Purpose
from lainbow.db import neighbors
from lainbow.db.migrate import upgrade_to_head
from lainbow.db.neighbors import plan_refresh, profile_stats, refresh
from lainbow.db.schema import NeighborStat
from lainbow.db.session import make_session_factory
from lainbow.db.vectors import ensure_vector_store, vector_table

pytestmark = pytest.mark.postgres


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> str:
    """Register a fake low-dimensional model for fast, deterministic testing."""
    model_name = "fake_nb"
    spec = ModelSpec(
        id=model_name,
        dimension=4,
        sample_rate=16000,
        clip_sec=10.0,
        revision=1,
        description="Fake model for neighbor tests",
        short_name="FakeNB",
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
    """Database engine with head migrations and vector tables created."""
    upgrade_to_head(clean_db)
    eng = create_engine(clean_db)
    with eng.begin() as conn:
        for profile in ("p1", "p2"):
            ensure_vector_store(conn, fake_model, profile)
    yield eng
    eng.dispose()


def add_tracks(engine: Engine, count: int) -> list[int]:
    """Insert tracks into database and return their generated IDs."""
    with Session(engine) as session:
        ids = list(
            session.execute(
                text(
                    "INSERT INTO tracks (path, duration_sec, mtime, size_bytes, seen_at) "
                    "SELECT '/track/' || g || '-' || gen_random_uuid(), 120.0, now(), 1000, now() "
                    "FROM generate_series(1, :count) g RETURNING id"
                ),
                {"count": count},
            ).scalars()
        )
        session.commit()
        return ids


def add_vectors(
    engine: Engine,
    model: str,
    profile: str,
    vectors: dict[int, list[float]],
) -> None:
    """Insert vector embeddings for tracks."""
    tbl = vector_table(model)
    records = [
        {"track_id": tid, "profile": profile, "embedding": vec} for tid, vec in vectors.items()
    ]
    with Session(engine) as session:
        session.execute(pg_insert(tbl).values(records))
        session.commit()


def cosine_similarity(u: list[float], v: list[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    dot = sum(x * y for x, y in zip(u, v, strict=True))
    norm_u = math.sqrt(sum(x * x for x in u))
    norm_v = math.sqrt(sum(y * y for y in v))
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    return dot / (norm_u * norm_v)


def test_plan_refresh_cycle(
    db_engine: Engine,
    fake_model: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """plan_refresh tests all transitions according to DoD:

    - n < MIN_VECTORS -> 'none'
    - no stats -> 'full'
    - after full -> 'none'
    - 1 track added -> 'missing'
    - 1.25x increase -> 'full'
    - changed k -> 'full'
    - k=0 -> 'none'
    """
    monkeypatch.setattr(neighbors, "MIN_VECTORS", 20)
    session_factory = make_session_factory(db_engine)

    # 1. Start with 10 tracks (< MIN_VECTORS=20)
    tids_10 = add_tracks(db_engine, 10)
    rng = random.Random(42)
    vecs_10 = {tid: [rng.uniform(-1, 1) for _ in range(4)] for tid in tids_10}
    add_vectors(db_engine, fake_model, "p1", vecs_10)

    with session_factory() as session:
        # k=0 -> none
        assert plan_refresh(session, model=fake_model, profile="p1", k=0) == "none"
        # n < MIN_VECTORS (10 < 20) -> none
        assert plan_refresh(session, model=fake_model, profile="p1", k=5) == "none"

    # 2. Add 10 more tracks so n=20 (>= MIN_VECTORS)
    tids_next_10 = add_tracks(db_engine, 10)
    vecs_next_10 = {tid: [rng.uniform(-1, 1) for _ in range(4)] for tid in tids_next_10}
    add_vectors(db_engine, fake_model, "p1", vecs_next_10)

    with session_factory() as session:
        # No stats in neighbor_profiles -> full
        assert plan_refresh(session, model=fake_model, profile="p1", k=5) == "full"

    # 3. Perform full refresh
    res = refresh(session_factory, model=fake_model, profile="p1", k=5, kind="full")
    assert res["computed"] == 20

    with session_factory() as session:
        # After full -> none
        assert plan_refresh(session, model=fake_model, profile="p1", k=5) == "none"
        # Changed k -> full
        assert plan_refresh(session, model=fake_model, profile="p1", k=6) == "full"

    # 4. Add 1 track with vector (n=21, 21 < 1.25 * 20 = 25)
    tid_single = add_tracks(db_engine, 1)[0]
    add_vectors(db_engine, fake_model, "p1", {tid_single: [rng.uniform(-1, 1) for _ in range(4)]})

    with session_factory() as session:
        # Missing stats for the new track -> missing
        assert plan_refresh(session, model=fake_model, profile="p1", k=5) == "missing"

    # 5. Add more tracks to reach 25 (n=25 >= 1.25 * 20 = 25)
    tids_grow = add_tracks(db_engine, 4)
    vecs_grow = {tid: [rng.uniform(-1, 1) for _ in range(4)] for tid in tids_grow}
    add_vectors(db_engine, fake_model, "p1", vecs_grow)

    with session_factory() as session:
        # 1.25x increase -> full
        assert plan_refresh(session, model=fake_model, profile="p1", k=5) == "full"


def test_refresh_full_accuracy_and_profiles(
    db_engine: Engine,
    fake_model: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """refresh('full') computes r values matching brute-force cosine similarity."""
    monkeypatch.setattr(neighbors, "CHUNK", 8)  # test chunking logic
    session_factory = make_session_factory(db_engine)

    n_tracks = 24
    k = 5
    tids = add_tracks(db_engine, n_tracks)
    rng = random.Random(123)
    vec_map: dict[int, list[float]] = {
        tid: [rng.uniform(-1.0, 1.0) for _ in range(4)] for tid in tids
    }
    add_vectors(db_engine, fake_model, "p1", vec_map)

    # Brute-force calculation in Python
    expected_r: dict[int, float] = {}
    for tid, u in vec_map.items():
        sims: list[float] = []
        for other_tid, v in vec_map.items():
            if other_tid == tid:
                continue
            sims.append(cosine_similarity(u, v))
        sims.sort(reverse=True)
        top_k = sims[:k]
        expected_r[tid] = sum(top_k) / len(top_k)

    expected_mean_r = sum(expected_r.values()) / len(expected_r)

    # Execute refresh("full")
    res = refresh(session_factory, model=fake_model, profile="p1", k=k, kind="full")
    assert res == {"computed": n_tracks}

    with session_factory() as session:
        stats = session.scalars(
            select(NeighborStat).where(
                NeighborStat.model == fake_model,
                NeighborStat.profile == "p1",
            )
        ).all()
        assert len(stats) == n_tracks

        for stat in stats:
            exp = expected_r[stat.track_id]
            # pgvector HNSW with ef_search >= 100 on 24 tracks gives exact nearest neighbors
            assert stat.r == pytest.approx(exp, abs=1e-3)

        prof = profile_stats(session, model=fake_model, profile="p1")
        assert prof is not None
        assert prof.k == k
        assert prof.vectors == n_tracks
        assert prof.mean_r == pytest.approx(expected_mean_r, abs=1e-3)


def test_refresh_missing(
    db_engine: Engine,
    fake_model: str,
) -> None:
    """refresh('missing') calculates only missing tracks and leaves neighbor_profiles untouched."""
    session_factory = make_session_factory(db_engine)
    k = 3

    tids = add_tracks(db_engine, 10)
    rng = random.Random(456)
    vec_map = {tid: [rng.uniform(-1.0, 1.0) for _ in range(4)] for tid in tids}
    add_vectors(db_engine, fake_model, "p1", vec_map)

    # Full refresh
    refresh(session_factory, model=fake_model, profile="p1", k=k, kind="full")

    with session_factory() as session:
        initial_prof = profile_stats(session, model=fake_model, profile="p1")
        assert initial_prof is not None
        initial_computed_at = initial_prof.computed_at
        initial_mean_r = initial_prof.mean_r
        initial_vectors = initial_prof.vectors

    # Add 2 new tracks with vectors
    new_tids = add_tracks(db_engine, 2)
    new_vecs = {tid: [rng.uniform(-1.0, 1.0) for _ in range(4)] for tid in new_tids}
    add_vectors(db_engine, fake_model, "p1", new_vecs)

    # Missing refresh
    res = refresh(session_factory, model=fake_model, profile="p1", k=k, kind="missing")
    assert res == {"computed": 2}

    with session_factory() as session:
        # All 12 tracks now have stats
        all_stats = session.scalars(
            select(NeighborStat).where(
                NeighborStat.model == fake_model,
                NeighborStat.profile == "p1",
            )
        ).all()
        assert len(all_stats) == 12
        stat_tids = {s.track_id for s in all_stats}
        assert set(new_tids).issubset(stat_tids)

        # neighbor_profiles was NOT updated
        current_prof = profile_stats(session, model=fake_model, profile="p1")
        assert current_prof is not None
        assert current_prof.computed_at == initial_computed_at
        assert current_prof.mean_r == initial_mean_r
        assert current_prof.vectors == initial_vectors


def test_refresh_none(
    db_engine: Engine,
    fake_model: str,
) -> None:
    """refresh('none') does nothing."""
    session_factory = make_session_factory(db_engine)
    res = refresh(session_factory, model=fake_model, profile="p1", k=5, kind="none")
    assert res == {"computed": 0}


def test_delete_track_cascades_to_neighbor_stats(
    db_engine: Engine,
    fake_model: str,
) -> None:
    """Deleting a track cascades and removes its neighbor_stats row."""
    session_factory = make_session_factory(db_engine)
    tids = add_tracks(db_engine, 5)
    rng = random.Random(789)
    vecs = {tid: [rng.uniform(-1, 1) for _ in range(4)] for tid in tids}
    add_vectors(db_engine, fake_model, "p1", vecs)

    refresh(session_factory, model=fake_model, profile="p1", k=3, kind="full")

    with session_factory() as session:
        stat = session.scalar(select(NeighborStat).where(NeighborStat.track_id == tids[0]))
        assert stat is not None

    with Session(db_engine) as session:
        session.execute(text("DELETE FROM tracks WHERE id = :id"), {"id": tids[0]})
        session.commit()

    with session_factory() as session:
        assert session.scalar(select(NeighborStat).where(NeighborStat.track_id == tids[0])) is None
        # Other tracks remain
        remaining = session.scalars(select(NeighborStat).where(NeighborStat.profile == "p1")).all()
        assert len(remaining) == 4


def test_profile_isolation(
    db_engine: Engine,
    fake_model: str,
) -> None:
    """Operations on one profile do not affect another profile."""
    session_factory = make_session_factory(db_engine)
    tids_p1 = add_tracks(db_engine, 5)
    tids_p2 = add_tracks(db_engine, 5)

    rng = random.Random(999)
    vecs_p1 = {tid: [rng.uniform(-1, 1) for _ in range(4)] for tid in tids_p1}
    vecs_p2 = {tid: [rng.uniform(-1, 1) for _ in range(4)] for tid in tids_p2}

    add_vectors(db_engine, fake_model, "p1", vecs_p1)
    add_vectors(db_engine, fake_model, "p2", vecs_p2)

    refresh(session_factory, model=fake_model, profile="p1", k=3, kind="full")

    with session_factory() as session:
        p1_stats = session.scalars(select(NeighborStat).where(NeighborStat.profile == "p1")).all()
        assert len(p1_stats) == 5

        p2_stats = session.scalars(select(NeighborStat).where(NeighborStat.profile == "p2")).all()
        assert len(p2_stats) == 0

        p1_prof = profile_stats(session, model=fake_model, profile="p1")
        assert p1_prof is not None
        p2_prof = profile_stats(session, model=fake_model, profile="p2")
        assert p2_prof is None


def test_refresh_isolated_single_track_null_r(
    db_engine: Engine,
    fake_model: str,
) -> None:
    """When a track has no other neighbors, r is NULL and the track is not stored."""
    session_factory = make_session_factory(db_engine)
    tid = add_tracks(db_engine, 1)[0]
    add_vectors(db_engine, fake_model, "p1", {tid: [1.0, 0.0, 0.0, 0.0]})

    res = refresh(session_factory, model=fake_model, profile="p1", k=3, kind="full")
    assert res == {"computed": 0}

    with session_factory() as session:
        stats = session.scalars(select(NeighborStat).where(NeighborStat.profile == "p1")).all()
        assert len(stats) == 0
        prof = profile_stats(session, model=fake_model, profile="p1")
        assert prof is not None
        assert prof.vectors == 1
        assert prof.mean_r == 0.0
