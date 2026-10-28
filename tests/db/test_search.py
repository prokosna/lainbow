"""Tests for vector search and CSLS ranking (lainbow.db.search)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from lainbow.core.method import AnalysisMethod
from lainbow.core.models import MODELS, ModelSpec, Purpose
from lainbow.db.migrate import upgrade_to_head
from lainbow.db.schema import NeighborProfile, NeighborStat, Track
from lainbow.db.search import nearest, track_by_path, vector_of
from lainbow.db.vectors import ensure_vector_store, vector_table

pytestmark = pytest.mark.postgres


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> str:
    """Register a fake low-dimensional model for deterministic search testing."""
    model_name = "fake_search"
    spec = ModelSpec(
        id=model_name,
        dimension=4,
        sample_rate=16000,
        clip_sec=10.0,
        revision=1,
        description="Fake model for search tests",
        short_name="FakeSearch",
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


def add_track(session: Session, path: str) -> int:
    now = datetime.now(UTC)
    track = Track(path=path, duration_sec=120.0, mtime=now, size_bytes=1000, seen_at=now)
    session.add(track)
    session.flush()
    return track.id


def add_vector(session: Session, model: str, profile: str, track_id: int, vec: list[float]) -> None:
    tbl = vector_table(model)
    session.execute(tbl.insert().values(track_id=track_id, profile=profile, embedding=vec))


def test_search_csls_ranking_and_fallback(db_engine: Engine, fake_model: str) -> None:
    profile = "p1"
    query_vec = [1.0, 0.0, 0.0, 0.0]

    with Session(db_engine) as session:
        # Create seed and candidate tracks
        seed_id = add_track(session, "/seed.flac")
        hub_id = add_track(session, "/hub.flac")
        match_id = add_track(session, "/match.flac")
        nostat_id = add_track(session, "/nostat.flac")

        # Cosine similarities to query_vec:
        # Seed:   1.00
        # Hub:    0.95
        # Match:  0.90
        # NoStat: 0.80
        add_vector(session, fake_model, profile, seed_id, [1.0, 0.0, 0.0, 0.0])
        add_vector(session, fake_model, profile, hub_id, [0.95, 0.3122498999, 0.0, 0.0])
        add_vector(session, fake_model, profile, match_id, [0.90, 0.4358898944, 0.0, 0.0])
        add_vector(session, fake_model, profile, nostat_id, [0.80, 0.60, 0.0, 0.0])
        session.commit()

        # (a) & (f): Without stats, standard cosine order, seed track excluded
        hits = nearest(
            session,
            fake_model,
            profile,
            query_vec,
            limit=10,
            exclude_track_id=seed_id,
            csls_k=0,
        )
        assert [h.track_id for h in hits] == [hub_id, match_id, nostat_id]
        assert seed_id not in [h.track_id for h in hits]

        # Insert neighbor_profiles and neighbor_stats:
        # mean_r = 0.3
        # Hub r = 0.8 -> CSLS score = 2 * 0.95 - 0.8 = 1.10
        # Match r = 0.2 -> CSLS score = 2 * 0.90 - 0.2 = 1.60
        # NoStat has no stat -> uses mean_r (0.3) -> CSLS score = 2 * 0.80 - 0.3 = 1.30
        session.add(
            NeighborProfile(
                model=fake_model,
                profile=profile,
                k=50,
                vectors=4,
                mean_r=0.3,
            )
        )
        session.add(NeighborStat(model=fake_model, profile=profile, track_id=hub_id, r=0.8))
        session.add(NeighborStat(model=fake_model, profile=profile, track_id=match_id, r=0.2))
        session.commit()

        # (b), (d), (e), (f): CSLS ranking with k=50
        # Expected CSLS order: Match (1.60), NoStat (1.30), Hub (1.10)
        csls_hits = nearest(
            session,
            fake_model,
            profile,
            query_vec,
            limit=10,
            exclude_track_id=seed_id,
            csls_k=50,
        )
        assert [h.track_id for h in csls_hits] == [match_id, nostat_id, hub_id]

        # (e): score remains cosine similarity
        assert csls_hits[0].score == pytest.approx(0.90, abs=1e-3)
        assert csls_hits[1].score == pytest.approx(0.80, abs=1e-3)
        assert csls_hits[2].score == pytest.approx(0.95, abs=1e-3)

        # (c): neighbor_profiles.k (50) != csls_k (20) -> falls back to cosine order
        mismatched_hits = nearest(
            session,
            fake_model,
            profile,
            query_vec,
            limit=10,
            exclude_track_id=seed_id,
            csls_k=20,
        )
        assert [h.track_id for h in mismatched_hits] == [hub_id, match_id, nostat_id]

        # Missing profile stats on p2 -> falls back to cosine order
        p2_track_id = add_track(session, "/p2.flac")
        add_vector(session, fake_model, "p2", p2_track_id, [1.0, 0.0, 0.0, 0.0])
        session.commit()
        p2_hits = nearest(
            session,
            fake_model,
            "p2",
            query_vec,
            limit=10,
            csls_k=50,
        )
        assert len(p2_hits) == 1
        assert p2_hits[0].track_id == p2_track_id


def test_search_helpers(db_engine: Engine, fake_model: str) -> None:
    profile = "p1"
    with Session(db_engine) as session:
        tid = add_track(session, "/path/track.flac")
        add_vector(session, fake_model, profile, tid, [0.1, 0.2, 0.3, 0.4])
        session.commit()

        # track_by_path
        track = track_by_path(session, "/path/track.flac")
        assert track is not None
        assert track.id == tid
        assert track_by_path(session, "/nonexistent") is None

        # vector_of
        vec = vector_of(session, fake_model, profile, tid)
        assert vec is not None
        assert len(vec) == 4
        assert vec[0] == pytest.approx(0.1, abs=1e-3)
        assert vector_of(session, fake_model, profile, 999999) is None
