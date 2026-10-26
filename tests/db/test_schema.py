"""Tests for core database schema and migrations."""

from __future__ import annotations

import uuid

import pytest
from alembic import command
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.orm import Session

from lainbow.db.migrate import get_alembic_config, upgrade_to_head
from lainbow.db.schema import (
    AnalysisDetail,
    RegisteredEngine,
)

# ---------------------------------------------------------------------------
# Unit tests (no database required)
# ---------------------------------------------------------------------------


def test_registered_engine_tablename() -> None:
    """RegisteredEngine ORM class maps to 'engines' table."""
    assert RegisteredEngine.__tablename__ == "engines"
    assert hasattr(RegisteredEngine, "version")


def test_analysis_detail_tablename() -> None:
    """AnalysisDetail ORM class maps to 'analysis_details' table."""
    assert AnalysisDetail.__tablename__ == "analysis_details"


# ---------------------------------------------------------------------------
# Postgres integration tests (@pytest.mark.postgres)
# ---------------------------------------------------------------------------


@pytest.mark.postgres
def test_upgrade_to_head_creates_all_tables_and_is_idempotent(clean_db: str) -> None:
    """upgrade_to_head creates all migration tables and returns revision id."""
    rev = upgrade_to_head(clean_db)
    assert rev == "0002_engine_version"

    engine = create_engine(clean_db)
    with engine.connect() as conn:
        inspector = inspect(conn)
        tables = set(inspector.get_table_names())
        expected_tables = {
            "tracks",
            "engines",
            "analyses",
            "analysis_details",
            "runs",
            "active_profiles",
            "alembic_version",
        }
        assert tables == expected_tables

        # Check version column exists on engines table and is nullable
        cols = {c["name"]: c for c in inspector.get_columns("engines")}
        assert "version" in cols
        assert cols["version"]["nullable"] is True

    # Second run should be a no-op and succeed without errors
    rev2 = upgrade_to_head(clean_db)
    assert rev2 == "0002_engine_version"


@pytest.mark.postgres
def test_delete_track_cascades(clean_db: str) -> None:
    """Deleting a track cascades to analyses and analysis_details."""
    upgrade_to_head(clean_db)

    engine = create_engine(clean_db)
    with engine.begin() as conn:
        res = conn.execute(
            text(
                "INSERT INTO tracks (path, duration_sec, mtime, size_bytes, seen_at) "
                "VALUES ('test/track.flac', 180.5, now(), 123456, now()) RETURNING id"
            )
        )
        track_id = res.scalar_one()

        conn.execute(
            text(
                "INSERT INTO analyses (track_id, model, profile, state) "
                "VALUES (:t_id, 'mert', 'profile-1', 'done')"
            ),
            {"t_id": track_id},
        )
        conn.execute(
            text(
                "INSERT INTO analysis_details (track_id, model, profile, details) "
                "VALUES (:t_id, 'mert', 'profile-1', '{\"bpm\": 120}'::jsonb)"
            ),
            {"t_id": track_id},
        )

        assert (
            conn.execute(
                text("SELECT count(*) FROM analyses WHERE track_id = :t_id"),
                {"t_id": track_id},
            ).scalar()
            == 1
        )
        assert (
            conn.execute(
                text("SELECT count(*) FROM analysis_details WHERE track_id = :t_id"),
                {"t_id": track_id},
            ).scalar()
            == 1
        )

        conn.execute(text("DELETE FROM tracks WHERE id = :t_id"), {"t_id": track_id})

        assert (
            conn.execute(
                text("SELECT count(*) FROM analyses WHERE track_id = :t_id"),
                {"t_id": track_id},
            ).scalar()
            == 0
        )
        assert (
            conn.execute(
                text("SELECT count(*) FROM analysis_details WHERE track_id = :t_id"),
                {"t_id": track_id},
            ).scalar()
            == 0
        )


@pytest.mark.postgres
def test_registered_engine_orm_crud(clean_db: str) -> None:
    """RegisteredEngine ORM can insert and query engines."""
    upgrade_to_head(clean_db)

    engine = create_engine(clean_db)
    with Session(engine) as session:
        eng_id = uuid.uuid4()
        eng = RegisteredEngine(
            id=eng_id,
            name="worker-1",
            token_hash="hash123",
        )
        session.add(eng)
        session.commit()

        loaded = session.scalars(
            select(RegisteredEngine).where(RegisteredEngine.id == eng_id)
        ).one()
        assert loaded.name == "worker-1"


@pytest.mark.postgres
def test_alembic_check_reports_no_diff(clean_db: str) -> None:
    """Alembic check reports no differences between the models and the migrated database."""
    upgrade_to_head(clean_db)
    config = get_alembic_config(clean_db)
    command.check(config)
