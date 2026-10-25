"""Tests for runtime-managed vector tables and HNSW indexes."""

from __future__ import annotations

import pytest
from alembic import command
from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine, inspect, text

from lainbow.core.method import AnalysisMethod
from lainbow.core.models import MODELS, ModelId, ModelSpec, Purpose
from lainbow.db.migrate import get_alembic_config, upgrade_to_head
from lainbow.db.vectors import (
    ensure_vector_store,
    vector_index_name,
    vector_table,
)

# ---------------------------------------------------------------------------
# Unit tests (no database required)
# ---------------------------------------------------------------------------


def test_vector_table_fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """vector_table for a fake model monkeypatched into MODELS (8 dim) has embedding dim 8."""
    fake_spec = ModelSpec(
        id="fake_model",
        dimension=8,
        sample_rate=16000,
        clip_sec=10.0,
        revision=1,
        description="Fake 8-dim model for testing",
        short_name="Fake",
        purposes=frozenset({Purpose.SIMILAR}),
        default_method=AnalysisMethod(max_sec=180, drop_tail=False),
    )
    fake_models: dict[str, ModelSpec] = {str(k): v for k, v in MODELS.items()}
    fake_models["fake_model"] = fake_spec
    monkeypatch.setattr("lainbow.core.models.MODELS", fake_models)

    tbl = vector_table("fake_model")
    emb_type = tbl.c.embedding.type
    assert isinstance(emb_type, Vector)
    assert emb_type.dim == 8
    assert tbl.name == "vectors_fake_model"


def test_vector_table_dimensions_standard_models() -> None:
    """vector_table(model) embedding dim matches spec.dimension for all standard models."""
    for model_id, spec in MODELS.items():
        tbl = vector_table(model_id)
        emb_type = tbl.c.embedding.type
        assert isinstance(emb_type, Vector)
        assert emb_type.dim == spec.dimension


def test_vector_index_name_rules() -> None:
    """Index name respects 63-byte limit and profile hash uniqueness."""
    long_profile_1 = "a" * 50 + "-1"
    long_profile_2 = "a" * 50 + "-2"

    name_1 = vector_index_name(ModelId.MUQ, long_profile_1)
    name_2 = vector_index_name(ModelId.MUQ, long_profile_2)

    assert len(name_1) <= 63
    assert len(name_2) <= 63
    assert name_1 != name_2


def test_vector_index_name_rejects_invalid_profile() -> None:
    """vector_index_name validates profile pattern."""
    for invalid in ["bad'profile", "bad profile", "UPPER_CASE", "semi;colon", ""]:
        with pytest.raises(ValueError, match="Invalid profile name"):
            vector_index_name("muq", invalid)


# ---------------------------------------------------------------------------
# Postgres integration tests (@pytest.mark.postgres)
# ---------------------------------------------------------------------------


@pytest.mark.postgres
def test_ensure_vector_store_fake_model_and_cascade(
    clean_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ensure_vector_store creates table/index, is idempotent, and cascades on track delete."""
    fake_spec = ModelSpec(
        id="fake_casc",
        dimension=8,
        sample_rate=16000,
        clip_sec=10.0,
        revision=1,
        description="Fake cascade test model",
        short_name="FakeCasc",
        purposes=frozenset({Purpose.SIMILAR}),
        default_method=AnalysisMethod(max_sec=180, drop_tail=False),
    )
    fake_models: dict[str, ModelSpec] = {str(k): v for k, v in MODELS.items()}
    fake_models["fake_casc"] = fake_spec
    monkeypatch.setattr("lainbow.core.models.MODELS", fake_models)

    upgrade_to_head(clean_db)
    engine = create_engine(clean_db)

    with engine.begin() as conn:
        # First creation creates table and index
        ensure_vector_store(conn, "fake_casc", "test-profile")
        # Running a second time is a no-op
        ensure_vector_store(conn, "fake_casc", "test-profile")

    with engine.connect() as conn:
        inspector = inspect(conn)
        assert "vectors_fake_casc" in inspector.get_table_names()
        indexes = {idx["name"] for idx in inspector.get_indexes("vectors_fake_casc")}
        expected_index = vector_index_name("fake_casc", "test-profile")
        assert expected_index in indexes

    # Test track deletion cascade
    with engine.begin() as conn:
        res = conn.execute(
            text(
                "INSERT INTO tracks (path, duration_sec, mtime, size_bytes, seen_at) "
                "VALUES ('test/cascade.flac', 120.0, now(), 54321, now()) RETURNING id"
            )
        )
        track_id = res.scalar_one()

        conn.execute(
            text(
                "INSERT INTO vectors_fake_casc (track_id, profile, embedding) "
                "VALUES (:t_id, 'test-profile', array_fill(0.0, ARRAY[8]))"
            ),
            {"t_id": track_id},
        )

        assert (
            conn.execute(
                text("SELECT count(*) FROM vectors_fake_casc WHERE track_id = :t_id"),
                {"t_id": track_id},
            ).scalar()
            == 1
        )

        conn.execute(text("DELETE FROM tracks WHERE id = :t_id"), {"t_id": track_id})

        assert (
            conn.execute(
                text("SELECT count(*) FROM vectors_fake_casc WHERE track_id = :t_id"),
                {"t_id": track_id},
            ).scalar()
            == 0
        )


@pytest.mark.postgres
def test_alembic_check_ignores_runtime_vector_store(clean_db: str) -> None:
    """alembic check reports no diff after upgrade_to_head and ensure_vector_store for muq."""
    upgrade_to_head(clean_db)
    engine = create_engine(clean_db)
    with engine.begin() as conn:
        ensure_vector_store(conn, "muq", "default")

    config = get_alembic_config(clean_db)
    command.check(config)
