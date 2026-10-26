"""Tests for the admin command."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from lainbow.db.schema import RegisteredEngine
from lainbow.server import admin, engines


@pytest.fixture
def env(clean_db: str, monkeypatch: pytest.MonkeyPatch) -> str:
    for key in list(os.environ):
        if key.startswith("LAINBOW_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("LAINBOW_DATABASE_URL", clean_db)
    return clean_db


@pytest.mark.postgres
def test_add_engine_prints_token(env: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert admin.main(["add-engine", "gpu1"]) == 0
    token = capsys.readouterr().out.strip()
    assert token.startswith("lbe_")
    eng = create_engine(env)
    with Session(eng) as s:
        assert engines.authenticate(s, token) is not None
    eng.dispose()

    assert admin.main(["list-engines"]) == 0
    assert "gpu1" in capsys.readouterr().out


@pytest.mark.postgres
def test_duplicate_name_fails(env: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert admin.main(["add-engine", "gpu1"]) == 0
    capsys.readouterr()
    assert admin.main(["add-engine", "gpu1"]) != 0
    assert "already exists" in capsys.readouterr().err


@pytest.mark.postgres
def test_remove_engine(env: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert admin.main(["add-engine", "gpu1"]) == 0
    assert admin.main(["remove-engine", "gpu1"]) == 0
    assert admin.main(["remove-engine", "gpu1"]) != 0


def test_missing_settings_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("LAINBOW_DATABASE_URL", raising=False)
    assert admin.main(["list-engines"]) == 2
    assert capsys.readouterr().err


@pytest.mark.postgres
def test_status_shows_counts(env: str, capsys: pytest.CaptureFixture[str]) -> None:
    from datetime import UTC, datetime

    from lainbow.db.schema import ActiveProfile, Analysis, Track

    assert admin.main(["add-engine", "gpu1"]) == 0
    capsys.readouterr()
    now = datetime.now(UTC)
    eng = create_engine(env)
    with Session(eng) as s:
        s.add_all(
            Track(path=f"a/{i}.flac", duration_sec=100.0, mtime=now, size_bytes=1, seen_at=now)
            for i in range(4)
        )
        s.add(ActiveProfile(model="muq", profile="p1"))
        s.flush()
        ids = [t.id for t in s.query(Track).order_by(Track.id)]
        s.add(Analysis(track_id=ids[0], model="muq", profile="p1", state="done"))
        s.add(Analysis(track_id=ids[1], model="muq", profile="p1", state="done"))
        s.add(Analysis(track_id=ids[2], model="muq", profile="p1", state="failed"))
        s.commit()
    eng.dispose()

    assert admin.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "Tracks: 4" in out
    muq = next(line for line in out.splitlines() if line.startswith("muq"))
    assert muq.split() == ["muq", "p1", "2", "1", "0", "1", "50.0%"]
    gpu = next(line for line in out.splitlines() if line.startswith("gpu1"))
    assert "never" in gpu
    assert "scan" in out


@pytest.mark.postgres
def test_config_lists_settings(env: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert admin.main(["config"]) == 0
    assert "LAINBOW_MUQ_MAX_SEC" in capsys.readouterr().out


def test_engine_state() -> None:
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    assert (
        admin._engine_state(None, now, lease_sec=60, version=None, server_version="1.0") == "never"
    )
    assert (
        admin._engine_state(
            now - timedelta(seconds=30), now, lease_sec=60, version=None, server_version="1.0"
        )
        == "online"
    )
    assert (
        admin._engine_state(
            now - timedelta(seconds=90), now, lease_sec=60, version=None, server_version="1.0"
        )
        == "offline"
    )
    assert (
        admin._engine_state(
            now - timedelta(seconds=30), now, lease_sec=60, version="0.1.0", server_version="0.1.0"
        )
        == "online"
    )
    assert (
        admin._engine_state(
            now - timedelta(seconds=30), now, lease_sec=60, version="0.0.9", server_version="0.1.0"
        )
        == "outdated"
    )
    assert (
        admin._engine_state(
            now - timedelta(seconds=90), now, lease_sec=60, version="0.0.9", server_version="0.1.0"
        )
        == "outdated"
    )
    assert (
        admin._engine_state(None, now, lease_sec=60, version="0.0.9", server_version="0.1.0")
        == "outdated"
    )


@pytest.mark.postgres
def test_status_engine_online_and_offline_postgres(
    env: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert admin.main(["add-engine", "gpu1"]) == 0
    capsys.readouterr()

    # Engine seen recently -> online, no hint line
    eng = create_engine(env)
    with Session(eng) as s:
        engine = s.scalar(select(RegisteredEngine).where(RegisteredEngine.name == "gpu1"))
        assert engine is not None
        engine.last_seen_at = datetime.now(UTC)
        s.commit()
    eng.dispose()

    assert admin.main(["status"]) == 0
    out = capsys.readouterr().out
    gpu = next(line for line in out.splitlines() if line.startswith("gpu1"))
    assert "online" in gpu
    assert "offline engines keep their registration" not in out

    # Engine seen longer ago than threshold -> offline, hint line appears
    eng = create_engine(env)
    with Session(eng) as s:
        engine = s.scalar(select(RegisteredEngine).where(RegisteredEngine.name == "gpu1"))
        assert engine is not None
        engine.last_seen_at = datetime.now(UTC) - timedelta(seconds=1800)
        s.commit()
    eng.dispose()

    assert admin.main(["status"]) == 0
    out = capsys.readouterr().out
    gpu = next(line for line in out.splitlines() if line.startswith("gpu1"))
    assert "offline" in gpu
    assert (
        "offline engines keep their registration;"
        " remove one with: ./lainbow remove-engine <name>" in out
    )


@pytest.mark.postgres
def test_status_engine_outdated_shows_hint(env: str, capsys: pytest.CaptureFixture[str]) -> None:
    import lainbow

    assert admin.main(["add-engine", "gpu1"]) == 0
    capsys.readouterr()

    # Engine with different stored version -> outdated and hint
    eng = create_engine(env)
    with Session(eng) as s:
        engine = s.scalar(select(RegisteredEngine).where(RegisteredEngine.name == "gpu1"))
        assert engine is not None
        engine.last_seen_at = datetime.now(UTC)
        engine.version = "0.0.1-old"
        s.commit()
    eng.dispose()

    assert admin.main(["status"]) == 0
    out = capsys.readouterr().out
    gpu = next(line for line in out.splitlines() if line.startswith("gpu1"))
    assert "outdated" in gpu
    assert (
        f"outdated engines must be rebuilt from this release: "
        f"on that machine run git checkout v{lainbow.__version__} and join again" in out
    )

    # Matching version -> online and no hint
    eng = create_engine(env)
    with Session(eng) as s:
        engine = s.scalar(select(RegisteredEngine).where(RegisteredEngine.name == "gpu1"))
        assert engine is not None
        engine.version = lainbow.__version__
        s.commit()
    eng.dispose()

    assert admin.main(["status"]) == 0
    out = capsys.readouterr().out
    gpu = next(line for line in out.splitlines() if line.startswith("gpu1"))
    assert "online" in gpu
    assert "outdated engines must be rebuilt from this release" not in out
