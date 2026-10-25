"""Tests for engine tokens and registration."""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from lainbow.db.migrate import upgrade_to_head
from lainbow.db.session import make_session_factory
from lainbow.server import engines


def test_parse_token_cases() -> None:
    eid = uuid.uuid4()
    assert engines.parse_token(f"lbe_{eid.hex}_abc-_DEF_x") == (eid, "abc-_DEF_x")
    for bad in ["", "lbe", "lbe_x_y", f"xxx_{eid.hex}_s", f"lbe_{eid.hex}_", f"lbe_{eid.hex}"]:
        with pytest.raises(ValueError):
            engines.parse_token(bad)


@pytest.fixture
def db(clean_db: str) -> Iterator[Engine]:
    upgrade_to_head(clean_db)
    eng = create_engine(clean_db)
    yield eng
    eng.dispose()


@pytest.fixture
def factory(db: Engine) -> sessionmaker[Session]:
    return make_session_factory(db)


@pytest.mark.postgres
def test_token_round_trip(factory: sessionmaker[Session]) -> None:
    with factory() as s:
        token = engines.create_engine(s, "gpu1")
        s.commit()
    with factory() as s:
        eng = engines.authenticate(s, token)
        assert eng is not None
        assert eng.name == "gpu1"
        assert eng.last_seen_at is not None
        s.commit()
        with pytest.raises(engines.EngineExistsError):
            engines.create_engine(s, "gpu1")


@pytest.mark.postgres
def test_bad_tokens(factory: sessionmaker[Session]) -> None:
    with factory() as s:
        token = engines.create_engine(s, "gpu1")
        s.commit()
        assert engines.authenticate(s, "garbage") is None
        assert engines.authenticate(s, token + "x") is None
        eid, _ = engines.parse_token(token)
        assert engines.authenticate(s, f"lbe_{eid.hex}_wrong") is None
        assert engines.authenticate(s, f"lbe_{uuid.uuid4().hex}_wrong") is None


@pytest.mark.postgres
def test_ensure_local_engine_idempotent(factory: sessionmaker[Session]) -> None:
    token = f"lbe_{uuid.uuid4().hex}_secret"
    for _ in range(2):
        with factory() as s:
            engines.ensure_local_engine(s, token)
            s.commit()
    with factory() as s:
        assert [e.name for e in engines.list_engines(s)] == ["local"]
        assert engines.authenticate(s, token) is not None
        with pytest.raises(ValueError):
            engines.ensure_local_engine(s, "bad")


@pytest.mark.postgres
def test_remove_engine_releases_leases(db: Engine, factory: sessionmaker[Session]) -> None:
    with factory() as s:
        token = engines.create_engine(s, "gpu1")
        s.commit()
        eid, _ = engines.parse_token(token)
        s.execute(
            text(
                "INSERT INTO tracks (path, duration_sec, mtime, size_bytes, seen_at) "
                "VALUES ('a.flac', 1, now(), 1, now())"
            )
        )
        s.execute(
            text(
                "INSERT INTO analyses (track_id, model, profile, state, engine_id, lease_until) "
                "SELECT id, 'muq', 'p', 'leased', :e, now() + interval '1 hour' FROM tracks"
            ),
            {"e": eid},
        )
        s.commit()
        assert engines.remove_engine(s, "gpu1") is True
        s.commit()
        assert engines.remove_engine(s, "gpu1") is False
        row = s.execute(text("SELECT lease_until <= now(), engine_id FROM analyses")).one()
        assert row[0] is True
        assert row[1] is None
