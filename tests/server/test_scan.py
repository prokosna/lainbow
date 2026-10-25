"""Tests for library scanning against PostgreSQL."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine, func, select, text
from sqlalchemy.orm import Session, sessionmaker

from lainbow.db.migrate import upgrade_to_head
from lainbow.db.schema import Analysis, Track
from lainbow.db.session import make_session_factory
from lainbow.db.vectors import ensure_vector_store
from lainbow.server.library import LibraryUnavailableError, Tags, scan

pytestmark = pytest.mark.postgres


@pytest.fixture
def engine(clean_db: str) -> Iterator[Engine]:
    upgrade_to_head(clean_db)
    eng = create_engine(clean_db)
    yield eng
    eng.dispose()


@pytest.fixture
def factory(engine: Engine) -> sessionmaker[Session]:
    return make_session_factory(engine)


def fake_tags(path: Path) -> Tags:
    if path.name.startswith("bad"):
        return Tags()
    return Tags(title=path.stem, artist="A", duration_sec=10.0)


def make_files(root: Path, names: list[str]) -> None:
    for n in names:
        (root / n).write_bytes(b"x")


def count(factory: sessionmaker[Session]) -> int:
    with factory() as s:
        return s.scalar(select(func.count()).select_from(Track)) or 0


def test_scan_lifecycle(tmp_path: Path, engine: Engine, factory: sessionmaker[Session]) -> None:
    make_files(tmp_path, ["a.flac", "b.flac", "bad.flac"])

    r = scan(factory, tmp_path, read_tags=fake_tags, batch_size=2)
    assert (r.added, r.updated, r.removed, r.unchanged, r.unreadable) == (2, 0, 0, 0, 1)
    assert count(factory) == 2

    r = scan(factory, tmp_path, read_tags=fake_tags)
    assert (r.added, r.updated, r.removed, r.unchanged) == (0, 0, 0, 2)

    with engine.begin() as conn:
        ensure_vector_store(conn, "muq", "p1")
    with factory() as s:
        tid = s.scalars(select(Track.id).where(Track.path == "a.flac")).one()
        s.add(Analysis(track_id=tid, model="muq", profile="p1", state="done"))
        s.commit()
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO vectors_muq (track_id, profile, embedding) VALUES (:t, 'p1', :e)"),
            {"t": tid, "e": "[" + ",".join(["0.1"] * _dim()) + "]"},
        )

    os.utime(tmp_path / "a.flac", (1_000_000_000, 1_000_000_000))
    r = scan(factory, tmp_path, read_tags=fake_tags)
    assert (r.updated, r.unchanged) == (1, 1)
    with factory() as s:
        assert s.scalar(select(func.count()).select_from(Analysis)) == 0
        assert s.execute(text("SELECT count(*) FROM vectors_muq")).scalar_one() == 0

    (tmp_path / "b.flac").unlink()
    r = scan(factory, tmp_path, read_tags=fake_tags)
    assert (r.removed, r.unchanged) == (1, 1)
    assert count(factory) == 1


def _dim() -> int:
    from lainbow.core.models import MODELS

    return MODELS["muq"].dimension


def test_empty_walk_is_refused(tmp_path: Path, factory: sessionmaker[Session]) -> None:
    make_files(tmp_path, ["a.flac"])
    scan(factory, tmp_path, read_tags=fake_tags)
    (tmp_path / "a.flac").unlink()
    with pytest.raises(LibraryUnavailableError):
        scan(factory, tmp_path, read_tags=fake_tags)
    assert count(factory) == 1
