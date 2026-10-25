"""Tests for the run executor."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from lainbow.db.migrate import upgrade_to_head
from lainbow.db.schema import Run
from lainbow.db.session import make_session_factory
from lainbow.server.runs import RunExecutor, get_run

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


def wait_done(factory: sessionmaker[Session], run_id: Any) -> Run:
    for _ in range(200):
        with factory() as s:
            run = get_run(s, run_id)
            assert run is not None
            if run.state in ("succeeded", "failed"):
                return run
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_success(factory: sessionmaker[Session]) -> None:
    ex = RunExecutor(factory)
    rid = ex.submit("scan", lambda: {"n": 1})
    run = wait_done(factory, rid)
    ex.shutdown()
    assert run.state == "succeeded"
    assert run.detail == {"n": 1}
    assert run.started_at is not None
    assert run.finished_at is not None


def test_failure(factory: sessionmaker[Session]) -> None:
    def boom() -> dict[str, Any]:
        raise ValueError("boom")

    ex = RunExecutor(factory)
    run = wait_done(factory, ex.submit("scan", boom))
    ex.shutdown()
    assert run.state == "failed"
    assert run.error == "boom"
    assert run.finished_at is not None


def test_duplicate_submit(factory: sessionmaker[Session]) -> None:
    gate = threading.Event()
    ex = RunExecutor(factory)
    first = ex.submit("scan", lambda: (gate.wait(10), {})[1])
    assert ex.submit("scan", lambda: {}) == first
    assert ex.submit("vacuum", lambda: {}) != first
    gate.set()
    wait_done(factory, first)
    second = ex.submit("scan", lambda: {})
    assert second != first
    ex.shutdown()


def test_interrupted_rows(factory: sessionmaker[Session]) -> None:
    import uuid

    rid = uuid.uuid4()
    with factory() as s:
        s.add(Run(id=rid, kind="scan", state="running"))
        s.commit()
    ex = RunExecutor(factory)
    ex.shutdown()
    with factory() as s:
        run = get_run(s, rid)
        assert run is not None
        assert (run.state, run.error) == ("failed", "interrupted")
