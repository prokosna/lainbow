"""Tests for leasing analysis work (lainbow.db.work)."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session

from lainbow.core import models
from lainbow.core.method import AnalysisMethod
from lainbow.core.models import MODELS, ModelSpec, Purpose
from lainbow.db import work
from lainbow.db.migrate import upgrade_to_head
from lainbow.db.vectors import ensure_vector_store
from lainbow.db.work import Task

pytestmark = pytest.mark.postgres

MUQ = Task("muq", "p1")
MULAN = Task("muq_mulan", "p1")


@pytest.fixture
def engine(clean_db: str) -> Iterator[Engine]:
    upgrade_to_head(clean_db)
    eng = create_engine(clean_db)
    with eng.begin() as conn:
        for model in ("muq", "muq_mulan"):
            ensure_vector_store(conn, model, "p1")
            ensure_vector_store(conn, model, "p2")
    yield eng
    eng.dispose()


def add_tracks(engine: Engine, n: int) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO tracks (path, duration_sec, mtime, size_bytes, seen_at) "
                "SELECT '/t/' || g, 100, now(), 1, now() FROM generate_series(1, :n) g"
            ),
            {"n": n},
        )


def add_engine(engine: Engine, name: str = "e") -> uuid.UUID:
    eid = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO engines (id, name, token_hash) VALUES (:i, :n, 'x')"),
            {"i": eid, "n": f"{name}-{eid}"},
        )
    return eid


def do_claim(
    engine: Engine, eid: uuid.UUID, wanted: list[Task], limit: int = 100, lease_sec: int = 600
) -> list[work.WorkItem]:
    with Session(engine) as s:
        items = work.claim(s, engine_id=eid, wanted=wanted, limit=limit, lease_sec=lease_sec)
        s.commit()
    return items


def do_complete(engine: Engine, eid: uuid.UUID, track_id: int, task: Task, **kw: object) -> bool:
    dim = models.MODELS[task.model].dimension
    with Session(engine) as s:
        ok = work.complete(
            s,
            engine_id=eid,
            track_id=track_id,
            model=task.model,
            profile=task.profile,
            embedding=[0.5] * dim,
            **kw,  # type: ignore[arg-type]  # test helper forwards details
        )
        s.commit()
    return ok


def scalar(engine: Engine, sql: str, **params: object) -> object:
    with engine.connect() as conn:
        return conn.execute(text(sql), params).scalar()


def expire(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("UPDATE analyses SET lease_until = now() - interval '1 second'"))


def test_concurrent_claims_are_disjoint(engine: Engine) -> None:
    add_tracks(engine, 50)
    e1, e2 = add_engine(engine), add_engine(engine)
    barrier = threading.Barrier(2)
    results: dict[int, list[work.WorkItem]] = {}

    def run(idx: int, eid: uuid.UUID) -> None:
        with engine.connect() as conn, Session(conn) as s:
            barrier.wait()
            results[idx] = work.claim(s, engine_id=eid, wanted=[MUQ], limit=30, lease_sec=60)
            s.commit()

    threads = [threading.Thread(target=run, args=(i, e)) for i, e in enumerate((e1, e2))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    a = {i.track_id for i in results[0]}
    b = {i.track_id for i in results[1]}
    assert not a & b
    assert len(a) + len(b) == 50


def test_lease_expiry_reclaim(engine: Engine) -> None:
    add_tracks(engine, 1)
    e1, e2 = add_engine(engine), add_engine(engine)
    assert len(do_claim(engine, e1, [MUQ], lease_sec=1)) == 1
    assert do_claim(engine, e2, [MUQ]) == []
    expire(engine)
    items = do_claim(engine, e2, [MUQ])
    assert len(items) == 1
    assert scalar(engine, "SELECT attempts FROM analyses") == 2
    assert scalar(engine, "SELECT engine_id FROM analyses") == e2


def test_losing_lease_rejects_complete(engine: Engine) -> None:
    add_tracks(engine, 1)
    e1, e2 = add_engine(engine), add_engine(engine)
    tid = do_claim(engine, e1, [MUQ])[0].track_id
    expire(engine)
    do_claim(engine, e2, [MUQ])
    assert do_complete(engine, e1, tid, MUQ) is False
    assert scalar(engine, "SELECT count(*) FROM vectors_muq") == 0
    assert do_complete(engine, e2, tid, MUQ) is True


def test_expired_but_not_released_complete_accepted(engine: Engine) -> None:
    add_tracks(engine, 1)
    e1 = add_engine(engine)
    tid = do_claim(engine, e1, [MUQ])[0].track_id
    expire(engine)
    assert do_complete(engine, e1, tid, MUQ) is True
    assert scalar(engine, "SELECT completed FROM engines WHERE id = :i", i=e1) == 1


def test_done_rows_not_claimed(engine: Engine) -> None:
    add_tracks(engine, 2)
    e1 = add_engine(engine)
    for item in do_claim(engine, e1, [MUQ]):
        assert do_complete(engine, e1, item.track_id, MUQ)
    assert do_claim(engine, e1, [MUQ]) == []
    expire(engine)
    assert do_claim(engine, e1, [MUQ]) == []


def test_profile_change_makes_candidates(engine: Engine) -> None:
    add_tracks(engine, 3)
    e1 = add_engine(engine)
    for item in do_claim(engine, e1, [MUQ]):
        do_complete(engine, e1, item.track_id, MUQ)
    assert len(do_claim(engine, e1, [Task("muq", "p2")])) == 3


def test_track_is_the_lease_unit(engine: Engine) -> None:
    add_tracks(engine, 3)
    e1 = add_engine(engine)
    items = do_claim(engine, e1, [MUQ, MULAN])
    assert len(items) == 3
    assert all(set(i.tasks) == {MUQ, MULAN} for i in items)
    tid = items[0].track_id
    assert do_complete(engine, e1, tid, MUQ)
    expire(engine)
    again = {i.track_id: i for i in do_claim(engine, e1, [MUQ, MULAN])}
    assert again[tid].tasks == (MULAN,)
    assert len(again) == 3


def test_concurrent_different_tasks_disjoint(engine: Engine) -> None:
    add_tracks(engine, 40)
    e1, e2 = add_engine(engine), add_engine(engine)
    barrier = threading.Barrier(2)
    results: dict[int, list[work.WorkItem]] = {}

    def run(idx: int, eid: uuid.UUID, wanted: list[Task]) -> None:
        with engine.connect() as conn, Session(conn) as s:
            barrier.wait()
            results[idx] = work.claim(s, engine_id=eid, wanted=wanted, limit=25, lease_sec=60)
            s.commit()

    threads = [
        threading.Thread(target=run, args=(0, e1, [MUQ])),
        threading.Thread(target=run, args=(1, e2, [MULAN])),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    a = {i.track_id for i in results[0]}
    b = {i.track_id for i in results[1]}
    assert not a & b
    assert len(a) + len(b) == 40


def test_fail_and_retry(engine: Engine) -> None:
    add_tracks(engine, 1)
    e1 = add_engine(engine)
    tid = do_claim(engine, e1, [MUQ])[0].track_id

    def do_fail() -> None:
        with Session(engine) as s:
            work.fail(
                s, engine_id=e1, track_id=tid, model="muq", profile="p1", error="boom",
                max_attempts=2,
            )  # fmt: skip
            s.commit()

    do_fail()
    assert scalar(engine, "SELECT state FROM analyses") == "leased"
    assert len(do_claim(engine, e1, [MUQ])) == 1  # retried at once, attempts=2
    assert scalar(engine, "SELECT attempts FROM analyses") == 2
    do_fail()
    assert scalar(engine, "SELECT state FROM analyses") == "failed"
    assert scalar(engine, "SELECT error FROM analyses") == "boom"
    assert do_claim(engine, e1, [MUQ]) == []
    with Session(engine) as s:
        assert work.retry_failed(s, model="muq", profile="p1") == 1
        s.commit()
    assert len(do_claim(engine, e1, [MUQ])) == 1


def test_fail_ignored_for_other_engine(engine: Engine) -> None:
    add_tracks(engine, 1)
    e1, e2 = add_engine(engine), add_engine(engine)
    tid = do_claim(engine, e1, [MUQ])[0].track_id
    with Session(engine) as s:
        work.fail(s, engine_id=e2, track_id=tid, model="muq", profile="p1", error="x")
        s.commit()
    assert scalar(engine, "SELECT error FROM analyses") is None


def test_wrong_dimension(engine: Engine) -> None:
    add_tracks(engine, 1)
    e1 = add_engine(engine)
    tid = do_claim(engine, e1, [MUQ])[0].track_id
    with Session(engine) as s, pytest.raises(ValueError):
        work.complete(s, engine_id=e1, track_id=tid, model="muq", profile="p1", embedding=[0.0] * 3)


def test_details_stored_and_replaced(engine: Engine) -> None:
    add_tracks(engine, 1)
    e1 = add_engine(engine)
    tid = do_claim(engine, e1, [MUQ])[0].track_id
    assert do_complete(engine, e1, tid, MUQ, details={"a": 1})
    assert scalar(engine, "SELECT details->>'a' FROM analysis_details") == "1"
    expire(engine)
    with engine.begin() as conn:
        conn.execute(text("UPDATE analyses SET state='leased', lease_until=now()"))
    do_claim(engine, e1, [MUQ])
    assert do_complete(engine, e1, tid, MUQ, details={"b": 2})
    assert scalar(engine, "SELECT count(*) FROM analysis_details") == 1
    assert scalar(engine, "SELECT details->>'b' FROM analysis_details") == "2"
    assert scalar(engine, "SELECT details->>'a' FROM analysis_details") is None


def test_fake_model_extensibility(engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = ModelSpec(
        id="fake_work",
        dimension=8,
        sample_rate=16000,
        clip_sec=10.0,
        revision=1,
        description="fake",
        short_name="Fake",
        purposes=frozenset({Purpose.SIMILAR}),
        default_method=AnalysisMethod(max_sec=180, drop_tail=False),
        memory_gb=1.0,
    )
    fake: dict[str, ModelSpec] = {str(k): v for k, v in MODELS.items()}
    fake["fake_work"] = spec
    monkeypatch.setattr("lainbow.core.models.MODELS", fake)
    with engine.begin() as conn:
        ensure_vector_store(conn, "fake_work", "p1")
    add_tracks(engine, 1)
    e1 = add_engine(engine)
    task = Task("fake_work", "p1")
    tid = do_claim(engine, e1, [task])[0].track_id
    assert do_complete(engine, e1, tid, task, details={"k": "v"})
    assert scalar(engine, "SELECT count(*) FROM vectors_fake_work") == 1


def test_progress_sequence(engine: Engine) -> None:
    add_tracks(engine, 5)
    e1 = add_engine(engine)
    with Session(engine) as s:
        assert work.progress(s, model="muq", profile="p1") == work.Progress(5, 0, 0, 0)
    items = do_claim(engine, e1, [MUQ], limit=4)
    do_complete(engine, e1, items[0].track_id, MUQ)
    with Session(engine) as s:
        work.fail(
            s, engine_id=e1, track_id=items[1].track_id, model="muq", profile="p1",
            error="e", max_attempts=1,
        )  # fmt: skip
        s.commit()
    with Session(engine) as s:
        p = work.progress(s, model="muq", profile="p1")
    assert (p.total, p.done, p.failed, p.leased, p.pending) == (5, 1, 1, 2, 1)
    expire(engine)
    with Session(engine) as s:
        p = work.progress(s, model="muq", profile="p1")
    assert (p.leased, p.pending) == (0, 3)


def test_release(engine: Engine) -> None:
    add_tracks(engine, 3)
    e1, e2 = add_engine(engine), add_engine(engine)
    assert len(do_claim(engine, e1, [MUQ, MULAN])) == 3
    assert do_claim(engine, e2, [MUQ, MULAN]) == []
    with Session(engine) as s:
        assert work.release(s, engine_id=e1) == 6
        s.commit()
    assert len(do_claim(engine, e2, [MUQ, MULAN])) == 3
