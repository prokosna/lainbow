"""Runs: background jobs (scan, vacuum, analyze) executed one at a time and recorded in `runs`."""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any, Literal

from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from lainbow.db.schema import Run

logger = logging.getLogger(__name__)

RunKind = Literal["scan", "vacuum", "analyze", "neighbors"]
RunState = Literal["pending", "running", "succeeded", "failed"]


class RunExecutor:
    """Runs jobs sequentially in one background thread and records them in `runs`."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lainbow-run")
        self._closed = False
        with session_factory() as session:
            session.execute(
                update(Run)
                .where(Run.state.in_(("pending", "running")))
                .values(state="failed", error="interrupted", finished_at=datetime.now(UTC))
            )
            session.commit()

    def submit(self, kind: RunKind, job: Callable[[], Mapping[str, Any]]) -> uuid.UUID:
        """Queue a job; return the id of an active run of the same kind if there is one."""
        with self._lock:
            if self._closed:
                raise RuntimeError("executor is shut down")
            with self._session_factory() as session:
                active = session.execute(
                    select(Run.id).where(Run.kind == kind, Run.state.in_(("pending", "running")))
                ).first()
                if active is not None:
                    return active[0]
                run_id = uuid.uuid4()
                session.add(Run(id=run_id, kind=kind, state="pending"))
                session.commit()
            self._pool.submit(self._execute, run_id, job)
            return run_id

    def _execute(self, run_id: uuid.UUID, job: Callable[[], Mapping[str, Any]]) -> None:
        self._update(run_id, state="running", started_at=datetime.now(UTC))
        try:
            detail = dict(job())
        except Exception as e:
            logger.exception("run %s failed", run_id)
            self._update(run_id, state="failed", error=str(e), finished_at=datetime.now(UTC))
        else:
            self._update(run_id, state="succeeded", detail=detail, finished_at=datetime.now(UTC))

    def _update(self, run_id: uuid.UUID, **values: Any) -> None:
        with self._session_factory() as session:
            session.execute(update(Run).where(Run.id == run_id).values(**values))
            session.commit()

    def shutdown(self) -> None:
        """Stop accepting new runs and wait for the queued ones."""
        with self._lock:
            self._closed = True
        self._pool.shutdown(wait=True)


def get_run(session: Session, run_id: uuid.UUID) -> Run | None:
    """Return a run by id, or None."""
    return session.get(Run, run_id)


__all__ = ["RunExecutor", "RunKind", "RunState", "get_run"]
