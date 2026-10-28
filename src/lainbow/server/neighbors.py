"""Background neighbor statistics refresh job and trigger conditions."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from lainbow.db.neighbors import plan_refresh, refresh
from lainbow.db.work import Task

logger = logging.getLogger(__name__)


def needs_refresh(
    session_factory: sessionmaker[Session] | Callable[[], Session],
    tasks: Sequence[Task],
    k: int,
) -> bool:
    """Return True if any task requires a neighbor stats refresh."""
    if k <= 0:
        return False
    with session_factory() as session:
        for task in tasks:
            if plan_refresh(session, model=task.model, profile=task.profile, k=k) != "none":
                return True
    return False


def make_neighbors_job(
    session_factory: sessionmaker[Session] | Callable[[], Session],
    tasks: Sequence[Task],
    k: int,
) -> Callable[[], dict[str, Any]]:
    """Build the job that computes neighbor stats for active tasks needing refresh."""

    def job() -> dict[str, Any]:
        results: dict[str, Any] = {}
        for task in tasks:
            with session_factory() as session:
                kind = plan_refresh(session, model=task.model, profile=task.profile, k=k)
            if kind == "none":
                continue
            logger.info(
                "computing neighbor stats for %s (%s, kind=%s, k=%d)",
                task.model,
                task.profile,
                kind,
                k,
            )
            t0 = time.monotonic()
            stats = refresh(
                session_factory,
                model=task.model,
                profile=task.profile,
                k=k,
                kind=kind,
            )
            elapsed = time.monotonic() - t0
            computed = stats.get("computed", 0)
            logger.info(
                "computed neighbor stats for %s (%s, kind=%s): %d tracks in %.2f s",
                task.model,
                task.profile,
                kind,
                computed,
                elapsed,
            )
            results[task.model] = {
                "profile": task.profile,
                "kind": kind,
                "computed": computed,
            }
        return results

    return job


__all__ = ["make_neighbors_job", "needs_refresh"]
