"""Library and analysis statistics."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from sqlalchemy import func, select

from lainbow.core import models
from lainbow.db import work
from lainbow.db.schema import Run, Track

router = APIRouter(prefix="/api/v1")


@router.get("/stats")
def get_stats(request: Request) -> dict[str, Any]:
    """Return song counts per model and the state of background runs."""
    stats: dict[str, Any] = {}
    analysis: dict[str, Any] = {}
    done: dict[str, int] = {}
    with request.app.state.session_factory() as session:
        stats["total_songs"] = session.scalar(select(func.count()).select_from(Track)) or 0
        for task in request.app.state.active_tasks:
            p = work.progress(session, model=task.model, profile=task.profile)
            done[task.model] = p.done
            analysis[task.model] = {
                "profile": task.profile,
                "done": p.done,
                "failed": p.failed,
                "leased": p.leased,
                "pending": p.pending,
            }
        for state, key in (("pending", "pending_tasks"), ("running", "running_tasks")):
            stats[key] = (
                session.scalar(select(func.count()).select_from(Run).where(Run.state == state)) or 0
            )
    for model in models.MODELS:
        stats[f"songs_with_{model}"] = done.get(model, 0)
    stats["analysis"] = analysis
    return stats
