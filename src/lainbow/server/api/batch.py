"""Batch endpoints used by Sola MPD: scan, vacuum and task status."""

from __future__ import annotations

import dataclasses
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from lainbow.db import work
from lainbow.db.schema import Run
from lainbow.server.library import scan
from lainbow.server.runs import RunExecutor, RunKind, RunState, get_run

router = APIRouter(prefix="/api/v1/batch")


class RunOut(BaseModel):
    """Public view of a run."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    kind: RunKind
    state: RunState
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    detail: dict[str, Any]
    error: str | None


def make_scan_job(request_or_app_state: Any) -> Any:
    """Build the job that scans the configured music folder."""
    state = request_or_app_state
    root: Path = state.settings.music_dir
    session_factory = state.session_factory

    def job() -> dict[str, Any]:
        return dataclasses.asdict(scan(session_factory, root))

    return job


def _submit(request: Request, kind: RunKind) -> RunOut:
    state = request.app.state
    executor: RunExecutor = state.executor
    run_id = executor.submit(kind, make_scan_job(state))
    return _load(request, run_id)


def _load(request: Request, run_id: uuid.UUID) -> RunOut:
    with request.app.state.session_factory() as session:
        run: Run | None = get_run(session, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return RunOut.model_validate(run)


@router.post("/scan/run")
def run_scan(request: Request) -> RunOut:
    """Start a library scan."""
    return _submit(request, "scan")


@router.post("/vacuum/run")
def run_vacuum(request: Request) -> RunOut:
    """Start a library scan (in v2 a scan also removes deleted tracks)."""
    return _submit(request, "vacuum")


@router.post("/analyze/run")
def run_analyze(request: Request) -> RunOut:
    """Give failed tracks another chance (analysis itself runs continuously)."""
    state = request.app.state
    session_factory = state.session_factory
    tasks = tuple(state.active_tasks)

    def job() -> dict[str, Any]:
        retried: dict[str, Any] = {}
        with session_factory() as session:
            for task in tasks:
                retried[task.model] = work.retry_failed(
                    session, model=task.model, profile=task.profile
                )
            session.commit()
        return retried

    run_id = state.executor.submit("analyze", job)
    return _load(request, run_id)


@router.get("/tasks/{run_id}")
def get_task(request: Request, run_id: uuid.UUID) -> RunOut:
    """Return the status of a run."""
    return _load(request, run_id)
