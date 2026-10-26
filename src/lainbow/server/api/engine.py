"""Engine protocol: the only way engines talk to the server (``/engine/v1``)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

import lainbow
from lainbow.core.method import AnalysisMethod
from lainbow.core.models import MODELS
from lainbow.core.settings import ServerSettings
from lainbow.db import work
from lainbow.db.schema import RegisteredEngine, Track
from lainbow.db.work import Task
from lainbow.server import engines

router = APIRouter(prefix="/engine/v1")


def get_session(request: Request) -> Iterator[Session]:
    """Yield a database session for the request."""
    with request.app.state.session_factory() as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


def current_engine(request: Request, session: SessionDep) -> RegisteredEngine:
    """Authenticate the bearer token and return the calling engine."""
    unauthorized = HTTPException(
        status_code=401,
        detail="invalid or missing engine token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    header = request.headers.get("Authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise unauthorized
    engine = engines.authenticate(session, token.strip())
    if engine is None:
        session.rollback()
        raise unauthorized

    version = request.headers.get("X-Lainbow-Version")
    engine.version = version
    session.commit()

    if version != lainbow.__version__:
        reported = version if version is not None else "unknown"
        server_ver = lainbow.__version__
        raise HTTPException(
            status_code=409,
            detail=(
                f"server is {server_ver} but this engine is {reported}; "
                f"on the engine machine run: git fetch --tags && git checkout v{server_ver} "
                "&& ./lainbow join <server url> <token> --build"
            ),
        )

    return engine


EngineDep = Annotated[RegisteredEngine, Depends(current_engine)]


class PlanTask(BaseModel):
    """An active task together with what an engine needs to run it."""

    model: str
    profile: str
    method: AnalysisMethod
    sample_rate: int
    clip_sec: float | None
    dimension: int


class PlanOut(BaseModel):
    """Response of GET /plan."""

    tasks: list[PlanTask]


class ClaimIn(BaseModel):
    """Body of POST /claim."""

    models: list[str]
    limit: int = Field(ge=1, le=64)


class TaskRef(BaseModel):
    """A (model, profile) pair."""

    model: str
    profile: str


class ClaimItem(BaseModel):
    """A leased track."""

    track_id: int
    path: str
    duration_sec: float
    lease_until: str
    tasks: list[TaskRef]


class ClaimOut(BaseModel):
    """Response of POST /claim."""

    items: list[ClaimItem]


class ResultIn(BaseModel):
    """A computed embedding."""

    track_id: int
    model: str
    profile: str
    embedding: list[float]
    details: dict[str, Any] | None = None


class FailureIn(BaseModel):
    """A failed analysis."""

    track_id: int
    model: str
    profile: str
    error: str


class ResultsIn(BaseModel):
    """Body of POST /results."""

    results: list[ResultIn] = []
    failures: list[FailureIn] = []


class ResultsOut(BaseModel):
    """Response of POST /results."""

    accepted: int
    rejected: int
    failed: int


class ReleaseOut(BaseModel):
    """Response of POST /release."""

    released: int


def _active_tasks(request: Request) -> tuple[Task, ...]:
    tasks: tuple[Task, ...] = request.app.state.active_tasks
    return tasks


@router.get("/plan")
def get_plan(request: Request, _engine: EngineDep) -> PlanOut:
    """Return the active tasks with the parameters needed to run them."""
    settings: ServerSettings = request.app.state.settings
    out: list[PlanTask] = []
    for t in _active_tasks(request):
        spec = MODELS[t.model]
        out.append(
            PlanTask(
                model=t.model,
                profile=t.profile,
                method=settings.method_for(t.model),
                sample_rate=spec.sample_rate,
                clip_sec=spec.clip_sec,
                dimension=spec.dimension,
            )
        )
    return PlanOut(tasks=out)


@router.post("/claim")
def claim(request: Request, body: ClaimIn, engine: EngineDep, session: SessionDep) -> ClaimOut:
    """Lease tracks that still need work for the models this engine can run."""
    settings: ServerSettings = request.app.state.settings
    wanted = [t for t in _active_tasks(request) if t.model in body.models]
    items = work.claim(
        session,
        engine_id=engine.id,
        wanted=wanted,
        limit=body.limit,
        lease_sec=settings.lease_sec,
    )
    session.commit()
    return ClaimOut(
        items=[
            ClaimItem(
                track_id=i.track_id,
                path=i.path,
                duration_sec=i.duration_sec,
                lease_until=i.lease_until.isoformat(),
                tasks=[TaskRef(model=t.model, profile=t.profile) for t in i.tasks],
            )
            for i in items
        ]
    )


@router.post("/results")
def post_results(
    request: Request, body: ResultsIn, engine: EngineDep, session: SessionDep
) -> ResultsOut:
    """Store results and failures reported by an engine."""
    active = {(t.model, t.profile) for t in _active_tasks(request)}

    for r in body.results:
        if (r.model, r.profile) not in active:
            continue
        expected = MODELS[r.model].dimension
        if len(r.embedding) != expected:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"embedding for model {r.model!r} has dimension {len(r.embedding)}, "
                    f"expected {expected}"
                ),
            )

    accepted = rejected = failed = 0
    for r in body.results:
        if (r.model, r.profile) not in active:
            rejected += 1
            continue
        ok = work.complete(
            session,
            engine_id=engine.id,
            track_id=r.track_id,
            model=r.model,
            profile=r.profile,
            embedding=r.embedding,
            details=r.details,
        )
        if ok:
            accepted += 1
        else:
            rejected += 1
    for f in body.failures:
        if (f.model, f.profile) not in active:
            continue
        work.fail(
            session,
            engine_id=engine.id,
            track_id=f.track_id,
            model=f.model,
            profile=f.profile,
            error=f.error,
        )
        failed += 1
    session.commit()
    return ResultsOut(accepted=accepted, rejected=rejected, failed=failed)


@router.post("/release")
def release(engine: EngineDep, session: SessionDep) -> ReleaseOut:
    """Release every lease held by the calling engine."""
    n = work.release(session, engine_id=engine.id)
    session.commit()
    return ReleaseOut(released=n)


@router.get("/tracks/{track_id}/audio")
def get_audio(
    track_id: int, request: Request, _engine: EngineDep, session: SessionDep
) -> FileResponse:
    """Stream a track's audio file (supports Range requests)."""
    settings: ServerSettings = request.app.state.settings
    rel = session.scalar(select(Track.path).where(Track.id == track_id))
    if rel is None:
        raise HTTPException(status_code=404, detail="track not found")
    root = settings.music_dir.resolve()
    path = (root / rel).resolve()
    if not path.is_relative_to(root) or not Path(path).is_file():
        raise HTTPException(status_code=404, detail="audio file not found")
    return FileResponse(path)
