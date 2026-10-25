"""Public song endpoints used by Sola MPD: similar songs, text search and analysis."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from lainbow.core import models
from lainbow.core.models import Purpose
from lainbow.db.schema import Analysis, AnalysisDetail
from lainbow.db.search import Hit, nearest, track_by_path, vector_of
from lainbow.db.work import Task
from lainbow.server.text_embedding import TextEmbedder, TextEmbeddingUnavailable

router = APIRouter(prefix="/api/v1/songs")

Limit = Annotated[int, Query(ge=1, le=200)]


class SongOut(BaseModel):
    """A song in a search result."""

    file_path: str
    title: str | None
    artist: str | None
    album: str | None
    genre: str | None
    score: float


class SongsOut(BaseModel):
    """Search result."""

    songs: list[SongOut]


def _songs(hits: list[Hit]) -> SongsOut:
    return SongsOut(
        songs=[
            SongOut(
                file_path=h.path,
                title=h.title,
                artist=h.artist,
                album=h.album,
                genre=h.genre,
                score=h.score,
            )
            for h in hits
        ]
    )


def active_task(request: Request, model: str, purpose: Purpose) -> Task:
    """Return the active task of ``model``; raise 422 if it is inactive or lacks ``purpose``."""
    spec = models.MODELS.get(model)
    if spec is None:
        raise HTTPException(status_code=422, detail=f"unknown model {model!r}")
    if purpose not in spec.purposes:
        raise HTTPException(
            status_code=422, detail=f"model {model!r} does not support {purpose.value} search"
        )
    for task in request.app.state.active_tasks:
        if task.model == model:
            return task  # type: ignore[no-any-return]  # app.state is untyped
    raise HTTPException(status_code=422, detail=f"model {model!r} is not active on this server")


@router.get("/search")
def search_songs(
    request: Request,
    q: Annotated[str, Query(min_length=1)],
    model_name: str = "muq_mulan",
    limit: Limit = 10,
) -> SongsOut:
    """Find songs matching a natural-language description."""
    task = active_task(request, model_name, Purpose.TEXT)
    embedder: TextEmbedder = request.app.state.text_embedder
    try:
        vector = embedder.embed(model_name, q)
    except TextEmbeddingUnavailable as e:
        raise HTTPException(
            status_code=503,
            detail="text search needs the engine next to the server, which is not reachable",
        ) from e
    with request.app.state.session_factory() as session:
        return _songs(nearest(session, task.model, task.profile, vector, limit))


@router.get("/{file_path:path}/similar")
def similar_songs(
    request: Request,
    file_path: str,
    model_name: str = "muq",
    limit: Limit = 10,
) -> SongsOut:
    """Find songs that sound like the given song."""
    task = active_task(request, model_name, Purpose.SIMILAR)
    with request.app.state.session_factory() as session:
        track = track_by_path(session, file_path)
        vector = vector_of(session, task.model, task.profile, track.id) if track else None
        if track is None or vector is None:
            raise HTTPException(
                status_code=404, detail=f"{file_path} has not been analyzed with {model_name} yet"
            )
        hits = nearest(session, task.model, task.profile, vector, limit, exclude_track_id=track.id)
        return _songs(hits)


def _analyses(session: Session, track_id: int, tasks: tuple[Task, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task in tasks:
        row = session.execute(
            select(Analysis.state, Analysis.lease_until, Analysis.lease_until > func.now()).where(
                Analysis.track_id == track_id,
                Analysis.model == task.model,
                Analysis.profile == task.profile,
            )
        ).first()
        state = "pending"
        if row is not None and (row.state != "leased" or row[2]):
            state = row.state
        details = session.scalar(
            select(AnalysisDetail.details).where(
                AnalysisDetail.track_id == track_id,
                AnalysisDetail.model == task.model,
                AnalysisDetail.profile == task.profile,
            )
        )
        result[task.model] = {"profile": task.profile, "state": state, "details": details}
    return result


@router.get("/{file_path:path}/analysis")
def song_analysis(request: Request, file_path: str) -> dict[str, Any]:
    """Return the analysis state of a song for every active model."""
    with request.app.state.session_factory() as session:
        track = track_by_path(session, file_path)
        if track is None:
            raise HTTPException(status_code=404, detail=f"{file_path} is not in the library")
        return {
            "file_path": track.path,
            "title": track.title,
            "artist": track.artist,
            "albumartist": track.albumartist,
            "album": track.album,
            "genre": track.genre,
            "duration_sec": track.duration_sec,
            "analyses": _analyses(session, track.id, request.app.state.active_tasks),
        }
