"""Vector search over the runtime-managed vector tables."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from lainbow.db.schema import Track
from lainbow.db.vectors import vector_table


@dataclass(frozen=True)
class Hit:
    """A track found by vector search; ``score`` is the cosine similarity."""

    track_id: int
    path: str
    title: str | None
    artist: str | None
    album: str | None
    genre: str | None
    score: float


def vector_of(session: Session, model: str, profile: str, track_id: int) -> list[float] | None:
    """Return the stored vector of a track for (model, profile), or None."""
    tbl = vector_table(model)
    value = session.scalar(
        select(tbl.c.embedding).where(tbl.c.track_id == track_id, tbl.c.profile == profile)
    )
    if value is None:
        return None
    return [float(x) for x in value]


def nearest(
    session: Session,
    model: str,
    profile: str,
    query: Sequence[float],
    limit: int,
    exclude_track_id: int | None = None,
) -> list[Hit]:
    """Return the ``limit`` tracks closest to ``query`` by cosine distance."""
    tbl = vector_table(model)
    session.execute(text(f"SET LOCAL hnsw.ef_search = {min(1000, max(40, limit * 2))}"))
    distance = tbl.c.embedding.cosine_distance(list(query))
    inner = select(tbl.c.track_id, distance.label("distance")).where(tbl.c.profile == profile)
    if exclude_track_id is not None:
        inner = inner.where(tbl.c.track_id != exclude_track_id)
    near = inner.order_by(distance).limit(limit).subquery("near")
    rows = session.execute(
        select(
            Track.id,
            Track.path,
            Track.title,
            Track.artist,
            Track.album,
            Track.genre,
            (1 - near.c.distance).label("score"),
        )
        .join(near, near.c.track_id == Track.id)
        .order_by(near.c.distance, Track.id)
    ).all()
    return [
        Hit(
            track_id=r.id,
            path=r.path,
            title=r.title,
            artist=r.artist,
            album=r.album,
            genre=r.genre,
            score=float(r.score),
        )
        for r in rows
    ]


def track_by_path(session: Session, path: str) -> Track | None:
    """Return the track with the given library-relative path, or None."""
    return session.scalar(select(Track).where(Track.path == path))


__all__ = ["Hit", "nearest", "track_by_path", "vector_of"]
