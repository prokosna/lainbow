"""Vector search over the runtime-managed vector tables."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Select, func, literal_column, select, text
from sqlalchemy.orm import Session

from lainbow.db.neighbors import profile_stats
from lainbow.db.schema import NeighborProfile, NeighborStat, Track
from lainbow.db.vectors import validate_profile, vector_table

CANDIDATES = 200


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


def _nearest_statement(
    model: str,
    profile: str,
    query: Sequence[float],
    limit: int,
    exclude_track_id: int | None = None,
    prof: NeighborProfile | None = None,
    csls_k: int = 0,
) -> Select[Any]:
    """Build the SELECT query for nearest neighbor search."""
    valid_profile = validate_profile(profile)
    tbl = vector_table(model)
    use_csls = prof is not None and prof.k == csls_k

    fetch_limit = int(max(CANDIDATES, limit) if use_csls else limit)

    distance = tbl.c.embedding.cosine_distance(list(query))
    inner = select(tbl.c.track_id, distance.label("distance")).where(
        tbl.c.profile == literal_column(f"'{valid_profile}'")
    )
    if exclude_track_id is not None:
        inner = inner.where(tbl.c.track_id != exclude_track_id)
    near = inner.order_by(distance).limit(literal_column(str(fetch_limit))).subquery("near")

    stmt = select(
        Track.id,
        Track.path,
        Track.title,
        Track.artist,
        Track.album,
        Track.genre,
        (1 - near.c.distance).label("score"),
    ).join(near, near.c.track_id == Track.id)

    if use_csls:
        assert prof is not None
        stmt = (
            stmt.outerjoin(
                NeighborStat,
                (NeighborStat.track_id == near.c.track_id)
                & (NeighborStat.model == model)
                & (NeighborStat.profile == valid_profile),
            )
            .order_by(
                (2 * (1 - near.c.distance) - func.coalesce(NeighborStat.r, prof.mean_r)).desc(),
                near.c.distance,
                Track.id,
            )
            .limit(literal_column(str(int(limit))))
        )
    else:
        stmt = stmt.order_by(near.c.distance, Track.id)

    return stmt


def nearest(
    session: Session,
    model: str,
    profile: str,
    query: Sequence[float],
    limit: int,
    exclude_track_id: int | None = None,
    csls_k: int = 0,
) -> list[Hit]:
    """Return the ``limit`` tracks closest to ``query`` by cosine distance or CSLS."""
    prof = profile_stats(session, model=model, profile=profile) if csls_k > 0 else None
    use_csls = prof is not None and prof.k == csls_k

    fetch_limit = max(CANDIDATES, limit) if use_csls else limit
    ef_search = min(1000, max(40, fetch_limit * 2))
    session.execute(text(f"SET LOCAL hnsw.ef_search = {ef_search}"))

    stmt = _nearest_statement(
        model=model,
        profile=profile,
        query=query,
        limit=limit,
        exclude_track_id=exclude_track_id,
        prof=prof,
        csls_k=csls_k,
    )
    rows = session.execute(stmt).all()
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


__all__ = ["CANDIDATES", "Hit", "nearest", "track_by_path", "vector_of"]
