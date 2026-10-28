"""Neighbor statistics for Cross-domain Similarity Local Scaling (CSLS)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from sqlalchemy import exists, func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from lainbow.db.schema import NeighborProfile, NeighborStat
from lainbow.db.vectors import vector_table

MIN_VECTORS = 1000
GROW_RATIO = 1.25
SHRINK_RATIO = 0.8
CHUNK = 1000

Refresh = Literal["none", "full", "missing"]


def plan_refresh(
    session: Session,
    *,
    model: str,
    profile: str,
    k: int,
) -> Refresh:
    """Determine which kind of neighbor stats refresh is needed for a model profile."""
    if k <= 0:
        return "none"

    tbl = vector_table(model)
    n = session.scalar(select(func.count()).select_from(tbl).where(tbl.c.profile == profile)) or 0

    if n < MIN_VECTORS:
        return "none"

    prof = session.scalar(
        select(NeighborProfile).where(
            NeighborProfile.model == model,
            NeighborProfile.profile == profile,
        )
    )
    if prof is None:
        return "full"

    if prof.k != k:
        return "full"

    if n >= GROW_RATIO * prof.vectors or n <= SHRINK_RATIO * prof.vectors:
        return "full"

    missing_exists = session.scalar(
        select(
            exists().where(
                tbl.c.profile == profile,
                ~exists().where(
                    NeighborStat.model == model,
                    NeighborStat.profile == profile,
                    NeighborStat.track_id == tbl.c.track_id,
                ),
            )
        )
    )
    if missing_exists:
        return "missing"

    return "none"


def refresh(
    session_factory: Callable[[], Session],
    *,
    model: str,
    profile: str,
    k: int,
    kind: Refresh,
) -> dict[str, int]:
    """Compute and update neighbor stats according to the requested refresh kind."""
    if kind == "none":
        return {"computed": 0}

    tbl = vector_table(model)
    tbl_name = tbl.name

    if kind == "full":
        with session_factory() as session:
            target_ids = list(
                session.scalars(
                    select(tbl.c.track_id).where(tbl.c.profile == profile).order_by(tbl.c.track_id)
                )
            )
        n = len(target_ids)
    elif kind == "missing":
        with session_factory() as session:
            target_ids = list(
                session.scalars(
                    select(tbl.c.track_id)
                    .where(
                        tbl.c.profile == profile,
                        ~exists().where(
                            NeighborStat.model == model,
                            NeighborStat.profile == profile,
                            NeighborStat.track_id == tbl.c.track_id,
                        ),
                    )
                    .order_by(tbl.c.track_id)
                )
            )
        n = 0
    else:
        raise ValueError(f"Unknown refresh kind: {kind!r}")

    computed = 0
    ef_search = max(100, 2 * k)

    compute_sql = text(
        f"""
        SELECT v.track_id,
          (SELECT avg(1 - d) FROM (
             SELECT w.embedding <=> v.embedding AS d
             FROM "{tbl_name}" w
             WHERE w.profile = :profile AND w.track_id <> v.track_id
             ORDER BY w.embedding <=> v.embedding
             LIMIT :k
          ) n) AS r
        FROM "{tbl_name}" v
        WHERE v.profile = :profile AND v.track_id = ANY(:ids)
        """
    )

    for i in range(0, len(target_ids), CHUNK):
        chunk_ids = target_ids[i : i + CHUNK]
        with session_factory() as session:
            session.execute(text(f"SET LOCAL hnsw.ef_search = {ef_search}"))
            rows = session.execute(
                compute_sql,
                {"profile": profile, "k": k, "ids": chunk_ids},
            ).all()

            records = [
                {"model": model, "profile": profile, "track_id": row[0], "r": float(row[1])}
                for row in rows
                if row[1] is not None
            ]
            if records:
                stmt = pg_insert(NeighborStat).values(records)
                session.execute(
                    stmt.on_conflict_do_update(
                        index_elements=[
                            NeighborStat.model,
                            NeighborStat.profile,
                            NeighborStat.track_id,
                        ],
                        set_={"r": stmt.excluded.r},
                    )
                )
            session.commit()
            computed += len(records)

    if kind == "full":
        with session_factory() as session:
            mean_r = session.scalar(
                select(func.avg(NeighborStat.r)).where(
                    NeighborStat.model == model,
                    NeighborStat.profile == profile,
                )
            )
            prof_stmt = pg_insert(NeighborProfile).values(
                model=model,
                profile=profile,
                k=k,
                vectors=n,
                mean_r=float(mean_r) if mean_r is not None else 0.0,
                computed_at=func.now(),
            )
            session.execute(
                prof_stmt.on_conflict_do_update(
                    index_elements=[NeighborProfile.model, NeighborProfile.profile],
                    set_={
                        "k": prof_stmt.excluded.k,
                        "vectors": prof_stmt.excluded.vectors,
                        "mean_r": prof_stmt.excluded.mean_r,
                        "computed_at": prof_stmt.excluded.computed_at,
                    },
                )
            )
            session.commit()

    return {"computed": computed}


def profile_stats(
    session: Session,
    *,
    model: str,
    profile: str,
) -> NeighborProfile | None:
    """Return the neighbor profile summary for (model, profile), or None."""
    return session.scalar(
        select(NeighborProfile).where(
            NeighborProfile.model == model,
            NeighborProfile.profile == profile,
        )
    )


__all__ = [
    "CHUNK",
    "GROW_RATIO",
    "MIN_VECTORS",
    "SHRINK_RATIO",
    "Refresh",
    "plan_refresh",
    "profile_stats",
    "refresh",
]
