"""Leasing of analysis work, derived from the database (no message queue).

All functions take a Session and never commit; the caller owns the transaction.
All timestamps come from the database clock (``now()``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from lainbow.core import models
from lainbow.db.schema import AnalysisDetail
from lainbow.db.vectors import vector_table


@dataclass(frozen=True)
class Task:
    """A (model, profile) pair to analyze."""

    model: str
    profile: str


@dataclass(frozen=True)
class WorkItem:
    """A leased track together with the tasks that were leased for it."""

    track_id: int
    path: str
    duration_sec: float
    tasks: tuple[Task, ...]
    lease_until: datetime


@dataclass(frozen=True)
class Progress:
    """Analysis progress for one (model, profile)."""

    total: int
    done: int
    failed: int
    leased: int

    @property
    def pending(self) -> int:
        """Tracks that are neither done, failed, nor under an unexpired lease."""
        return self.total - self.done - self.failed - self.leased


_CLAIM_SQL = text(
    """
    WITH wanted AS (
        SELECT model, profile
        FROM unnest(CAST(:models AS text[]), CAST(:profiles AS text[])) AS w(model, profile)
    ),
    candidates AS (
        SELECT t.id
        FROM tracks t
        WHERE EXISTS (
            SELECT 1
            FROM wanted w
            LEFT JOIN analyses a
                ON a.track_id = t.id AND a.model = w.model AND a.profile = w.profile
            WHERE a.track_id IS NULL
               OR (a.state = 'leased' AND a.lease_until <= now())
        )
        ORDER BY t.id
        LIMIT :limit
        FOR UPDATE OF t SKIP LOCKED
    ),
    pending AS (
        SELECT c.id AS track_id, w.model, w.profile
        FROM candidates c
        CROSS JOIN wanted w
        LEFT JOIN analyses a
            ON a.track_id = c.id AND a.model = w.model AND a.profile = w.profile
        WHERE a.track_id IS NULL
           OR (a.state = 'leased' AND a.lease_until <= now())
    ),
    upserted AS (
        INSERT INTO analyses AS an
            (track_id, model, profile, state, engine_id, lease_until, attempts, error, updated_at)
        SELECT p.track_id, p.model, p.profile, 'leased', :engine_id,
               now() + make_interval(secs => :lease_sec), 1, NULL, now()
        FROM pending p
        ON CONFLICT (track_id, model, profile) DO UPDATE
        SET state = 'leased',
            engine_id = EXCLUDED.engine_id,
            lease_until = EXCLUDED.lease_until,
            attempts = an.attempts + 1,
            error = NULL,
            updated_at = now()
        WHERE an.state = 'leased' AND an.lease_until <= now()
        RETURNING an.track_id, an.model, an.profile, an.lease_until
    )
    SELECT u.track_id, t.path, t.duration_sec, u.model, u.profile, u.lease_until
    FROM upserted u
    JOIN tracks t ON t.id = u.track_id
    ORDER BY u.track_id, u.model, u.profile
    """
)


def claim(
    session: Session,
    *,
    engine_id: UUID,
    wanted: Sequence[Task],
    limit: int,
    lease_sec: int,
) -> list[WorkItem]:
    """Lease up to ``limit`` tracks that still need at least one wanted task."""
    unique = sorted({(t.model, t.profile) for t in wanted})
    if not unique or limit <= 0:
        return []

    rows = session.execute(
        _CLAIM_SQL,
        {
            "models": [m for m, _ in unique],
            "profiles": [p for _, p in unique],
            "limit": limit,
            "engine_id": engine_id,
            "lease_sec": float(lease_sec),
        },
    ).all()

    items: dict[int, dict[str, Any]] = {}
    for track_id, path, duration_sec, model, profile, lease_until in rows:
        entry = items.setdefault(
            track_id,
            {
                "path": path,
                "duration": float(duration_sec),
                "tasks": [],
                "lease_until": lease_until,
            },
        )
        entry["tasks"].append(Task(model=model, profile=profile))
        entry["lease_until"] = min(entry["lease_until"], lease_until)

    return [
        WorkItem(
            track_id=track_id,
            path=e["path"],
            duration_sec=e["duration"],
            tasks=tuple(e["tasks"]),
            lease_until=e["lease_until"],
        )
        for track_id, e in sorted(items.items())
    ]


def complete(
    session: Session,
    *,
    engine_id: UUID,
    track_id: int,
    model: str,
    profile: str,
    embedding: Sequence[float],
    details: Mapping[str, Any] | None = None,
) -> bool:
    """Store a result if the row is still leased by this engine."""
    expected = models.MODELS[model].dimension
    if len(embedding) != expected:
        raise ValueError(
            f"Embedding for model {model!r} has dimension {len(embedding)}, expected {expected}"
        )

    owned = session.execute(
        text(
            "SELECT 1 FROM analyses WHERE track_id = :track_id AND model = :model "
            "AND profile = :profile AND state = 'leased' AND engine_id = :engine_id "
            "FOR UPDATE"
        ),
        {"track_id": track_id, "model": model, "profile": profile, "engine_id": engine_id},
    ).first()
    if owned is None:
        return False

    vec = vector_table(model)
    vec_stmt = pg_insert(vec).values(track_id=track_id, profile=profile, embedding=list(embedding))
    session.execute(
        vec_stmt.on_conflict_do_update(
            index_elements=[vec.c.track_id, vec.c.profile],
            set_={"embedding": vec_stmt.excluded.embedding},
        )
    )

    if details is not None:
        det_stmt = pg_insert(AnalysisDetail).values(
            track_id=track_id, model=model, profile=profile, details=dict(details)
        )
        session.execute(
            det_stmt.on_conflict_do_update(
                index_elements=[
                    AnalysisDetail.track_id,
                    AnalysisDetail.model,
                    AnalysisDetail.profile,
                ],
                set_={"details": det_stmt.excluded.details},
            )
        )

    session.execute(
        text(
            "UPDATE analyses SET state = 'done', error = NULL, lease_until = NULL, "
            "updated_at = now() "
            "WHERE track_id = :track_id AND model = :model AND profile = :profile"
        ),
        {"track_id": track_id, "model": model, "profile": profile},
    )
    session.execute(
        text(
            "UPDATE engines SET completed = completed + 1, last_seen_at = now() "
            "WHERE id = :engine_id"
        ),
        {"engine_id": engine_id},
    )
    return True


def fail(
    session: Session,
    *,
    engine_id: UUID,
    track_id: int,
    model: str,
    profile: str,
    error: str,
    max_attempts: int = 3,
) -> None:
    """Record a failure: mark failed after ``max_attempts``, otherwise allow an immediate retry."""
    session.execute(
        text(
            "UPDATE analyses SET "
            "state = CASE WHEN attempts >= :max_attempts THEN 'failed' ELSE 'leased' END, "
            "lease_until = CASE WHEN attempts >= :max_attempts THEN NULL ELSE now() END, "
            "error = :error, updated_at = now() "
            "WHERE track_id = :track_id AND model = :model AND profile = :profile "
            "AND state = 'leased' AND engine_id = :engine_id"
        ),
        {
            "max_attempts": max_attempts,
            "error": error,
            "track_id": track_id,
            "model": model,
            "profile": profile,
            "engine_id": engine_id,
        },
    )


def release(session: Session, *, engine_id: UUID) -> int:
    """Expire every lease held by this engine and return the number of leases expired."""
    result = session.execute(
        text(
            "UPDATE analyses SET lease_until = now(), updated_at = now() "
            "WHERE state = 'leased' AND engine_id = :engine_id"
        ),
        {"engine_id": engine_id},
    )
    return int(getattr(result, "rowcount", 0))


def progress(session: Session, *, model: str, profile: str) -> Progress:
    """Return progress counts for (model, profile) using a single query."""
    row = session.execute(
        text(
            "SELECT (SELECT count(*) FROM tracks) AS total, "
            "count(*) FILTER (WHERE state = 'done') AS done, "
            "count(*) FILTER (WHERE state = 'failed') AS failed, "
            "count(*) FILTER (WHERE state = 'leased' AND lease_until > now()) AS leased "
            "FROM analyses WHERE model = :model AND profile = :profile"
        ),
        {"model": model, "profile": profile},
    ).one()
    return Progress(
        total=int(row.total), done=int(row.done), failed=int(row.failed), leased=int(row.leased)
    )


def retry_failed(session: Session, *, model: str, profile: str) -> int:
    """Delete failed rows for (model, profile) so the tracks become candidates again."""
    result = session.execute(
        text(
            "DELETE FROM analyses WHERE state = 'failed' AND model = :model AND profile = :profile"
        ),
        {"model": model, "profile": profile},
    )
    return int(getattr(result, "rowcount", 0))


__all__ = [
    "Progress",
    "Task",
    "WorkItem",
    "claim",
    "complete",
    "fail",
    "progress",
    "release",
    "retry_failed",
]
