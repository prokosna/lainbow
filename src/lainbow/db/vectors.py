"""Runtime-managed vector tables and HNSW index storage."""

from __future__ import annotations

import hashlib
import re

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, Column, Connection, ForeignKey, MetaData, Table, Text, text

from lainbow.core import models
from lainbow.db.schema import POSTGRES_NAMING_CONVENTION, Track

_PROFILE_PATTERN = re.compile(r"^[a-z0-9-]+$")

# Vector Table objects live in their own MetaData, never in Base.metadata
vector_metadata = MetaData(naming_convention=POSTGRES_NAMING_CONVENTION)

_VECTOR_TABLE_CACHE: dict[tuple[str, int], Table] = {}


def vector_index_name(model: str, profile: str) -> str:
    """Generate the index name for a vector table's profile HNSW index.

    Validates that the profile only contains [a-z0-9-].
    If the full index name exceeds PostgreSQL's 63-byte limit, truncates
    the prefix and appends an 8-hex-digit hash of the full name so that
    it is at most 63 characters and unique.
    """
    if not _PROFILE_PATTERN.match(profile):
        raise ValueError(f"Invalid profile name: {profile!r}. Profile must match '^[a-z0-9-]+$'.")

    safe_profile = profile.replace("-", "_")
    raw_name = f"ix_vectors_{model}_{safe_profile}_hnsw"

    if len(raw_name.encode("utf-8")) <= 63:
        return raw_name

    hash_suffix = hashlib.sha256(raw_name.encode("utf-8")).hexdigest()[:8]
    return raw_name[:55] + hash_suffix


def vector_table(model: str) -> Table:
    """Return the cached or newly built vector table for the given model."""
    spec = models.MODELS[model]  # type: ignore[index]  # ModelId is a StrEnum; lookup by str

    cached = _VECTOR_TABLE_CACHE.get((model, spec.dimension))
    if cached is not None:
        return cached

    table = Table(
        f"vectors_{model}",
        vector_metadata,
        Column(
            "track_id",
            BigInteger,
            ForeignKey(Track.id, ondelete="CASCADE"),
            primary_key=True,
        ),
        Column("profile", Text, primary_key=True),
        Column("embedding", Vector(spec.dimension), nullable=False),
        # extend_existing is needed when a test re-registers the same name with another dimension.
        extend_existing=True,
    )
    _VECTOR_TABLE_CACHE[(model, spec.dimension)] = table
    return table


def ensure_vector_store(conn: Connection, model: str, profile: str) -> None:
    """Ensure that the vector table and the HNSW index for the given profile exist."""
    tbl = vector_table(model)
    tbl.create(conn, checkfirst=True)

    index_name = vector_index_name(model, profile)
    sql = text(
        f"CREATE INDEX IF NOT EXISTS {index_name} "
        f"ON {tbl.name} USING hnsw (embedding vector_cosine_ops) "
        f"WHERE profile = '{profile}'"
    )
    conn.execute(sql)


__all__ = [
    "ensure_vector_store",
    "vector_index_name",
    "vector_metadata",
    "vector_table",
]
