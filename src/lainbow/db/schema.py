"""Database schema and tables for Lainbow (PostgreSQL with pgvector)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    MetaData,
    Text,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, REAL
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Stable naming conventions for constraints and indexes
POSTGRES_NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=POSTGRES_NAMING_CONVENTION)


class Base(DeclarativeBase):
    """Base class for ORM mapped models."""

    metadata = metadata


class Track(Base):
    """A track in the music library."""

    __tablename__ = "tracks"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    path: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    artist: Mapped[str | None] = mapped_column(Text, nullable=True)
    albumartist: Mapped[str | None] = mapped_column(Text, nullable=True)
    album: Mapped[str | None] = mapped_column(Text, nullable=True)
    genre: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_sec: Mapped[float] = mapped_column(REAL, nullable=False)
    mtime: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Analysis(Base):
    """State of an audio analysis task for a track, model, and profile."""

    __tablename__ = "analyses"
    __table_args__ = (
        CheckConstraint("state IN ('leased', 'done', 'failed')", name="state"),
        Index("ix_analyses_model_profile_state", "model", "profile", "state"),
    )

    track_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True
    )
    model: Mapped[str] = mapped_column(Text, primary_key=True)
    profile: Mapped[str] = mapped_column(Text, primary_key=True)
    state: Mapped[str] = mapped_column(Text, nullable=False)
    engine_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("engines.id", ondelete="SET NULL"), nullable=True
    )
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class AnalysisDetail(Base):
    """Model-produced human-readable analysis details for a track, model, and profile."""

    __tablename__ = "analysis_details"

    track_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True
    )
    model: Mapped[str] = mapped_column(Text, primary_key=True)
    profile: Mapped[str] = mapped_column(Text, primary_key=True)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)


class RegisteredEngine(Base):
    """Registered analysis engine."""

    __tablename__ = "engines"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    name: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    token_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed: Mapped[int] = mapped_column(
        BigInteger, default=0, server_default="0", nullable=False
    )


class Run(Base):
    """Task run status for scan, vacuum, and analyze jobs."""

    __tablename__ = "runs"
    __table_args__ = (
        CheckConstraint("kind IN ('scan', 'vacuum', 'analyze')", name="kind"),
        CheckConstraint("state IN ('pending', 'running', 'succeeded', 'failed')", name="state"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    state: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    detail: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb"), nullable=False
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class ActiveProfile(Base):
    """Active profile for each model."""

    __tablename__ = "active_profiles"

    model: Mapped[str] = mapped_column(Text, primary_key=True)
    profile: Mapped[str] = mapped_column(Text, nullable=False)
    activated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


__all__ = [
    "POSTGRES_NAMING_CONVENTION",
    "ActiveProfile",
    "Analysis",
    "AnalysisDetail",
    "Base",
    "RegisteredEngine",
    "Run",
    "Track",
    "metadata",
]
