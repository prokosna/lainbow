"""Initial schema migration: tables and pgvector extension.

Revision ID: 0001_initial
Revises: None
Create Date: 2026-10-25 21:05:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Enable pgvector extension before runtime vector tables are created
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "tracks",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("artist", sa.Text(), nullable=True),
        sa.Column("albumartist", sa.Text(), nullable=True),
        sa.Column("album", sa.Text(), nullable=True),
        sa.Column("genre", sa.Text(), nullable=True),
        sa.Column("duration_sec", sa.REAL(), nullable=False),
        sa.Column("mtime", sa.DateTime(timezone=True), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tracks")),
        sa.UniqueConstraint("path", name=op.f("uq_tracks_path")),
    )

    op.create_table(
        "engines",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("token_hash", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed", sa.BigInteger(), server_default="0", nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_engines")),
        sa.UniqueConstraint("name", name=op.f("uq_engines_name")),
    )

    op.create_table(
        "analyses",
        sa.Column("track_id", sa.BigInteger(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("profile", sa.Text(), nullable=False),
        sa.Column("state", sa.Text(), nullable=False),
        sa.Column("engine_id", sa.Uuid(), nullable=True),
        sa.Column("lease_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "state IN ('leased', 'done', 'failed')",
            name=op.f("ck_analyses_state"),
        ),
        sa.ForeignKeyConstraint(
            ["engine_id"],
            ["engines.id"],
            name=op.f("fk_analyses_engine_id_engines"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["track_id"],
            ["tracks.id"],
            name=op.f("fk_analyses_track_id_tracks"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("track_id", "model", "profile", name=op.f("pk_analyses")),
    )
    op.create_index(
        "ix_analyses_model_profile_state",
        "analyses",
        ["model", "profile", "state"],
        unique=False,
    )

    op.create_table(
        "analysis_details",
        sa.Column("track_id", sa.BigInteger(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("profile", sa.Text(), nullable=False),
        sa.Column("details", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(
            ["track_id"],
            ["tracks.id"],
            name=op.f("fk_analysis_details_track_id_tracks"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("track_id", "model", "profile", name=op.f("pk_analysis_details")),
    )

    op.create_table(
        "runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("state", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "detail",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("error", sa.Text(), nullable=True),
        sa.CheckConstraint("kind IN ('scan', 'vacuum', 'analyze')", name=op.f("ck_runs_kind")),
        sa.CheckConstraint(
            "state IN ('pending', 'running', 'succeeded', 'failed')",
            name=op.f("ck_runs_state"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_runs")),
    )

    op.create_table(
        "active_profiles",
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("profile", sa.Text(), nullable=False),
        sa.Column(
            "activated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("model", name=op.f("pk_active_profiles")),
    )


def downgrade() -> None:
    op.drop_table("active_profiles")
    op.drop_table("runs")
    op.drop_table("analysis_details")
    op.drop_index("ix_analyses_model_profile_state", table_name="analyses")
    op.drop_table("analyses")
    op.drop_table("engines")
    op.drop_table("tracks")
