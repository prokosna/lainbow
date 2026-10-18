"""Baseline migration matching legacy pre-Alembic schema.

Revision ID: 0001_baseline
Revises:
Create Date: 2026-10-18

"""

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001_baseline"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

embeddingmodel_enum = postgresql.ENUM(
    "CLAP",
    "MERT",
    "MUQ",
    "MUQ_MULAN",
    "ACOUSTIC_FEATURES",
    name="embeddingmodel",
)

status_enum = postgresql.ENUM(
    "COMPLETED",
    "FAILED",
    "PENDING",
    name="status",
)

taskstatus_enum = postgresql.ENUM(
    "PENDING",
    "RUNNING",
    "SUCCESS",
    "FAILURE",
    "SKIPPED",
    name="taskstatus",
)


def upgrade() -> None:
    """Create legacy schema: enums, tables, constraints, and indexes."""
    embeddingmodel_enum.create(op.get_bind(), checkfirst=True)
    status_enum.create(op.get_bind(), checkfirst=True)
    taskstatus_enum.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "song",
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("file_path", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("artist", sa.String(), nullable=True),
        sa.Column("album", sa.String(), nullable=True),
        sa.Column("genre", sa.String(), nullable=True),
        sa.Column("duration_seconds", sa.Integer(), nullable=True),
        sa.Column("mtime", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("file_path", name="song_pkey"),
    )

    op.create_table(
        "songembedding",
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("file_path", sa.String(), nullable=False),
        sa.Column(
            "model_name",
            postgresql.ENUM(
                "CLAP",
                "MERT",
                "MUQ",
                "MUQ_MULAN",
                "ACOUSTIC_FEATURES",
                name="embeddingmodel",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column(
            "status",
            postgresql.ENUM(
                "COMPLETED",
                "FAILED",
                "PENDING",
                name="status",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("dimension", sa.Integer(), nullable=True),
        sa.Column("milvus_collection_name", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(
            ["file_path"],
            ["song.file_path"],
            name="songembedding_file_path_fkey",
        ),
        sa.PrimaryKeyConstraint("file_path", "model_name", name="songembedding_pkey"),
    )

    songfeatures_columns: list[sa.Column[Any] | sa.Constraint] = [
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("file_path", sa.String(), nullable=False),
        sa.Column(
            "status",
            postgresql.ENUM(
                "COMPLETED",
                "FAILED",
                "PENDING",
                name="status",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("bpm", sa.Float(), nullable=True),
        sa.Column("spectral_centroid_mean", sa.Float(), nullable=True),
        sa.Column("spectral_centroid_std", sa.Float(), nullable=True),
        sa.Column("spectral_bandwidth_mean", sa.Float(), nullable=True),
        sa.Column("spectral_bandwidth_std", sa.Float(), nullable=True),
    ]
    for i in range(13):
        songfeatures_columns.append(sa.Column(f"mfcc_mean_{i}", sa.Float(), nullable=True))
        songfeatures_columns.append(sa.Column(f"mfcc_std_{i}", sa.Float(), nullable=True))
    for i in range(12):
        songfeatures_columns.append(sa.Column(f"chroma_mean_{i}", sa.Float(), nullable=True))
        songfeatures_columns.append(sa.Column(f"chroma_std_{i}", sa.Float(), nullable=True))

    songfeatures_columns.append(sa.PrimaryKeyConstraint("file_path", name="songfeatures_pkey"))
    songfeatures_columns.append(
        sa.ForeignKeyConstraint(
            ["file_path"],
            ["song.file_path"],
            name="songfeatures_file_path_fkey",
        )
    )

    op.create_table("songfeatures", *songfeatures_columns)
    op.create_index("ix_songfeatures_status", "songfeatures", ["status"], unique=False)

    op.create_table(
        "taskresult",
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column(
            "status",
            postgresql.ENUM(
                "PENDING",
                "RUNNING",
                "SUCCESS",
                "FAILURE",
                "SKIPPED",
                name="taskstatus",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("progress", sa.Integer(), nullable=False),
        sa.Column("elapsed_time", sa.Integer(), nullable=True),
        sa.Column("message", sa.String(), nullable=True),
        sa.Column("traceback", sa.String(), nullable=True),
        sa.Column("details", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("id", name="taskresult_pkey"),
    )
    op.create_index("ix_taskresult_created_at", "taskresult", ["created_at"], unique=False)
    op.create_index("ix_taskresult_name", "taskresult", ["name"], unique=False)
    op.create_index("ix_taskresult_updated_at", "taskresult", ["updated_at"], unique=False)


def downgrade() -> None:
    """Drop legacy tables and enum types."""
    op.drop_index("ix_taskresult_updated_at", table_name="taskresult")
    op.drop_index("ix_taskresult_name", table_name="taskresult")
    op.drop_index("ix_taskresult_created_at", table_name="taskresult")
    op.drop_table("taskresult")
    op.drop_index("ix_songfeatures_status", table_name="songfeatures")
    op.drop_table("songfeatures")
    op.drop_table("songembedding")
    op.drop_table("song")

    taskstatus_enum.drop(op.get_bind(), checkfirst=True)
    status_enum.drop(op.get_bind(), checkfirst=True)
    embeddingmodel_enum.drop(op.get_bind(), checkfirst=True)
