"""Add neighbor_stats and neighbor_profiles tables.

Revision ID: 0003_neighbor_stats
Revises: 0002_engine_version
Create Date: 2026-10-28 22:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003_neighbor_stats"
down_revision: str | None = "0002_engine_version"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "neighbor_stats",
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("profile", sa.Text(), nullable=False),
        sa.Column("track_id", sa.BigInteger(), nullable=False),
        sa.Column("r", sa.REAL(), nullable=False),
        sa.ForeignKeyConstraint(
            ["track_id"],
            ["tracks.id"],
            name=op.f("fk_neighbor_stats_track_id_tracks"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("model", "profile", "track_id", name=op.f("pk_neighbor_stats")),
    )

    op.create_table(
        "neighbor_profiles",
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("profile", sa.Text(), nullable=False),
        sa.Column("k", sa.Integer(), nullable=False),
        sa.Column("vectors", sa.Integer(), nullable=False),
        sa.Column("mean_r", sa.REAL(), nullable=False),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("model", "profile", name=op.f("pk_neighbor_profiles")),
    )

    op.drop_constraint(op.f("ck_runs_kind"), "runs", type_="check")
    op.create_check_constraint(
        op.f("ck_runs_kind"),
        "runs",
        "kind IN ('scan', 'vacuum', 'analyze', 'neighbors')",
    )


def downgrade() -> None:
    op.drop_constraint(op.f("ck_runs_kind"), "runs", type_="check")
    op.create_check_constraint(
        op.f("ck_runs_kind"),
        "runs",
        "kind IN ('scan', 'vacuum', 'analyze')",
    )
    op.drop_table("neighbor_profiles")
    op.drop_table("neighbor_stats")
