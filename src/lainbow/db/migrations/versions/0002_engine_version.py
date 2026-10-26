"""Add version column to engines table.

Revision ID: 0002_engine_version
Revises: 0001_initial
Create Date: 2026-10-26 10:45:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_engine_version"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("engines", sa.Column("version", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("engines", "version")
