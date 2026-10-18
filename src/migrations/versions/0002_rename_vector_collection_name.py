"""Rename songembedding.milvus_collection_name to vector_collection_name.

Revision ID: 0002_rename_vector_collection_name
Revises: 0001_baseline
Create Date: 2026-10-18

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_rename_vector_collection"
down_revision: str | Sequence[str] | None = "0001_baseline"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Rename milvus_collection_name to vector_collection_name."""
    op.alter_column(
        "songembedding",
        "milvus_collection_name",
        new_column_name="vector_collection_name",
    )


def downgrade() -> None:
    """Rename vector_collection_name back to milvus_collection_name."""
    op.alter_column(
        "songembedding",
        "vector_collection_name",
        new_column_name="milvus_collection_name",
    )
