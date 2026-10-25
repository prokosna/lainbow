"""Database migration runner using programmatic Alembic configuration."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory


def get_alembic_config(database_url: str) -> Config:
    """Build Alembic Config referencing the package migrations directory."""
    migrations_dir = Path(__file__).parent / "migrations"
    config = Config()
    config.set_main_option("script_location", str(migrations_dir))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def upgrade_to_head(database_url: str) -> str:
    """Upgrade database to head revision and return head revision ID."""
    config = get_alembic_config(database_url)
    command.upgrade(config, "head")
    script = ScriptDirectory.from_config(config)
    head = script.get_current_head()
    if head is None:
        raise RuntimeError("No head revision found in migrations directory")
    return head


__all__ = ["get_alembic_config", "upgrade_to_head"]
