import logging
from pathlib import Path
from typing import Final

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import Engine, inspect

logger = logging.getLogger(__name__)

BASELINE_REVISION: Final[str] = "0001_baseline"
APP_TABLES: Final[set[str]] = {"song", "songfeatures", "songembedding", "taskresult"}
MIGRATIONS_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "migrations"


def migrate_database(engine: Engine, target: str = "head") -> None:
    """Migrate database to the specified target revision (default 'head').

    If legacy application tables exist without an alembic_version table,
    the database is stamped at BASELINE_REVISION before upgrading.
    """
    alembic_cfg = Config()
    alembic_cfg.set_main_option("script_location", str(MIGRATIONS_DIR))

    with engine.connect() as connection, connection.begin():
        alembic_cfg.attributes["connection"] = connection

        inspector = inspect(connection)
        tables = set(inspector.get_table_names())

        context = MigrationContext.configure(connection)
        current_rev = context.get_current_revision()

        has_app_tables = bool(APP_TABLES & tables)
        has_alembic_version = "alembic_version" in tables

        if has_app_tables and not has_alembic_version:
            logger.info(
                "Detected legacy tables without alembic_version. Stamping baseline %s",
                BASELINE_REVISION,
            )
            command.stamp(alembic_cfg, BASELINE_REVISION)
            current_rev = BASELINE_REVISION

        command.upgrade(alembic_cfg, target)
        new_rev = MigrationContext.configure(connection).get_current_revision()

        if current_rev == new_rev:
            logger.info("Database is already at head (%s)", current_rev)
        else:
            logger.info("Upgraded database from %s to %s", current_rev, new_rev)
