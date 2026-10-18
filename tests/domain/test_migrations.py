"""WI-M2-05 (Critical Module, invariant I2): migrations must never lose an existing install's data.

Needs PostgreSQL. Run with:
    LAINBOW_TEST_POSTGRES_URL=postgresql+psycopg://test:test@127.0.0.1:55432/lainbow_test \
        uv run pytest -m postgres tests/domain/test_migrations.py
Each test creates and drops its own database next to the one in the URL.
"""

import os
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlmodel import SQLModel

import domain.schemas  # noqa: F401  (registers the tables on SQLModel.metadata)
from domain.migrations import BASELINE_REVISION, migrate_database

pytestmark = pytest.mark.postgres

LEGACY_SCHEMA = Path(__file__).parent.parent / "fixtures" / "legacy_schema.sql"
APP_TABLES = {"song", "songfeatures", "songembedding", "taskresult"}


@pytest.fixture
def engine() -> Iterator[Engine]:
    base_url = os.getenv("LAINBOW_TEST_POSTGRES_URL")
    if not base_url:
        pytest.skip("LAINBOW_TEST_POSTGRES_URL is not set")
    admin = create_engine(base_url, isolation_level="AUTOCOMMIT")
    name = f"lainbow_mig_{uuid4().hex[:12]}"
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    db_engine = create_engine(make_url(base_url).set(database=name))
    try:
        yield db_engine
    finally:
        db_engine.dispose()
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def _load_legacy_schema(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(LEGACY_SCHEMA.read_text())
        # Celery's result backend shares the database; migrations must leave its tables alone
        conn.exec_driver_sql("CREATE TABLE celery_taskmeta (id integer PRIMARY KEY, task_id text)")
        conn.exec_driver_sql("INSERT INTO celery_taskmeta VALUES (1, 'keep-me')")


def _schema_diff(engine: Engine) -> list[object]:
    def include_object(
        obj: object, name: str, type_: str, reflected: bool, compare_to: object
    ) -> bool:
        return not (type_ == "table" and name not in APP_TABLES)

    with engine.connect() as conn:
        context = MigrationContext.configure(conn, opts={"include_object": include_object})
        return list(compare_metadata(context, SQLModel.metadata))


def _revision(engine: Engine) -> str | None:
    with engine.connect() as conn:
        return MigrationContext.configure(conn).get_current_revision()


def _head_revision(engine: Engine) -> str | None:
    migrate_database(engine)
    return _revision(engine)


def test_fresh_database_reaches_head_and_matches_models(engine: Engine) -> None:
    migrate_database(engine)

    assert set(inspect(engine).get_table_names()) >= APP_TABLES
    columns = {c["name"] for c in inspect(engine).get_columns("songembedding")}
    assert "vector_collection_name" in columns
    assert "milvus_collection_name" not in columns
    assert _revision(engine) is not None
    assert _revision(engine) != BASELINE_REVISION
    assert _schema_diff(engine) == []


def test_legacy_database_is_stamped_upgraded_and_keeps_data(engine: Engine) -> None:
    _load_legacy_schema(engine)
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO song (file_path, title) VALUES ('a/b.flac', 'Kept')"))
        conn.execute(
            text(
                "INSERT INTO songembedding (file_path, model_name, status, dimension,"
                " milvus_collection_name) VALUES ('a/b.flac', 'MUQ', 'COMPLETED', 1024,"
                " 'muq_audio_embeddings')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO taskresult (id, name, status, progress, details) VALUES"
                " (:id, 'Library Scan', 'SUCCESS', 100, '{\"k\": 1}'::jsonb)"
            ),
            {"id": uuid4()},
        )

    migrate_database(engine)

    with engine.connect() as conn:
        assert conn.execute(text("SELECT title FROM song")).scalar_one() == "Kept"
        assert (
            conn.execute(text("SELECT vector_collection_name FROM songembedding")).scalar_one()
            == "muq_audio_embeddings"
        )
        assert conn.execute(text("SELECT details->>'k' FROM taskresult")).scalar_one() == "1"
        assert conn.execute(text("SELECT task_id FROM celery_taskmeta")).scalar_one() == "keep-me"
    assert _schema_diff(engine) == []


def test_migrating_twice_is_a_no_op(engine: Engine) -> None:
    _load_legacy_schema(engine)
    migrate_database(engine)
    first = _revision(engine)

    migrate_database(engine)

    assert _revision(engine) == first
    assert _schema_diff(engine) == []


def test_baseline_alone_reproduces_the_legacy_schema(engine: Engine) -> None:
    """The baseline revision must describe exactly what create_all produced before Alembic."""
    migrate_database(engine, target=BASELINE_REVISION)
    baseline_columns = {
        table: {c["name"] for c in inspect(engine).get_columns(table)} for table in APP_TABLES
    }
    head = _head_revision(engine)
    assert head != BASELINE_REVISION

    assert "milvus_collection_name" in baseline_columns["songembedding"]
    legacy_sql = LEGACY_SCHEMA.read_text()
    for table, columns in baseline_columns.items():
        for column in columns:
            assert f"    {column} " in legacy_sql, f"{table}.{column} is not in the legacy schema"
