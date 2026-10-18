import importlib
from collections.abc import Generator

import pytest

from domain import config

CONFIG_ENV_VARS = [
    "DATABASE_SERVER_ENDPOINT",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DB",
    "POSTGRES_PORT",
    "POSTGRES_HOST",
    "POSTGRES_CONNECT_PORT",
    "RABBITMQ_DEFAULT_USER",
    "RABBITMQ_DEFAULT_PASS",
    "RABBITMQ_HOST",
    "RABBITMQ_PORT",
    "RABBITMQ_CONNECT_PORT",
    "QDRANT_HOST",
    "QDRANT_PORT_HTTP",
    "QDRANT_CONNECT_PORT",
]


@pytest.fixture(autouse=True)
def restore_config_after_test(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    """Ensures domain.config is reloaded back to clean environment at the end of each test."""
    yield
    monkeypatch.undo()
    importlib.reload(config)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Clears all config-related environment variables so test is deterministic."""
    for var in CONFIG_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_default_config(clean_env: pytest.MonkeyPatch) -> None:
    """When nothing is set, verify today's defaults exactly."""
    cfg = importlib.reload(config)

    # General fallback
    assert cfg.DATABASE_SERVER_ENDPOINT == "host.docker.internal"

    # PostgreSQL
    assert cfg.POSTGRES_USER == "user"
    assert cfg.POSTGRES_PASSWORD == "password"
    assert cfg.POSTGRES_DB == "lainbow_db"
    assert cfg.POSTGRES_PORT == "5433"
    assert cfg.POSTGRES_HOST == "host.docker.internal"
    assert cfg.POSTGRES_CONNECT_PORT == "5433"
    assert (
        cfg.POSTGRES_URL
        == "postgresql+psycopg://user:password@host.docker.internal:5433/lainbow_db"
    )
    assert (
        cfg.CELERY_BACKEND_URL
        == "db+postgresql://user:password@host.docker.internal:5433/lainbow_db"
    )

    # RabbitMQ
    assert cfg.RABBITMQ_USER == "user"
    assert cfg.RABBITMQ_PASS == "password"
    assert cfg.RABBITMQ_HOST == "host.docker.internal"
    assert cfg.RABBITMQ_PORT == "5672"
    assert cfg.RABBITMQ_CONNECT_PORT == "5672"
    assert cfg.RABBITMQ_URL == "amqp://user:password@host.docker.internal:5672/"

    # Qdrant
    assert cfg.QDRANT_HOST == "host.docker.internal"
    assert cfg.QDRANT_PORT == 6333
    assert cfg.QDRANT_CONNECT_PORT == 6333


def test_legacy_config(clean_env: pytest.MonkeyPatch) -> None:
    """When only legacy DATABASE_SERVER_ENDPOINT and POSTGRES_PORT are set."""
    clean_env.setenv("DATABASE_SERVER_ENDPOINT", "dbhost")
    clean_env.setenv("POSTGRES_PORT", "5433")
    cfg = importlib.reload(config)

    # PostgreSQL
    assert cfg.POSTGRES_HOST == "dbhost"
    assert cfg.POSTGRES_PORT == "5433"
    assert cfg.POSTGRES_CONNECT_PORT == "5433"
    assert cfg.POSTGRES_URL == "postgresql+psycopg://user:password@dbhost:5433/lainbow_db"
    assert cfg.CELERY_BACKEND_URL == "db+postgresql://user:password@dbhost:5433/lainbow_db"

    # RabbitMQ
    assert cfg.RABBITMQ_HOST == "dbhost"
    assert cfg.RABBITMQ_PORT == "5672"
    assert cfg.RABBITMQ_CONNECT_PORT == "5672"
    assert cfg.RABBITMQ_URL == "amqp://user:password@dbhost:5672/"

    # Qdrant
    assert cfg.QDRANT_HOST == "dbhost"
    assert cfg.QDRANT_PORT == 6333
    assert cfg.QDRANT_CONNECT_PORT == 6333


def test_compose_config(clean_env: pytest.MonkeyPatch) -> None:
    """When containers reach databases by service names and internal ports."""
    clean_env.setenv("POSTGRES_HOST", "postgres")
    clean_env.setenv("POSTGRES_CONNECT_PORT", "5432")
    clean_env.setenv("POSTGRES_PORT", "5433")
    clean_env.setenv("RABBITMQ_HOST", "rabbitmq")
    clean_env.setenv("QDRANT_HOST", "qdrant")
    cfg = importlib.reload(config)

    # PostgreSQL
    assert cfg.POSTGRES_HOST == "postgres"
    assert cfg.POSTGRES_PORT == "5433"
    assert cfg.POSTGRES_CONNECT_PORT == "5432"
    assert cfg.POSTGRES_URL == "postgresql+psycopg://user:password@postgres:5432/lainbow_db"
    assert cfg.CELERY_BACKEND_URL == "db+postgresql://user:password@postgres:5432/lainbow_db"

    # RabbitMQ
    assert cfg.RABBITMQ_HOST == "rabbitmq"
    assert cfg.RABBITMQ_PORT == "5672"
    assert cfg.RABBITMQ_CONNECT_PORT == "5672"
    assert cfg.RABBITMQ_URL == "amqp://user:password@rabbitmq:5672/"

    # Qdrant
    assert cfg.QDRANT_HOST == "qdrant"
    assert cfg.QDRANT_PORT == 6333
    assert cfg.QDRANT_CONNECT_PORT == 6333


def test_independent_connect_ports(clean_env: pytest.MonkeyPatch) -> None:
    """Verify published ports and connect ports can be configured independently."""
    clean_env.setenv("POSTGRES_PORT", "5433")
    clean_env.setenv("POSTGRES_CONNECT_PORT", "5432")
    clean_env.setenv("RABBITMQ_PORT", "15672")
    clean_env.setenv("RABBITMQ_CONNECT_PORT", "5672")
    clean_env.setenv("QDRANT_PORT_HTTP", "16333")
    clean_env.setenv("QDRANT_CONNECT_PORT", "6333")
    cfg = importlib.reload(config)

    assert cfg.POSTGRES_PORT == "5433"
    assert cfg.POSTGRES_CONNECT_PORT == "5432"
    assert (
        f"postgresql+psycopg://{cfg.POSTGRES_USER}:{cfg.POSTGRES_PASSWORD}@{cfg.POSTGRES_HOST}:5432/{cfg.POSTGRES_DB}"
        == cfg.POSTGRES_URL
    )

    assert cfg.RABBITMQ_PORT == "15672"
    assert cfg.RABBITMQ_CONNECT_PORT == "5672"
    assert (
        f"amqp://{cfg.RABBITMQ_USER}:{cfg.RABBITMQ_PASS}@{cfg.RABBITMQ_HOST}:5672/"
        == cfg.RABBITMQ_URL
    )

    assert cfg.QDRANT_PORT == 16333
    assert cfg.QDRANT_CONNECT_PORT == 6333
