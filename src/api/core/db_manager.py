import logging
import time
from dataclasses import asdict, dataclass

import pika
from domain import config, schemas  # noqa: F401
from pika import exceptions as pika_exceptions
from qdrant_client import QdrantClient
from sqlalchemy.exc import OperationalError
from sqlmodel import SQLModel, create_engine

logger = logging.getLogger(__name__)

RETRY_DELAY = 5


@dataclass
class DBConnectionState:
    postgres_ready: bool = False
    milvus_ready: bool = False
    rabbitmq_ready: bool = False

    def is_ready(self) -> bool:
        """Check if all database services are ready."""
        return self.postgres_ready and self.milvus_ready and self.rabbitmq_ready

    def to_dict(self) -> dict[str, bool]:
        """Return the state as a dictionary."""
        return asdict(self)


db_state = DBConnectionState()
engine = create_engine(config.POSTGRES_URL, echo=True)


def create_db_and_tables() -> None:
    """Initializes the database and creates tables if they don't exist."""
    logger.info("Checking and creating database tables...")
    try:
        SQLModel.metadata.create_all(engine)
        logger.info("Database tables are ready.")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise


def check_postgres() -> None:
    """Check if the PostgreSQL database is ready."""
    if db_state.postgres_ready:
        return
    try:
        with engine.connect() as _connection:
            logger.info("PostgreSQL connection successful.")
            db_state.postgres_ready = True
    except OperationalError as e:
        logger.warning(f"PostgreSQL connection failed: {e}")
        db_state.postgres_ready = False


def check_rabbitmq() -> None:
    """Check if the RabbitMQ server is ready."""
    if db_state.rabbitmq_ready:
        return
    try:
        connection = pika.BlockingConnection(pika.URLParameters(config.RABBITMQ_URL))
        if connection.is_open:
            connection.close()
            logger.info("RabbitMQ connection successful.")
            db_state.rabbitmq_ready = True
        else:
            logger.warning("RabbitMQ connection failed: Connection is not open.")
            db_state.rabbitmq_ready = False
    except pika_exceptions.AMQPConnectionError as e:
        logger.warning(f"RabbitMQ connection failed: {e}")
        db_state.rabbitmq_ready = False


def check_milvus() -> None:
    """Check if the Milvus server is ready."""
    if db_state.milvus_ready:
        return
    try:
        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        client.get_collections()
        logger.info("Qdrant connection successful.")
        db_state.milvus_ready = True
    except Exception as e:
        logger.warning(f"Qdrant connection failed: {e}")
        db_state.milvus_ready = False


def wait_for_databases_ready() -> None:
    """Check all database connections and wait until they are all ready."""
    logger.info("Waiting for all database services to be ready...")

    checks = {
        "PostgreSQL": check_postgres,
        "RabbitMQ": check_rabbitmq,
        "Milvus": check_milvus,
    }

    while not db_state.is_ready():
        for check_func in checks.values():
            check_func()

        if not db_state.is_ready():
            logger.info(
                f"Databases not yet ready. Current state: {db_state.to_dict()}. Retrying in {RETRY_DELAY}s..."
            )
            time.sleep(RETRY_DELAY)

    logger.info("All database services are ready.")
