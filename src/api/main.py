import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from domain.session_manager import init_db_session
from fastapi import FastAPI

from api.core.db_manager import create_db_and_tables, wait_for_databases_ready
from api.core.text_embedding import get_text_embedding_service
from api.v1.api import api_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown events."""
    # 1. Wait for all database services to be ready.
    # This is a blocking call that runs checks concurrently.
    logger.info("Waiting for databases to be ready...")
    wait_for_databases_ready()
    logger.info("Databases are ready.")

    # 2. Initialize the database session factory.
    logger.info("Initializing database session...")
    init_db_session()
    logger.info("Database session initialized.")

    # 3. Initialize PostgreSQL database and create tables.
    logger.info("Creating database tables...")
    create_db_and_tables()
    logger.info("Database tables created.")

    # 4. Pre-load the text embedding model.
    logger.info("Loading text embedding model...")
    get_text_embedding_service()
    logger.info("Text embedding model loaded.")

    yield


app = FastAPI(lifespan=lifespan)
app.include_router(api_router, prefix="/api/v1")


@app.get("/health", status_code=200)
async def health_check() -> dict[str, str]:
    """Health check endpoint for Docker and other monitoring systems."""
    return {"status": "ok"}
