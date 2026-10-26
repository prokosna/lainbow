"""Runnable entry point for the Lainbow server."""

import logging

import uvicorn

from lainbow.core.settings import ServerSettings, start
from lainbow.db.migrate import upgrade_to_head
from lainbow.server.app import create_app

logger = logging.getLogger(__name__)


def main() -> None:
    """Load settings, configure logging, run database migrations, and run the server."""
    settings = start(ServerSettings)
    revision = upgrade_to_head(settings.database_url.get_secret_value())
    logger.info("database schema is up to date (revision %s)", revision)
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level)


if __name__ == "__main__":
    main()
