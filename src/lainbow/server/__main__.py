"""Runnable entry point for the Lainbow server."""

import uvicorn

from lainbow.core.settings import ServerSettings, start
from lainbow.server.app import create_app


def main() -> None:
    """Load settings, configure logging, and run the Lainbow server."""
    settings = start(ServerSettings)
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level)


if __name__ == "__main__":
    main()
