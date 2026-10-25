"""FastAPI application factory for the Lainbow server."""

from fastapi import FastAPI

import lainbow
from lainbow.core.settings import ServerSettings


def create_app(settings: ServerSettings) -> FastAPI:
    """Create and configure the FastAPI application for Lainbow server."""
    app = FastAPI(title="Lainbow Server", version=lainbow.__version__)
    app.state.settings = settings

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": lainbow.__version__}

    return app
