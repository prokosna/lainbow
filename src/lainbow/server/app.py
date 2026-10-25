"""FastAPI application factory for the Lainbow server."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy.orm import Session, sessionmaker

import lainbow
from lainbow.core.settings import ServerSettings
from lainbow.db.session import make_engine, make_session_factory
from lainbow.server import engines
from lainbow.server.api import batch, songs, stats
from lainbow.server.api import engine as engine_api
from lainbow.server.profiles import activate_profiles
from lainbow.server.runs import RunExecutor
from lainbow.server.text_embedding import TextEmbedder

logger = logging.getLogger(__name__)

STARTUP_SCAN_DELAY_SEC = 5.0


async def _periodic_scan(app: FastAPI, interval_min: int) -> None:
    await asyncio.sleep(STARTUP_SCAN_DELAY_SEC)
    while True:
        try:
            executor: RunExecutor = app.state.executor
            await asyncio.to_thread(executor.submit, "scan", batch.make_scan_job(app.state))
        except Exception:
            logger.exception("could not start the periodic scan")
        await asyncio.sleep(interval_min * 60)


def create_app(
    settings: ServerSettings,
    *,
    session_factory: sessionmaker[Session] | None = None,
    text_embedder: TextEmbedder | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application for Lainbow server."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = None
        factory = session_factory
        if factory is None:
            engine = make_engine(settings.database_url.get_secret_value())
            factory = make_session_factory(engine)
        app.state.session_factory = factory
        if settings.local_engine_token is not None:
            token = settings.local_engine_token.get_secret_value()
            with factory() as session:
                engines.ensure_local_engine(session, token)
                session.commit()
        app.state.active_tasks = await asyncio.to_thread(activate_profiles, factory, settings)
        embedder = text_embedder or TextEmbedder(str(settings.text_engine_url))
        app.state.text_embedder = embedder
        executor = RunExecutor(factory)
        app.state.executor = executor
        task: asyncio.Task[None] | None = None
        if settings.scan_interval_min > 0:
            task = asyncio.create_task(_periodic_scan(app, settings.scan_interval_min))
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await asyncio.to_thread(executor.shutdown)
            if text_embedder is None:
                embedder.close()
            if engine is not None:
                engine.dispose()

    app = FastAPI(title="Lainbow Server", version=lainbow.__version__, lifespan=lifespan)
    app.state.settings = settings
    app.include_router(batch.router)
    app.include_router(engine_api.router)
    app.include_router(songs.router)
    app.include_router(stats.router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": lainbow.__version__}

    return app
