"""Activation of the (model, profile) pairs the server wants analyzed."""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker

from lainbow.core.models import MODELS
from lainbow.core.settings import ServerSettings
from lainbow.db.schema import ActiveProfile
from lainbow.db.vectors import ensure_vector_store
from lainbow.db.work import Task

logger = logging.getLogger(__name__)


def activate_profiles(
    session_factory: sessionmaker[Session], settings: ServerSettings
) -> tuple[Task, ...]:
    """Ensure vector stores, upsert ``active_profiles`` and return the active tasks."""
    tasks: list[Task] = []
    with session_factory() as session:
        for model in settings.models:
            profile = MODELS[model].profile(settings.method_for(model))
            ensure_vector_store(session.connection(), model, profile)
            previous = session.scalar(
                select(ActiveProfile.profile).where(ActiveProfile.model == model)
            )
            stmt = pg_insert(ActiveProfile).values(model=model, profile=profile)
            session.execute(
                stmt.on_conflict_do_update(
                    index_elements=[ActiveProfile.model],
                    set_={"profile": stmt.excluded.profile},
                    where=ActiveProfile.profile != stmt.excluded.profile,
                )
            )
            if previous is not None and previous != profile:
                logger.warning(
                    "%s: analysis method changed (%s -> %s); every track will be analyzed "
                    "again for this model; vectors of the previous method are kept",
                    model,
                    previous,
                    profile,
                )
            tasks.append(Task(model=model, profile=profile))
        session.commit()
    return tuple(tasks)


__all__ = ["activate_profiles"]
