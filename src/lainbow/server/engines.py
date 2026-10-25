"""Engine registration and bearer-token authentication.

All functions take a Session and never commit; the caller owns the transaction.
Token format: ``lbe_<engine uuid hex>_<secret>``. Only ``sha256(secret)`` is stored.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from lainbow.db import work
from lainbow.db.schema import RegisteredEngine

TOKEN_PREFIX = "lbe"
LOCAL_ENGINE_NAME = "local"


class EngineExistsError(Exception):
    """Raised when an engine with the same name is already registered."""


def _hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def parse_token(token: str) -> tuple[UUID, str]:
    """Split a token into the engine id and the secret; raise ValueError if malformed."""
    parts = token.split("_", 2)
    if len(parts) != 3 or parts[0] != TOKEN_PREFIX or not parts[2]:
        raise ValueError("malformed engine token")
    try:
        engine_id = UUID(hex=parts[1])
    except ValueError:
        raise ValueError("malformed engine token") from None
    if len(parts[1]) != 32:
        raise ValueError("malformed engine token")
    return engine_id, parts[2]


def create_engine(session: Session, name: str) -> str:
    """Register a new engine and return its token."""
    exists = session.scalar(select(RegisteredEngine.id).where(RegisteredEngine.name == name))
    if exists is not None:
        raise EngineExistsError(f"engine {name!r} already exists")
    engine_id = uuid.uuid4()
    secret = secrets.token_urlsafe(32)
    session.add(RegisteredEngine(id=engine_id, name=name, token_hash=_hash_secret(secret)))
    session.flush()
    return f"{TOKEN_PREFIX}_{engine_id.hex}_{secret}"


def authenticate(session: Session, token: str) -> RegisteredEngine | None:
    """Return the engine for a valid token (and touch last_seen_at), else None."""
    try:
        engine_id, secret = parse_token(token)
    except ValueError:
        return None
    engine = session.get(RegisteredEngine, engine_id)
    if engine is None:
        return None
    if not hmac.compare_digest(engine.token_hash, _hash_secret(secret)):
        return None
    engine.last_seen_at = func.now()  # SQL expression: server clock
    session.flush()
    session.refresh(engine)
    return engine


def ensure_local_engine(session: Session, token: str) -> None:
    """Upsert the engine named ``local`` using the id and secret hash of the given token."""
    engine_id, secret = parse_token(token)
    stale = session.scalar(
        select(RegisteredEngine).where(
            RegisteredEngine.name == LOCAL_ENGINE_NAME, RegisteredEngine.id != engine_id
        )
    )
    if stale is not None:
        # The token was regenerated: drop the old registration so the name is free.
        work.release(session, engine_id=stale.id)
        session.delete(stale)
        session.flush()
    stmt = pg_insert(RegisteredEngine).values(
        id=engine_id, name=LOCAL_ENGINE_NAME, token_hash=_hash_secret(secret)
    )
    session.execute(
        stmt.on_conflict_do_update(
            index_elements=[RegisteredEngine.id],
            set_={"name": stmt.excluded.name, "token_hash": stmt.excluded.token_hash},
        )
    )


def list_engines(session: Session) -> list[RegisteredEngine]:
    """Return all engines ordered by name."""
    return list(session.scalars(select(RegisteredEngine).order_by(RegisteredEngine.name)))


def remove_engine(session: Session, name: str) -> bool:
    """Release the engine's leases and delete it; return False if it does not exist."""
    engine = session.scalar(select(RegisteredEngine).where(RegisteredEngine.name == name))
    if engine is None:
        return False
    work.release(session, engine_id=engine.id)
    session.delete(engine)
    session.flush()
    return True


__all__ = [
    "LOCAL_ENGINE_NAME",
    "EngineExistsError",
    "authenticate",
    "create_engine",
    "ensure_local_engine",
    "list_engines",
    "parse_token",
    "remove_engine",
]
