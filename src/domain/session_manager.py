from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import Session, create_engine

from domain import config

engine: Optional[Engine] = None
SessionFactory: Optional[sessionmaker[Session]] = None


def init_db_session() -> None:
    """Initializes the database engine and session factory."""
    global engine, SessionFactory

    if not config.POSTGRES_URL:
        raise ValueError("POSTGRES_URL is not configured.")

    engine = create_engine(config.POSTGRES_URL, pool_pre_ping=True)
    SessionFactory = sessionmaker(
        bind=engine, class_=Session, autoflush=False, expire_on_commit=False
    )


@contextmanager
def db_session_context() -> Generator[Session, None, None]:
    """Provides a synchronous database session."""
    if SessionFactory is None:
        raise RuntimeError(
            "Database session factory is not initialized. Call init_db_session() first."
        )

    with SessionFactory() as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
