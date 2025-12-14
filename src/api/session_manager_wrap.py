from collections.abc import Generator

from domain.session_manager import db_session_context
from sqlmodel import Session


def get_db_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency to provide a synchronous database session.
    This uses the synchronous context manager from the domain layer.
    """
    with db_session_context() as session:
        yield session
