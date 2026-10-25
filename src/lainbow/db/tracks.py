"""Track-level helpers shared by the server."""

from __future__ import annotations

from sqlalchemy import delete, text
from sqlalchemy.orm import Session

from lainbow.db.schema import Analysis, AnalysisDetail


def forget_analysis(session: Session, track_id: int) -> None:
    """Delete every analysis result of a track so that it is analyzed again.

    Removes `analyses` and `analysis_details` rows and the track's vectors in every runtime
    vector table that exists (including tables of models that are no longer registered).
    """
    session.execute(delete(Analysis).where(Analysis.track_id == track_id))
    session.execute(delete(AnalysisDetail).where(AnalysisDetail.track_id == track_id))
    names = session.execute(
        text(
            "SELECT table_name FROM information_schema.tables "
            r"WHERE table_schema = current_schema() AND table_name LIKE 'vectors\_%'"
        )
    ).scalars()
    quote = session.get_bind().dialect.identifier_preparer.quote
    for name in list(names):
        session.execute(text(f"DELETE FROM {quote(name)} WHERE track_id = :tid"), {"tid": track_id})


__all__ = ["forget_analysis"]
