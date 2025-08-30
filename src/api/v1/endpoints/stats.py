from api.session_manager_wrap import get_db_session
from domain.schemas import (
    EmbeddingModel,
    Song,
    SongEmbedding,
    TaskResult,
    TaskStatus,
)
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel import Session, col, func, select

router = APIRouter()


class DBStats(BaseModel):
    """Dataclass for holding database statistics."""

    total_songs: int
    songs_missing_acoustic_features: int
    songs_missing_clap: int
    songs_missing_mert: int
    songs_missing_muq: int
    songs_missing_muq_mulan: int
    pending_tasks: int
    running_tasks: int


@router.get("/stats", response_model=DBStats, summary="Get database statistics")
def get_db_stats(db: Session = Depends(get_db_session)) -> DBStats:  # noqa: B008
    """Retrieve statistics about the current state of the database."""
    total_songs = db.exec(select(func.count(col(Song.file_path)))).one()

    songs_with_acoustic_features = db.exec(
        select(func.count(col(Song.file_path)))
        .join(SongEmbedding, col(Song.file_path) == col(SongEmbedding.file_path), isouter=True)
        .where(col(SongEmbedding.model_name) == EmbeddingModel.ACOUSTIC_FEATURES)
    ).one()

    songs_with_clap = db.exec(
        select(func.count(col(Song.file_path)))
        .join(SongEmbedding, col(Song.file_path) == col(SongEmbedding.file_path), isouter=True)
        .where(col(SongEmbedding.model_name) == EmbeddingModel.CLAP)
    ).one()

    songs_with_mert = db.exec(
        select(func.count(col(Song.file_path)))
        .join(SongEmbedding, col(Song.file_path) == col(SongEmbedding.file_path), isouter=True)
        .where(col(SongEmbedding.model_name) == EmbeddingModel.MERT)
    ).one()

    songs_with_muq = db.exec(
        select(func.count(col(Song.file_path)))
        .join(SongEmbedding, col(Song.file_path) == col(SongEmbedding.file_path), isouter=True)
        .where(col(SongEmbedding.model_name) == EmbeddingModel.MUQ)
    ).one()

    songs_with_muq_mulan = db.exec(
        select(func.count(col(Song.file_path)))
        .join(SongEmbedding, col(Song.file_path) == col(SongEmbedding.file_path), isouter=True)
        .where(col(SongEmbedding.model_name) == EmbeddingModel.MUQ_MULAN)
    ).one()

    pending_tasks = db.exec(
        select(func.count(col(TaskResult.id))).where(TaskResult.status == TaskStatus.PENDING)
    ).one()

    running_tasks = db.exec(
        select(func.count(col(TaskResult.id))).where(TaskResult.status == TaskStatus.RUNNING)
    ).one()

    return DBStats(
        total_songs=total_songs,
        songs_missing_acoustic_features=total_songs - songs_with_acoustic_features,
        songs_missing_clap=total_songs - songs_with_clap,
        songs_missing_mert=total_songs - songs_with_mert,
        songs_missing_muq=total_songs - songs_with_muq,
        songs_missing_muq_mulan=total_songs - songs_with_muq_mulan,
        pending_tasks=pending_tasks,
        running_tasks=running_tasks,
    )
