import datetime
import enum
from typing import Any, Optional
from uuid import UUID, uuid4

from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, DateTime, Field, Relationship, SQLModel, func


class TaskStatus(str, enum.Enum):
    """Enum for the status of a background task."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    SKIPPED = "SKIPPED"


class Status(str, enum.Enum):
    """Enum for the status of an analysis."""

    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PENDING = "PENDING"


class EmbeddingModel(str, enum.Enum):
    """Enum for the embedding model."""

    CLAP = "clap"
    MERT = "mert"
    MUQ = "muq"
    MUQ_MULAN = "muq_mulan"
    ACOUSTIC_FEATURES = "acoustic_features"


class TextEmbeddingModel(str, enum.Enum):
    """Enum for the available text embedding models."""

    CLAP = "clap"
    MUQ_MULAN = "muq_mulan"


class RecordBase(SQLModel):
    """Base model for records with timestamps."""

    created_at: datetime.datetime = Field(
        default=None,
        sa_type=type(DateTime(timezone=True)),
        sa_column_kwargs={"nullable": False, "server_default": func.now()},
    )
    updated_at: datetime.datetime = Field(
        default=None,
        sa_type=type(DateTime(timezone=True)),
        sa_column_kwargs={
            "nullable": False,
            "server_default": func.now(),
            "onupdate": func.now(),
        },
    )


class RecordBaseIndexed(SQLModel):
    """Base model for records with timestamps."""

    created_at: datetime.datetime = Field(
        default=None,
        sa_type=type(DateTime(timezone=True)),
        sa_column_kwargs={"nullable": False, "server_default": func.now(), "index": True},
    )
    updated_at: datetime.datetime = Field(
        default=None,
        sa_type=type(DateTime(timezone=True)),
        sa_column_kwargs={
            "nullable": False,
            "server_default": func.now(),
            "onupdate": func.now(),
            "index": True,
        },
    )


class Song(RecordBase, table=True):
    """Model for core song metadata."""

    file_path: str = Field(
        primary_key=True, description="Relative path to the music file in the storage."
    )

    title: Optional[str] = Field(default=None)
    artist: Optional[str] = Field(default=None)
    album: Optional[str] = Field(default=None)
    genre: Optional[str] = Field(default=None)
    duration_seconds: Optional[int] = Field(default=None)
    mtime: Optional[datetime.datetime] = Field(
        default=None, description="Last modification time of the file."
    )

    features: Optional["SongFeatures"] = Relationship(
        back_populates="song", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    embeddings: list["SongEmbedding"] = Relationship(
        back_populates="song", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )


class SongFeatures(RecordBase, table=True):
    """Model for detailed audio analysis features for each song."""

    file_path: str = Field(
        foreign_key="song.file_path",
        primary_key=True,
        description="Foreign key referencing the songs table.",
    )
    status: Status = Field(default=Status.PENDING, index=True)

    # Global features
    bpm: Optional[float] = Field(default=None)

    # Spectral features (mean and std)
    spectral_centroid_mean: Optional[float] = Field(default=None)
    spectral_centroid_std: Optional[float] = Field(default=None)
    spectral_bandwidth_mean: Optional[float] = Field(default=None)
    spectral_bandwidth_std: Optional[float] = Field(default=None)

    # MFCC (13 dimensions, mean and std)
    mfcc_mean_0: Optional[float] = Field(default=None)
    mfcc_std_0: Optional[float] = Field(default=None)
    mfcc_mean_1: Optional[float] = Field(default=None)
    mfcc_std_1: Optional[float] = Field(default=None)
    mfcc_mean_2: Optional[float] = Field(default=None)
    mfcc_std_2: Optional[float] = Field(default=None)
    mfcc_mean_3: Optional[float] = Field(default=None)
    mfcc_std_3: Optional[float] = Field(default=None)
    mfcc_mean_4: Optional[float] = Field(default=None)
    mfcc_std_4: Optional[float] = Field(default=None)
    mfcc_mean_5: Optional[float] = Field(default=None)
    mfcc_std_5: Optional[float] = Field(default=None)
    mfcc_mean_6: Optional[float] = Field(default=None)
    mfcc_std_6: Optional[float] = Field(default=None)
    mfcc_mean_7: Optional[float] = Field(default=None)
    mfcc_std_7: Optional[float] = Field(default=None)
    mfcc_mean_8: Optional[float] = Field(default=None)
    mfcc_std_8: Optional[float] = Field(default=None)
    mfcc_mean_9: Optional[float] = Field(default=None)
    mfcc_std_9: Optional[float] = Field(default=None)
    mfcc_mean_10: Optional[float] = Field(default=None)
    mfcc_std_10: Optional[float] = Field(default=None)
    mfcc_mean_11: Optional[float] = Field(default=None)
    mfcc_std_11: Optional[float] = Field(default=None)
    mfcc_mean_12: Optional[float] = Field(default=None)
    mfcc_std_12: Optional[float] = Field(default=None)

    # Chroma (12 dimensions, mean and std)
    chroma_mean_0: Optional[float] = Field(default=None)
    chroma_std_0: Optional[float] = Field(default=None)
    chroma_mean_1: Optional[float] = Field(default=None)
    chroma_std_1: Optional[float] = Field(default=None)
    chroma_mean_2: Optional[float] = Field(default=None)
    chroma_std_2: Optional[float] = Field(default=None)
    chroma_mean_3: Optional[float] = Field(default=None)
    chroma_std_3: Optional[float] = Field(default=None)
    chroma_mean_4: Optional[float] = Field(default=None)
    chroma_std_4: Optional[float] = Field(default=None)
    chroma_mean_5: Optional[float] = Field(default=None)
    chroma_std_5: Optional[float] = Field(default=None)
    chroma_mean_6: Optional[float] = Field(default=None)
    chroma_std_6: Optional[float] = Field(default=None)
    chroma_mean_7: Optional[float] = Field(default=None)
    chroma_std_7: Optional[float] = Field(default=None)
    chroma_mean_8: Optional[float] = Field(default=None)
    chroma_std_8: Optional[float] = Field(default=None)
    chroma_mean_9: Optional[float] = Field(default=None)
    chroma_std_9: Optional[float] = Field(default=None)
    chroma_mean_10: Optional[float] = Field(default=None)
    chroma_std_10: Optional[float] = Field(default=None)
    chroma_mean_11: Optional[float] = Field(default=None)
    chroma_std_11: Optional[float] = Field(default=None)

    song: Song = Relationship(back_populates="features")


class SongEmbedding(RecordBase, table=True):
    """Model to track the status and reference of high-dimensional embeddings stored in Milvus."""

    file_path: str = Field(foreign_key="song.file_path", primary_key=True)
    model_name: EmbeddingModel = Field(
        primary_key=True, description="Name of the model used (e.g., 'MERT', 'CLAP')."
    )
    status: Status = Field(default=Status.PENDING)
    dimension: Optional[int] = Field(
        default=None, description="The dimension of the embedding vector."
    )

    milvus_collection_name: Optional[str] = Field(
        default=None, description="The name of the collection in Milvus where the vector is stored."
    )

    song: Song = Relationship(back_populates="embeddings")


class TaskResult(RecordBaseIndexed, table=True):
    """Model to store the state and history of background tasks."""

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    name: str = Field(index=True, description="A task name.")
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    progress: int = Field(default=0, description="Progress percentage of the task.")
    elapsed_time: Optional[int] = Field(default=None, description="Elapsed time in seconds.")
    message: Optional[str] = Field(default=None)
    traceback: Optional[str] = Field(default=None)
    details: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSONB))
