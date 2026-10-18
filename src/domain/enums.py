import enum

__all__ = [
    "EmbeddingModel",
    "Status",
    "TaskStatus",
    "TextEmbeddingModel",
]


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
