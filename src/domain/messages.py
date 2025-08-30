from typing import Any, ClassVar, Literal, Type
from uuid import UUID

from pydantic import BaseModel, Field

from .schemas import EmbeddingModel


class PayloadBase(BaseModel):
    """Base model for task payloads."""

    task_id: UUID

    def __init_subclass__(cls: Type[BaseModel], **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)  # type: ignore[misc]
        if not hasattr(cls, "NAME"):
            raise ValueError("NAME attribute is required for PayloadBase subclasses.")


class ScanTaskPayload(PayloadBase):
    """Payload for a library scan task."""

    NAME: ClassVar[Literal["Library Scan"]] = "Library Scan"

    target_path: str = Field(
        ..., description="The relative directory path to the music root directory to scan."
    )


class AnalyzeSongTaskPayload(PayloadBase):
    """Payload for the analyze song task."""

    NAME: ClassVar[Literal["Analyze Songs"]] = "Analyze Songs"
    models: list[str] = Field(
        default_factory=lambda: [EmbeddingModel.MUQ.value, EmbeddingModel.MUQ_MULAN.value]
    )


class VacuumTaskPayload(PayloadBase):
    """Payload for a library vacuum task."""

    NAME: ClassVar[Literal["Library Vacuum"]] = "Library Vacuum"
