"""Wire models of the server's ``/engine/v1`` API, mirrored on the engine side."""

from typing import Any

from pydantic import BaseModel, Field

from lainbow.core.method import AnalysisMethod


class PlanTask(BaseModel):
    """An active task together with what an engine needs to run it."""

    model: str
    profile: str
    method: AnalysisMethod
    sample_rate: int
    clip_sec: float | None
    dimension: int


class PlanOut(BaseModel):
    """Response of GET /plan."""

    tasks: list[PlanTask]


class ClaimIn(BaseModel):
    """Body of POST /claim."""

    models: list[str]
    limit: int = Field(ge=1, le=64)


class TaskRef(BaseModel):
    """A (model, profile) pair."""

    model: str
    profile: str


class ClaimItem(BaseModel):
    """A leased track."""

    track_id: int
    path: str
    duration_sec: float
    lease_until: str
    tasks: list[TaskRef]


class ClaimOut(BaseModel):
    """Response of POST /claim."""

    items: list[ClaimItem]


class ResultIn(BaseModel):
    """A computed embedding."""

    track_id: int
    model: str
    profile: str
    embedding: list[float]
    details: dict[str, Any] | None = None


class FailureIn(BaseModel):
    """A failed analysis."""

    track_id: int
    model: str
    profile: str
    error: str


class ResultsIn(BaseModel):
    """Body of POST /results."""

    results: list[ResultIn] = []
    failures: list[FailureIn] = []


class ResultsOut(BaseModel):
    """Response of POST /results."""

    accepted: int
    rejected: int
    failed: int


class ReleaseOut(BaseModel):
    """Response of POST /release."""

    released: int


class TextEmbeddingIn(BaseModel):
    """Body of POST /text-embedding."""

    model: str
    text: str


class TextEmbeddingOut(BaseModel):
    """Response of POST /text-embedding."""

    embedding: list[float]
