"""Analysis method definitions, span calculations, and profile labels."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Final

from pydantic import BaseModel, ConfigDict, Field

# Minimum track duration in seconds before tail dropping applies.
TAIL_MIN_TRACK_SEC = 60.0
# Maximum number of seconds that can be dropped from the end of a track.
TAIL_MAX_SEC = 30.0
# Fraction of total track duration to drop from the end.
TAIL_FRACTION = 0.15


class _Always:
    """Sentinel indicating a label token is always rendered regardless of value."""

    def __repr__(self) -> str:
        return "ALWAYS"


ALWAYS: Final = _Always()


@dataclass(frozen=True)
class LabelToken:
    """Metadata declaring how a field is rendered into a profile label."""

    token: Callable[[Any], str]
    neutral: object = ALWAYS

    def renders(self, value: object) -> bool:
        """Return True if this token should be rendered for the given value."""
        return not (
            self.neutral is not ALWAYS
            and type(value) is type(self.neutral)
            and value == self.neutral
        )


class AnalysisMethod(BaseModel):
    """Configuration for track audio range selection during analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_sec: Annotated[
        int,
        LabelToken(
            token=lambda v: "maxnone" if v == 0 else f"max{v}",
        ),
    ] = Field(
        default=0,
        ge=0,
        description="seconds from the start of a track to analyze (0 = whole track)",
    )

    drop_tail: Annotated[
        bool,
        LabelToken(
            token=lambda v: "droptail" if v else "keeptail",
        ),
    ] = Field(
        default=False,
        description="drop the last min(30 s, 15%) of tracks of 60 s or longer",
    )


def label_tokens(method: AnalysisMethod) -> list[str]:
    """Return profile label tokens for method fields in declaration order."""
    tokens: list[str] = []
    for field_name, field_info in type(method).model_fields.items():
        meta: LabelToken | None = None
        for m in field_info.metadata:
            if isinstance(m, LabelToken):
                meta = m
                break
        if meta is None:
            continue
        val = getattr(method, field_name)
        if meta.renders(val):
            tokens.append(meta.token(val))
    return tokens


def tail_sec(duration: float) -> float:
    """Return the duration in seconds to drop from the track tail."""
    if duration >= TAIL_MIN_TRACK_SEC:
        return min(TAIL_MAX_SEC, TAIL_FRACTION * duration)
    return 0.0


@dataclass(frozen=True)
class Span:
    """A time range [start, end] in seconds."""

    start: float
    end: float

    @property
    def length(self) -> float:
        """Duration of the span in seconds."""
        return self.end - self.start


def analysis_span(duration: float, method: AnalysisMethod) -> Span:
    """Compute the analysis span for a track duration under the given method."""
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")
    end = duration - (tail_sec(duration) if method.drop_tail else 0.0)
    if method.max_sec > 0:
        end = min(end, float(method.max_sec))
    return Span(0.0, end)


def split_clips(span: Span, clip_sec: float | None) -> list[Span]:
    """Split an audio span into consecutive clips of up to clip_sec."""
    if clip_sec is not None and clip_sec <= 0:
        raise ValueError(f"clip_sec must be positive, got {clip_sec}")
    if span.length <= 0:
        return []
    if clip_sec is None:
        return [span]

    clips: list[Span] = []
    i = 0
    while True:
        start = span.start + i * clip_sec
        if start >= span.end:
            break
        end = min(span.start + (i + 1) * clip_sec, span.end)
        clips.append(Span(start, end))
        if end >= span.end:
            break
        i += 1
    return clips


def clip_plan(duration: float, method: AnalysisMethod, clip_sec: float | None) -> list[Span]:
    """Compute consecutive clip spans to analyze for a track."""
    return split_clips(analysis_span(duration, method), clip_sec)


__all__ = [
    "ALWAYS",
    "TAIL_FRACTION",
    "TAIL_MAX_SEC",
    "TAIL_MIN_TRACK_SEC",
    "AnalysisMethod",
    "LabelToken",
    "Span",
    "analysis_span",
    "clip_plan",
    "label_tokens",
    "split_clips",
    "tail_sec",
]
