"""Analysis method definitions, span calculations, and profile labels."""

from dataclasses import dataclass

from lainbow.core.models import ModelSpec

# Minimum track duration in seconds before tail dropping applies.
TAIL_MIN_TRACK_SEC = 60.0
# Maximum number of seconds that can be dropped from the end of a track.
TAIL_MAX_SEC = 30.0
# Fraction of total track duration to drop from the end.
TAIL_FRACTION = 0.15


@dataclass(frozen=True)
class AnalysisMethod:
    """Configuration for track audio range selection during analysis."""

    max_sec: int | None
    drop_tail: bool

    def __post_init__(self) -> None:
        if self.max_sec is not None and self.max_sec < 1:
            raise ValueError(f"max_sec must be >= 1 or None, got {self.max_sec}")


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
    if method.max_sec is not None:
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


def profile_of(spec: ModelSpec, method: AnalysisMethod) -> str:
    """Generate a profile label from a model specification and analysis method."""
    max_part = "none" if method.max_sec is None else str(method.max_sec)
    tail_part = "droptail" if method.drop_tail else "keeptail"
    return f"r{spec.revision}-max{max_part}-{tail_part}"
