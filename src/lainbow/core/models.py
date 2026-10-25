"""Model definitions and specifications for audio analysis models.

The `revision` field in `ModelSpec` tracks changes that alter the computed vector
but are not captured by the analysis method configuration (such as max_sec or drop_tail).
This includes:
- Model checkpoint / weights
- Target sample rate
- Clip length (`clip_sec`)
- Tail trimming rule
- Clip extraction, padding, or averaging logic

Any code or model update that affects the resulting vector representation must
bump the `revision` number so that existing vectors are invalidated and recomputed.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from lainbow.core.method import AnalysisMethod, label_tokens


class Purpose(StrEnum):
    """Purpose or search capability provided by a model."""

    SIMILAR = "similar"
    TEXT = "text"


class ModelId(StrEnum):
    """Identifier for an audio analysis model."""

    MUQ = "muq"
    MUQ_MULAN = "muq_mulan"
    CLAP = "clap"
    MERT = "mert"


@dataclass(frozen=True)
class ModelSpec:
    """Specification of an audio analysis model."""

    id: str
    dimension: int
    sample_rate: int
    clip_sec: float | None
    revision: int
    description: str
    short_name: str
    purposes: frozenset[Purpose]
    default_method: AnalysisMethod

    def profile(self, method: AnalysisMethod) -> str:
        """Generate a profile label from this model specification and analysis method."""
        return "-".join([f"r{self.revision}", *label_tokens(method)])


MODELS: Mapping[str, ModelSpec] = MappingProxyType(
    {
        ModelId.MUQ: ModelSpec(
            id=ModelId.MUQ,
            dimension=1024,
            sample_rate=24000,
            clip_sec=180.0,
            revision=1,
            description="MuQ: similar-song search",
            short_name="MuQ",
            purposes=frozenset({Purpose.SIMILAR}),
            default_method=AnalysisMethod(max_sec=180, drop_tail=False),
        ),
        ModelId.MUQ_MULAN: ModelSpec(
            id=ModelId.MUQ_MULAN,
            dimension=512,
            sample_rate=24000,
            clip_sec=10.0,
            revision=1,
            description="MuQ-MuLan: natural-language search",
            short_name="MuQ-MuLan",
            purposes=frozenset({Purpose.SIMILAR, Purpose.TEXT}),
            default_method=AnalysisMethod(max_sec=600, drop_tail=False),
        ),
        ModelId.CLAP: ModelSpec(
            id=ModelId.CLAP,
            dimension=512,
            sample_rate=48000,
            clip_sec=10.0,
            revision=1,
            description="CLAP: natural-language search (optional)",
            short_name="CLAP",
            purposes=frozenset({Purpose.SIMILAR, Purpose.TEXT}),
            default_method=AnalysisMethod(max_sec=600, drop_tail=False),
        ),
        ModelId.MERT: ModelSpec(
            id=ModelId.MERT,
            dimension=1024,
            sample_rate=24000,
            clip_sec=10.0,
            revision=1,
            description="MERT: similar-song search (optional)",
            short_name="MERT",
            purposes=frozenset({Purpose.SIMILAR}),
            default_method=AnalysisMethod(max_sec=600, drop_tail=False),
        ),
    }
)
