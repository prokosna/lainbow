from dataclasses import dataclass
from typing import Literal

from domain.enums import EmbeddingModel


@dataclass(frozen=True)
class ModelSpec:
    name: EmbeddingModel
    collection_name: str
    dimension: int
    sample_rate: int
    strategy: Literal["features", "segments", "full"]


MODEL_SPECS: dict[EmbeddingModel, ModelSpec] = {
    EmbeddingModel.ACOUSTIC_FEATURES: ModelSpec(
        name=EmbeddingModel.ACOUSTIC_FEATURES,
        collection_name="acoustic_features",
        dimension=55,
        sample_rate=44100,
        strategy="features",
    ),
    EmbeddingModel.CLAP: ModelSpec(
        name=EmbeddingModel.CLAP,
        collection_name="clap_audio_embeddings",
        dimension=512,
        sample_rate=48000,
        strategy="segments",
    ),
    EmbeddingModel.MERT: ModelSpec(
        name=EmbeddingModel.MERT,
        collection_name="mert_audio_embeddings",
        dimension=1024,
        sample_rate=24000,
        strategy="segments",
    ),
    EmbeddingModel.MUQ: ModelSpec(
        name=EmbeddingModel.MUQ,
        collection_name="muq_audio_embeddings",
        dimension=1024,
        sample_rate=24000,
        strategy="full",
    ),
    EmbeddingModel.MUQ_MULAN: ModelSpec(
        name=EmbeddingModel.MUQ_MULAN,
        collection_name="muq_mulan_audio_embeddings",
        dimension=512,
        sample_rate=24000,
        strategy="full",
    ),
}

INFERENCE_MODELS: tuple[EmbeddingModel, ...] = tuple(
    spec.name for spec in MODEL_SPECS.values() if spec.strategy != "features"
)
