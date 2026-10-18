from dataclasses import FrozenInstanceError

import pytest

from domain.embedding_models import INFERENCE_MODELS, MODEL_SPECS
from domain.schemas import EmbeddingModel

EXPECTED_SPECS = {
    EmbeddingModel.ACOUSTIC_FEATURES: {
        "collection_name": "acoustic_features",
        "dimension": 55,
        "sample_rate": 44100,
        "strategy": "features",
    },
    EmbeddingModel.CLAP: {
        "collection_name": "clap_audio_embeddings",
        "dimension": 512,
        "sample_rate": 48000,
        "strategy": "segments",
    },
    EmbeddingModel.MERT: {
        "collection_name": "mert_audio_embeddings",
        "dimension": 1024,
        "sample_rate": 24000,
        "strategy": "segments",
    },
    EmbeddingModel.MUQ: {
        "collection_name": "muq_audio_embeddings",
        "dimension": 1024,
        "sample_rate": 24000,
        "strategy": "full",
    },
    EmbeddingModel.MUQ_MULAN: {
        "collection_name": "muq_mulan_audio_embeddings",
        "dimension": 512,
        "sample_rate": 24000,
        "strategy": "full",
    },
}


def test_model_specs_exact_table() -> None:
    assert set(MODEL_SPECS.keys()) == set(EXPECTED_SPECS.keys())

    for model, expected in EXPECTED_SPECS.items():
        spec = MODEL_SPECS[model]
        assert spec.name == model
        assert spec.collection_name == expected["collection_name"]
        assert spec.dimension == expected["dimension"]
        assert spec.sample_rate == expected["sample_rate"]
        assert spec.strategy == expected["strategy"]


def test_inference_models() -> None:
    expected_inference = (
        EmbeddingModel.CLAP,
        EmbeddingModel.MERT,
        EmbeddingModel.MUQ,
        EmbeddingModel.MUQ_MULAN,
    )
    assert expected_inference == INFERENCE_MODELS
    assert all(MODEL_SPECS[m].strategy != "features" for m in INFERENCE_MODELS)
    assert EmbeddingModel.ACOUSTIC_FEATURES not in INFERENCE_MODELS


def test_model_spec_is_frozen() -> None:
    spec = MODEL_SPECS[EmbeddingModel.CLAP]
    with pytest.raises(FrozenInstanceError):
        spec.dimension = 128  # type: ignore[misc]
