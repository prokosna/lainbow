"""Tests for ModelId, ModelSpec, and MODELS definitions."""

import re
from types import MappingProxyType

import pytest

from lainbow.core.models import MODELS, ModelId, ModelSpec


def test_model_id_members() -> None:
    assert ModelId.MUQ.value == "muq"
    assert ModelId.MUQ_MULAN.value == "muq_mulan"
    assert ModelId.CLAP.value == "clap"
    assert ModelId.MERT.value == "mert"
    assert ModelId.ACOUSTIC.value == "acoustic"
    assert ModelId("muq") is ModelId.MUQ
    assert isinstance(ModelId.MUQ, str)
    assert len(ModelId) == 5


def test_models_spec_properties() -> None:
    assert isinstance(MODELS, MappingProxyType)
    assert set(MODELS.keys()) == {
        ModelId.MUQ,
        ModelId.MUQ_MULAN,
        ModelId.CLAP,
        ModelId.MERT,
        ModelId.ACOUSTIC,
    }

    muq = MODELS[ModelId.MUQ]
    assert muq.id == ModelId.MUQ
    assert muq.dimension == 1024
    assert muq.sample_rate == 24000
    assert muq.clip_sec == 180.0
    assert muq.revision == 1
    assert muq.description == "MuQ: similar-song search"

    muq_mulan = MODELS[ModelId.MUQ_MULAN]
    assert muq_mulan.id == ModelId.MUQ_MULAN
    assert muq_mulan.dimension == 512
    assert muq_mulan.sample_rate == 24000
    assert muq_mulan.clip_sec == 10.0
    assert muq_mulan.revision == 1
    assert muq_mulan.description == "MuQ-MuLan: natural-language search"

    clap = MODELS[ModelId.CLAP]
    assert clap.id == ModelId.CLAP
    assert clap.dimension == 512
    assert clap.sample_rate == 48000
    assert clap.clip_sec == 10.0
    assert clap.revision == 1
    assert clap.description == "CLAP: natural-language search (optional)"

    mert = MODELS[ModelId.MERT]
    assert mert.id == ModelId.MERT
    assert mert.dimension == 1024
    assert mert.sample_rate == 24000
    assert mert.clip_sec == 10.0
    assert mert.revision == 1
    assert mert.description == "MERT: similar-song search (optional)"

    acoustic = MODELS[ModelId.ACOUSTIC]
    assert acoustic.id == ModelId.ACOUSTIC
    assert acoustic.dimension == 55
    assert acoustic.sample_rate == 44100
    assert acoustic.clip_sec is None
    assert acoustic.revision == 1
    assert acoustic.description == "Acoustic features: tempo, timbre and harmony statistics"


def test_models_mapping_is_immutable() -> None:
    with pytest.raises(TypeError):
        MODELS[ModelId.MUQ] = ModelSpec(  # type: ignore[index]
            id=ModelId.MUQ,
            dimension=1024,
            sample_rate=24000,
            clip_sec=180.0,
            revision=2,
            description="Modified",
        )


def test_model_spec_is_frozen() -> None:
    spec = MODELS[ModelId.MUQ]
    with pytest.raises(AttributeError):
        spec.revision = 2  # type: ignore[misc]


def test_module_docstring_explains_revision() -> None:
    import lainbow.core.models as m

    assert m.__doc__ is not None
    assert "revision" in m.__doc__
    assert re.search(r"checkpoint", m.__doc__, re.IGNORECASE)
