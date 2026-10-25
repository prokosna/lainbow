"""Tests for ModelId, Purpose, ModelSpec, and MODELS definitions."""

import re
from types import MappingProxyType

import pytest

from lainbow.core.method import AnalysisMethod
from lainbow.core.models import MODELS, ModelId, ModelSpec, Purpose


def test_purpose_members() -> None:
    assert Purpose.SIMILAR.value == "similar"
    assert Purpose.TEXT.value == "text"
    assert Purpose("similar") is Purpose.SIMILAR
    assert len(Purpose) == 2


def test_model_id_members() -> None:
    assert ModelId.MUQ.value == "muq"
    assert ModelId.MUQ_MULAN.value == "muq_mulan"
    assert ModelId.CLAP.value == "clap"
    assert ModelId.MERT.value == "mert"
    assert ModelId("muq") is ModelId.MUQ
    assert isinstance(ModelId.MUQ, str)
    assert len(ModelId) == 4


def test_models_spec_properties() -> None:
    assert isinstance(MODELS, MappingProxyType)
    assert set(MODELS.keys()) == {
        ModelId.MUQ,
        ModelId.MUQ_MULAN,
        ModelId.CLAP,
        ModelId.MERT,
    }

    muq = MODELS[ModelId.MUQ]
    assert muq.id == ModelId.MUQ
    assert muq.dimension == 1024
    assert muq.sample_rate == 24000
    assert muq.clip_sec == 180.0
    assert muq.revision == 1
    assert muq.description == "MuQ: similar-song search"
    assert muq.short_name == "MuQ"
    assert muq.purposes == frozenset({Purpose.SIMILAR})
    assert muq.default_method == AnalysisMethod(max_sec=180, drop_tail=False)
    assert muq.profile(muq.default_method) == "r1-max180-keeptail"

    muq_mulan = MODELS[ModelId.MUQ_MULAN]
    assert muq_mulan.id == ModelId.MUQ_MULAN
    assert muq_mulan.dimension == 512
    assert muq_mulan.sample_rate == 24000
    assert muq_mulan.clip_sec == 10.0
    assert muq_mulan.revision == 1
    assert muq_mulan.description == "MuQ-MuLan: natural-language search"
    assert muq_mulan.short_name == "MuQ-MuLan"
    assert muq_mulan.purposes == frozenset({Purpose.SIMILAR, Purpose.TEXT})
    assert muq_mulan.default_method == AnalysisMethod(max_sec=600, drop_tail=False)
    assert muq_mulan.profile(muq_mulan.default_method) == "r1-max600-keeptail"

    clap = MODELS[ModelId.CLAP]
    assert clap.id == ModelId.CLAP
    assert clap.dimension == 512
    assert clap.sample_rate == 48000
    assert clap.clip_sec == 10.0
    assert clap.revision == 1
    assert clap.description == "CLAP: natural-language search (optional)"
    assert clap.short_name == "CLAP"
    assert clap.purposes == frozenset({Purpose.SIMILAR, Purpose.TEXT})
    assert clap.default_method == AnalysisMethod(max_sec=600, drop_tail=False)

    mert = MODELS[ModelId.MERT]
    assert mert.id == ModelId.MERT
    assert mert.dimension == 1024
    assert mert.sample_rate == 24000
    assert mert.clip_sec == 10.0
    assert mert.revision == 1
    assert mert.description == "MERT: similar-song search (optional)"
    assert mert.short_name == "MERT"
    assert mert.purposes == frozenset({Purpose.SIMILAR})
    assert mert.default_method == AnalysisMethod(max_sec=600, drop_tail=False)


def test_models_mapping_is_immutable() -> None:
    with pytest.raises(TypeError):
        MODELS[ModelId.MUQ] = ModelSpec(  # type: ignore[index]
            id=ModelId.MUQ,
            dimension=1024,
            sample_rate=24000,
            clip_sec=180.0,
            revision=2,
            description="Modified",
            short_name="MuQ",
            purposes=frozenset({Purpose.SIMILAR}),
            default_method=AnalysisMethod(max_sec=180, drop_tail=False),
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
