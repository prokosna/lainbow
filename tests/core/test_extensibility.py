"""Tests for model and analysis method extensibility."""

from types import MappingProxyType
from typing import Annotated

import pytest
from pydantic import Field

from lainbow.core import models
from lainbow.core.method import AnalysisMethod, LabelToken, label_tokens
from lainbow.core.models import ModelSpec, Purpose
from lainbow.core.settings import ServerSettings, describe, load_settings


def test_model_extensibility(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adding a model to MODELS allows ServerSettings and describe() to handle it."""
    fake_spec = ModelSpec(
        id="fake",
        dimension=8,
        sample_rate=16000,
        clip_sec=10.0,
        revision=1,
        description="Fake 8-dim model for testing",
        short_name="Fake",
        purposes=frozenset({Purpose.SIMILAR}),
        default_method=AnalysisMethod(max_sec=180, drop_tail=False),
    )
    fake_models: dict[str, ModelSpec] = {str(k): v for k, v in models.MODELS.items()}
    fake_models["fake"] = fake_spec
    monkeypatch.setattr(models, "MODELS", MappingProxyType(fake_models))

    monkeypatch.setenv("LAINBOW_DATABASE_URL", "postgresql://localhost/test")
    monkeypatch.setenv("LAINBOW_FAKE_MAX_SEC", "42")

    settings = load_settings(ServerSettings)

    # 1. ServerSettings reads LAINBOW_FAKE_MAX_SEC=42
    assert settings.methods["fake"].max_sec == 42

    # 2. describe() shows LAINBOW_FAKE_MAX_SEC
    rows = describe(settings)
    fake_rows = [r for r in rows if r.env_var == "LAINBOW_FAKE_MAX_SEC"]
    assert len(fake_rows) == 1
    fake_row = fake_rows[0]
    assert fake_row.value == "42"
    assert not fake_row.is_default
    assert "Fake: seconds from the start of a track to analyze" in fake_row.description

    # 3. method_for returns 42
    assert settings.method_for("fake").max_sec == 42


def test_method_parameter_extensibility() -> None:
    """A new method parameter declaring a neutral value adds to label only when non-neutral."""

    class ExtendedMethod(AnalysisMethod):
        skip_intro_sec: Annotated[
            int,
            LabelToken(
                neutral=0,
                token=lambda v: f"skip{v}",
            ),
        ] = Field(
            default=0,
            ge=0,
            description="seconds to skip from track start",
        )

    # Neutral value (0) -> not included in label tokens
    m_default = ExtendedMethod(max_sec=180, drop_tail=False, skip_intro_sec=0)
    assert label_tokens(m_default) == ["max180", "keeptail"]

    # Non-neutral value (15) -> included in label tokens
    m_skipped = ExtendedMethod(max_sec=180, drop_tail=False, skip_intro_sec=15)
    assert label_tokens(m_skipped) == ["max180", "keeptail", "skip15"]
