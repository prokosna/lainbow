"""Runs the real models; needs weights already present in LAINBOW_TEST_MODELS_DIR."""

import os
from pathlib import Path

import numpy as np
import pytest

from lainbow.engine.models.base import AudioModel

pytestmark = pytest.mark.models

_DIR = os.environ.get("LAINBOW_TEST_MODELS_DIR")
RATE = 24000


def _tone() -> np.ndarray:
    t = np.arange(30 * RATE) / RATE
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.mark.skipif(not _DIR or not Path(_DIR).is_dir(), reason="LAINBOW_TEST_MODELS_DIR not set")
@pytest.mark.parametrize(
    ("model_id", "dim"), [("muq", 1024), ("muq_mulan", 512), ("clap", 512), ("mert", 1024)]
)
def test_real_model(model_id: str, dim: int) -> None:
    from lainbow.engine.models import registry

    assert _DIR is not None
    os.environ.setdefault("HF_HOME", str(Path(_DIR) / "hf"))
    model: AudioModel = registry.FACTORIES[model_id](Path(_DIR))
    model.load("cpu")
    tone = _tone()
    a = model.embed([tone]).embedding
    b = model.embed([tone]).embedding
    assert a.shape == (dim,)
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-4)
    assert a == pytest.approx(b, abs=1e-4)
