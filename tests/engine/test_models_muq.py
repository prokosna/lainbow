"""MuQ and MuQ-MuLan wrappers with fake torch modules (no weights)."""

import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from lainbow.engine.models import muq, muq_mulan


class FakeMuQ(torch.nn.Module):
    """Returns a hidden state whose time-mean is chosen by the clip length."""

    def forward(self, wav: torch.Tensor, output_hidden_states: bool = False) -> Any:
        assert output_hidden_states
        vec = torch.tensor([1.0, 0.0]) if wav.shape[1] == 180 else torch.tensor([0.0, 1.0])
        return SimpleNamespace(last_hidden_state=vec.repeat(1, 5, 1))


class FakeMuLan(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    def forward(self, wavs: torch.Tensor | None = None, texts: list[str] | None = None) -> Any:
        self.calls.append({"wavs": wavs, "texts": texts})
        return torch.tensor([[3.0, 4.0]])


def test_muq_weighted_mean(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def from_pretrained(repo: str, cache_dir: Path) -> FakeMuQ:
        seen.update(repo=repo, cache_dir=cache_dir)
        return FakeMuQ()

    monkeypatch.setattr(muq, "_MuQ", SimpleNamespace(from_pretrained=from_pretrained))
    model = muq.MuQModel(tmp_path)
    model.load("cpu")
    assert seen == {"repo": "OpenMuQ/MuQ-large-msd-iter", "cache_dir": tmp_path / "hf"}
    clips = [np.zeros(180, dtype=np.float32), np.zeros(60, dtype=np.float32)]
    out = model.embed(clips).embedding
    expected = np.array([0.75, 0.25]) / np.linalg.norm([0.75, 0.25])
    assert out == pytest.approx(expected)


def test_muq_mulan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakeMuLan()
    monkeypatch.setattr(
        muq_mulan, "_MuQMuLan", SimpleNamespace(from_pretrained=lambda _r, cache_dir: fake)
    )
    model = muq_mulan.MuQMuLanModel(tmp_path)
    model.load("cpu")
    clips = [np.ones(10, dtype=np.float32), np.ones(4, dtype=np.float32)]
    assert model.embed(clips).embedding == pytest.approx([0.6, 0.8])
    assert len(fake.calls) == 1
    assert fake.calls[0]["wavs"].shape == (1, 14)
    assert model.embed_text("calm piano") == pytest.approx([0.6, 0.8])
    assert fake.calls[1]["texts"] == ["calm piano"]


def test_muq_input_tensor_is_writable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A read-only clip must not reach torch.from_numpy (it warns about non-writable arrays)."""
    monkeypatch.setattr(
        muq, "_MuQ", SimpleNamespace(from_pretrained=lambda _r, cache_dir: FakeMuQ())
    )
    model = muq.MuQModel(tmp_path)
    model.load("cpu")
    clip = np.zeros(180, dtype=np.float32)
    clip.flags.writeable = False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.embed([clip])
