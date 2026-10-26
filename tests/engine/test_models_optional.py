"""CLAP and MERT wrappers with fake modules (no weights)."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from lainbow.engine.models import clap, mert


class FakeClap:
    def __init__(self) -> None:
        self.batches: list[list[np.ndarray]] = []
        self.texts: list[list[str]] = []
        self.ckpt: str | None = None

    def load_ckpt(self, path: str) -> None:
        self.ckpt = path

    def get_audio_embedding_from_data(self, x: list[np.ndarray], use_tensor: bool) -> np.ndarray:
        assert not use_tensor
        self.batches.append(list(x))
        return np.array([[float(c[0]), 0.0] if i == 0 else [0.0, 1.0] for i, c in enumerate(x)])

    def get_text_embedding(self, texts: list[str], use_tensor: bool) -> np.ndarray:
        assert not use_tensor
        self.texts.append(texts)
        return np.array([[3.0, 4.0]])


def test_clap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakeClap()
    seen: dict[str, Any] = {}

    def module(**kwargs: Any) -> FakeClap:
        seen["module"] = kwargs
        return fake

    def download(**kwargs: Any) -> str:
        seen["download"] = kwargs
        return "/ckpt.pt"

    monkeypatch.setattr(clap, "_clap_module", module)
    monkeypatch.setattr(clap, "hf_hub_download", download)
    monkeypatch.setattr(clap, "SAMPLE_RATE", 2)
    monkeypatch.setattr(clap, "CLIP_SEC", 3)
    model = clap.ClapModel(tmp_path)
    model.load("cpu")
    assert seen["module"] == {"enable_fusion": False, "amodel": "HTSAT-base", "device": "cpu"}
    assert seen["download"]["cache_dir"] == tmp_path / "hf"
    assert fake.ckpt == "/ckpt.pt"
    clips = [np.full(6, 3.0, dtype=np.float32), np.full(4, 1.0, dtype=np.float32)]
    out = model.embed(clips).embedding
    # rows are [3, 0] and [0, 1]; their mean is [1.5, 0.5]
    assert out == pytest.approx(np.array([1.5, 0.5]) / np.linalg.norm([1.5, 0.5]))
    # the 4-sample clip is wrapped from the span start (3.0 ...) up to 6 samples
    assert fake.batches[0][1].tolist() == [1, 1, 1, 1, 3, 3]
    assert model.embed_text("calm piano") == pytest.approx([0.6, 0.8])
    assert fake.texts == [["calm piano"]]


class FakeMert(torch.nn.Module):
    """Layer ``i`` has a constant hidden state of ``i`` plus a per-clip offset."""

    def forward(self, input_values: torch.Tensor, output_hidden_states: bool = False) -> Any:
        assert output_hidden_states
        offset = input_values[:, :1]  # (clips, 1)
        layers = [
            (torch.full((input_values.shape[0], 5, 2), float(i)) + offset[:, :, None])
            for i in range(16)
        ]
        return SimpleNamespace(hidden_states=tuple(layers))


def test_mert(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def processor(clips: list[np.ndarray], **kwargs: Any) -> dict[str, torch.Tensor]:
        seen["processor"] = kwargs
        return {"input_values": torch.from_numpy(np.stack(clips))}

    def from_pretrained(repo: str, **kwargs: Any) -> Any:
        seen["model"] = (repo, kwargs)
        return FakeMert()

    monkeypatch.setattr(
        mert,
        "Wav2Vec2FeatureExtractor",
        SimpleNamespace(from_pretrained=lambda _r, **_k: processor),
    )
    monkeypatch.setattr(mert, "AutoModel", SimpleNamespace(from_pretrained=from_pretrained))
    monkeypatch.setattr(mert, "SAMPLE_RATE", 2)
    monkeypatch.setattr(mert, "CLIP_SEC", 3)
    model = mert.MertModel(tmp_path)
    model.load("cpu")
    assert seen["model"] == (
        "m-a-p/MERT-v1-330M",
        {"trust_remote_code": True, "cache_dir": tmp_path / "hf"},
    )
    clips = [np.zeros(6, dtype=np.float32), np.full(6, 2.0, dtype=np.float32)]
    out = model.embed(clips).embedding
    assert seen["processor"] == {"sampling_rate": 2, "padding": True, "return_tensors": "pt"}
    # layers 12..15 average to 13.5; clip offsets are 0 and 2, so the clip mean is 14.5
    assert out == pytest.approx(np.full(2, 14.5) / np.linalg.norm(np.full(2, 14.5)))
