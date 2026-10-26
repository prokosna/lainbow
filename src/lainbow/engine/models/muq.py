"""MuQ audio model (similar-song search)."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from muq import MuQ as _MuQ

from lainbow.engine.models.aggregate import l2_normalize, weighted_mean
from lainbow.engine.models.base import EmbedResult

REPO_ID = "OpenMuQ/MuQ-large-msd-iter"


class MuQModel:
    """One forward pass per clip; the track vector is the length-weighted mean, normalized."""

    id = "muq"

    def __init__(self, models_dir: Path) -> None:
        self._cache_dir = models_dir / "hf"
        self._device = "cpu"
        self._model: Any = None

    def load(self, device: str) -> None:
        """Download (first start) and load the weights onto the device."""
        self._device = device
        self._model = _MuQ.from_pretrained(REPO_ID, cache_dir=self._cache_dir).to(device).eval()

    def unload(self) -> None:
        """Drop the weights and free cached device memory."""
        self._model = None
        if self._device.startswith("cuda"):
            torch.cuda.empty_cache()

    def embed(self, clips: Sequence[npt.NDArray[np.float32]]) -> EmbedResult:
        """Embed the clips of one track."""
        if self._model is None:
            raise RuntimeError("model is not loaded")
        vectors: list[npt.NDArray[np.floating[Any]]] = []
        with torch.inference_mode():
            for clip in clips:
                wav = (
                    torch.from_numpy(np.array(clip, dtype=np.float32, copy=True))
                    .unsqueeze(0)
                    .to(self._device)
                )
                out = self._model(wav, output_hidden_states=True)
                vec = out.last_hidden_state.mean(dim=1).squeeze(0)
                vectors.append(vec.float().cpu().numpy())
        mean = weighted_mean(vectors, [float(len(c)) for c in clips])
        return EmbedResult(embedding=l2_normalize(mean))
