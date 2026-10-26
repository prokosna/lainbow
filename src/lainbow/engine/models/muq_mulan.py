"""MuQ-MuLan audio and text model (natural-language search)."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from muq import MuQMuLan as _MuQMuLan

from lainbow.engine.models.aggregate import l2_normalize
from lainbow.engine.models.base import EmbedResult

REPO_ID = "OpenMuQ/MuQ-MuLan-large"


class MuQMuLanModel:
    """Audio is embedded in one library call over the concatenated clips; text via ``texts=``."""

    id = "muq_mulan"

    def __init__(self, models_dir: Path) -> None:
        self._cache_dir = models_dir / "hf"
        self._device = "cpu"
        self._model: Any = None

    def load(self, device: str) -> None:
        """Download (first start) and load the weights onto the device."""
        self._device = device
        self._model = (
            _MuQMuLan.from_pretrained(REPO_ID, cache_dir=self._cache_dir).to(device).eval()
        )

    def unload(self) -> None:
        """Drop the weights and free cached device memory."""
        self._model = None
        if self._device.startswith("cuda"):
            torch.cuda.empty_cache()

    def embed(self, clips: Sequence[npt.NDArray[np.float32]]) -> EmbedResult:
        """Embed the analysis span (the clips are consecutive)."""
        if self._model is None:
            raise RuntimeError("model is not loaded")
        audio = np.array(np.concatenate(clips), dtype=np.float32, copy=True)
        with torch.inference_mode():
            wav = torch.from_numpy(audio).unsqueeze(0).to(self._device)
            vec = self._model(wavs=wav).flatten().float().cpu().numpy()
        return EmbedResult(embedding=l2_normalize(vec))

    def embed_text(self, text: str) -> npt.NDArray[np.floating[Any]]:
        """Embed a text query into the audio space."""
        if self._model is None:
            raise RuntimeError("model is not loaded")
        with torch.inference_mode():
            vec = self._model(texts=[text]).flatten().float().cpu().numpy()
        return l2_normalize(vec)
