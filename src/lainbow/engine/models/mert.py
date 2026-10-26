"""MERT audio model (optional; similar-song search)."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from lainbow.engine.models.aggregate import l2_normalize, pad_by_wrapping
from lainbow.engine.models.base import EmbedResult

REPO_ID = "m-a-p/MERT-v1-330M"
SAMPLE_RATE = 24000
CLIP_SEC = 10
START_LAYER = 12
NUM_LAYERS = 4


class MertModel:
    """Time-mean of hidden layers 12-15 per clip; the track vector is the normalized clip mean."""

    id = "mert"

    def __init__(self, models_dir: Path) -> None:
        self._cache_dir = models_dir / "hf"
        self._device = "cpu"
        self._model: Any = None
        self._processor: Any = None

    def load(self, device: str) -> None:
        """Download (first start) and load the weights onto the device."""
        self._device = device
        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
            REPO_ID, trust_remote_code=True, cache_dir=self._cache_dir
        )
        self._model = (
            AutoModel.from_pretrained(REPO_ID, trust_remote_code=True, cache_dir=self._cache_dir)
            .to(device)
            .eval()
        )

    def unload(self) -> None:
        """Drop the weights and free cached device memory."""
        self._model = None
        self._processor = None
        if self._device.startswith("cuda"):
            torch.cuda.empty_cache()

    def embed(self, clips: Sequence[npt.NDArray[np.float32]]) -> EmbedResult:
        """Embed each clip and average the rows."""
        if self._model is None:
            raise RuntimeError("model is not loaded")
        length = SAMPLE_RATE * CLIP_SEC
        source = np.concatenate(clips)
        padded = [pad_by_wrapping(c, length, source) for c in clips]
        inputs = self._processor(
            padded, sampling_rate=SAMPLE_RATE, padding=True, return_tensors="pt"
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self._model(**inputs, output_hidden_states=True)
            hidden = torch.stack(tuple(out.hidden_states), dim=1)  # (clips, layers, time, dim)
            picked = hidden[:, START_LAYER : START_LAYER + NUM_LAYERS]
            rows = picked.mean(dim=2).mean(dim=1).float().cpu().numpy()
        return EmbedResult(embedding=l2_normalize(rows.astype(np.float64).mean(axis=0)))
