"""CLAP audio and text model (optional; similar-song and natural-language search)."""

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from huggingface_hub import hf_hub_download

from lainbow.engine.models.aggregate import l2_normalize, pad_by_wrapping
from lainbow.engine.models.base import EmbedResult

REPO_ID = "lukewys/laion_clap"
CHECKPOINT = "music_audioset_epoch_15_esc_90.14.pt"
SAMPLE_RATE = 48000
CLIP_SEC = 10


def _clap_module(**kwargs: Any) -> Any:
    """Build a ``CLAP_Module``; laion_clap parses ``sys.argv`` at import, so hide the real one."""
    argv = sys.argv
    sys.argv = argv[:1]
    try:
        from laion_clap import CLAP_Module
    finally:
        sys.argv = argv
    return CLAP_Module(**kwargs)


class ClapModel:
    """Each clip is padded to 10 s by wrapping; the track vector is the normalized clip mean."""

    id = "clap"

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir
        self._device = "cpu"
        self._model: Any = None

    def load(self, device: str) -> None:
        """Download (first start) and load the checkpoint onto the device."""
        self._device = device
        path = hf_hub_download(
            repo_id=REPO_ID, filename=CHECKPOINT, cache_dir=self._models_dir / "hf"
        )
        model = _clap_module(enable_fusion=False, amodel="HTSAT-base", device=device)
        model.load_ckpt(path)
        self._model = model

    def unload(self) -> None:
        """Drop the weights and free cached device memory."""
        self._model = None
        if self._device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()

    def embed(self, clips: Sequence[npt.NDArray[np.float32]]) -> EmbedResult:
        """Embed each clip and average the rows."""
        if self._model is None:
            raise RuntimeError("model is not loaded")
        length = SAMPLE_RATE * CLIP_SEC
        source = np.concatenate(clips)
        padded = [pad_by_wrapping(c, length, source) for c in clips]
        rows = np.asarray(
            self._model.get_audio_embedding_from_data(x=padded, use_tensor=False),
            dtype=np.float64,
        )
        return EmbedResult(embedding=l2_normalize(rows.mean(axis=0)))

    def embed_text(self, text: str) -> npt.NDArray[np.floating[Any]]:
        """Embed a text query into the audio space."""
        if self._model is None:
            raise RuntimeError("model is not loaded")
        rows = np.asarray(self._model.get_text_embedding([text], use_tensor=False))
        return l2_normalize(rows[0])
