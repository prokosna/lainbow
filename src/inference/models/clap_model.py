import gc
import logging
import os
from typing import cast

import laion_clap
import numpy as np
import torch

from .base import InferenceModel

logger = logging.getLogger(__name__)

AUDIO_MODEL_TYPE = "HTSAT-base"
MODEL_CHECKPOINT_PATH = "./models/music_audioset_epoch_15_esc_90.14.pt"


class CLAPModel(InferenceModel):
    """A class to encapsulate the CLAP model loading and inference logic."""

    def __init__(self) -> None:
        """Initializes and loads the CLAP model."""
        logger.info(f"Initializing CLAP model with checkpoint '{MODEL_CHECKPOINT_PATH}'...")
        try:
            if not os.path.exists(MODEL_CHECKPOINT_PATH):
                raise FileNotFoundError(f"Model checkpoint not found at '{MODEL_CHECKPOINT_PATH}'.")

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = laion_clap.CLAP_Module(
                enable_fusion=False, amodel=AUDIO_MODEL_TYPE, device=self.device
            )
            self.model.load_ckpt(MODEL_CHECKPOINT_PATH)
            self.model.eval()
            logger.info("CLAP model initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize CLAP model: {e}")
            raise e

    def run_inference(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Computes embeddings for a batch of audio segments.

        Args:
            audio_data: A stacked numpy array of audio segments.

        Returns:
            A NumPy array containing the embeddings for the segments.
        """
        if not audio_data.size:
            return np.array([])

        audio_segments = list(audio_data)

        embeddings = self.model.get_audio_embedding_from_data(x=audio_segments, use_tensor=False)
        return cast(np.ndarray, embeddings)

    def release(self) -> None:
        """Releases the CLAP model and clears CUDA cache."""
        logger.info(f"Releasing CLAP model from {self.device}...")
        del self.model
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("CLAP model released successfully.")
