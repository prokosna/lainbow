import gc
import logging
import os

import numpy as np
import torch
from muq import MuQMuLan

from .base import InferenceModel

logger = logging.getLogger(__name__)

REPO_ID = "OpenMuQ/MuQ-MuLan-large"
MODEL_PATH = os.path.join("./models", REPO_ID.split("/")[-1])


class MuQMuLanModel(InferenceModel):
    """A class to encapsulate the MuQ-MuLan model loading and inference logic."""

    def __init__(self) -> None:
        logger.info(f"Initializing MuQ-MuLan model with path '{MODEL_PATH}'...")
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model directory not found at '{MODEL_PATH}'.")

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = (
                MuQMuLan.from_pretrained(REPO_ID, cache_dir=MODEL_PATH).to(self.device).eval()
            )
            logger.info("MuQ-MuLan model initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize MuQ-MuLan model: {e}")
            raise e

    def run_inference(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Computes embeddings for a batch of audio tracks by processing each track individually.
        """
        if not audio_data.size:
            return np.array([])

        all_embeddings = []
        for i in range(audio_data.shape[0]):
            wav = torch.from_numpy(audio_data[i]).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.model(wavs=wav)
                all_embeddings.append(embedding.cpu().numpy())

        if not all_embeddings:
            return np.array([])

        return np.vstack(all_embeddings)

    def release(self) -> None:
        """Releases the MuQ-MuLan model and clears CUDA cache."""
        logger.info(f"Releasing MuQ-MuLan model from {self.device}...")
        del self.model
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("MuQ-MuLan model released successfully.")
