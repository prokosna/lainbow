import gc
import logging
import os

import numpy as np
import torch
from muq import MuQ

from .base import InferenceModel

logger = logging.getLogger(__name__)

REPO_ID = "OpenMuQ/MuQ-large-msd-iter"
MODEL_PATH = os.path.join("./models", REPO_ID.split('/')[-1])


class MuQModel(InferenceModel):
    """A class to encapsulate the MuQ model loading and inference logic."""

    def __init__(self) -> None:
        logger.info(f"Initializing MuQ model with path '{MODEL_PATH}'...")
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model directory not found at '{MODEL_PATH}'.")

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = MuQ.from_pretrained(REPO_ID, cache_dir=MODEL_PATH).to(self.device).eval()
            logger.info("MuQ model initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize MuQ model: {e}")
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
                outputs = self.model(wav, output_hidden_states=True)
                embedding = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.append(embedding.cpu().numpy())

        if not all_embeddings:
            return np.array([])

        return np.vstack(all_embeddings)

    def release(self) -> None:
        """Releases the MuQ model and clears CUDA cache."""
        logger.info(f"Releasing MuQ model from {self.device}...")
        del self.model
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("MuQ model released successfully.")
