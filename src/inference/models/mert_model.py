import gc
import logging
import os
from typing import cast

import numpy as np
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from .base import InferenceModel

logger = logging.getLogger(__name__)

MODEL_NAME = "m-a-p/MERT-v1-330M"
LOCAL_MODEL_PATH = "./models/MERT-v1-330M"
START_LAYER = 12
NUM_LAYERS_TO_AVERAGE = 4


class MERTModel(InferenceModel):
    """A class to encapsulate the MERT model loading and inference logic."""

    def __init__(self) -> None:
        """Initializes and loads the MERT model and processor."""
        logger.info(f"Initializing MERT model '{MODEL_NAME}'...")
        try:
            if not os.path.exists(LOCAL_MODEL_PATH):
                raise FileNotFoundError(
                    f"MERT model not found at '{LOCAL_MODEL_PATH}'. "
                    f"Please run 'python download_models.py' to download it."
                )

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")

            logger.info(f"Loading MERT model from local path: {LOCAL_MODEL_PATH}")
            self.model = AutoModel.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
                LOCAL_MODEL_PATH, trust_remote_code=True
            )

            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info("MERT model initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize MERT model: {e}")
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

        inputs = self.processor(
            audio_segments,
            sampling_rate=self.processor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            all_layer_states = torch.stack(outputs.hidden_states, dim=1)
            time_reduced_states = all_layer_states.mean(dim=-2)
            segment_embeddings = (
                time_reduced_states[:, START_LAYER : START_LAYER + NUM_LAYERS_TO_AVERAGE, :]
                .mean(dim=1)
                .cpu()
                .numpy()
            )

        return cast(np.ndarray, segment_embeddings)

    def release(self) -> None:
        """Releases the MERT model and clears CUDA cache."""
        logger.info(f"Releasing MERT model from {self.device}...")
        del self.model
        del self.processor
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("MERT model released successfully.")
