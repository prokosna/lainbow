import logging
import os
from functools import lru_cache
from typing import cast

import laion_clap
import numpy as np
import torch

logger = logging.getLogger(__name__)

AUDIO_MODEL_TYPE = "HTSAT-base"
MODEL_CHECKPOINT_PATH = "./models/music_audioset_epoch_15_esc_90.14.pt"


class TextEmbeddingService:
    """A service to handle text embedding generation using the CLAP model."""

    def __init__(self) -> None:
        """Initializes and loads the CLAP model on the CPU."""
        logger.info(
            f"Initializing CLAP model for text embedding with checkpoint '{MODEL_CHECKPOINT_PATH}'..."
        )
        try:
            if not os.path.exists(MODEL_CHECKPOINT_PATH):
                raise FileNotFoundError(f"Model checkpoint not found at '{MODEL_CHECKPOINT_PATH}'.")

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = laion_clap.CLAP_Module(
                enable_fusion=False, amodel=AUDIO_MODEL_TYPE, device=self.device
            )
            self.model.load_ckpt(MODEL_CHECKPOINT_PATH)
            self.model.eval()
            logger.info("CLAP model for text embedding initialized successfully on CPU.")

        except Exception as e:
            logger.error(f"Failed to initialize CLAP model for text embedding: {e}")
            raise e

    def get_text_embedding(self, text: str) -> np.ndarray:
        """Generates a text embedding for a single string."""
        embedding = self.model.get_text_embedding([text], use_tensor=False)
        return cast(np.ndarray, embedding)


@lru_cache(maxsize=1)
def get_text_embedding_service() -> TextEmbeddingService:
    """Returns a cached instance of the TextEmbeddingService."""
    return TextEmbeddingService()


def get_text_embedding(text: str) -> np.ndarray:
    """Computes a text embedding for a given query string."""
    service = get_text_embedding_service()
    return service.get_text_embedding(text).squeeze(0)
