import logging
import os
from functools import lru_cache
from typing import Any, cast

import laion_clap
import numpy as np
import torch
from domain.schemas import TextEmbeddingModel
from muq import MuQMuLan

logger = logging.getLogger(__name__)

CLAP_AUDIO_MODEL_TYPE = "HTSAT-base"
CLAP_MODEL_CHECKPOINT_PATH = "./models/music_audioset_epoch_15_esc_90.14.pt"
MUQ_MULAN_REPO_ID = "OpenMuQ/MuQ-MuLan-large"
MUQ_MULAN_MODEL_PATH = os.path.join("./models", MUQ_MULAN_REPO_ID.split("/")[-1])


class TextEmbeddingService:
    """
    A service to handle text embedding generation using CLAP and MuQ-MuLan models.
    Ensures only one model is loaded in memory at a time to conserve resources.
    """

    def __init__(self) -> None:
        """Initializes the service with MuQ-MuLan model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._models: dict[TextEmbeddingModel, Any] = {}
        self._models[TextEmbeddingModel.MUQ_MULAN] = self._load_model(TextEmbeddingModel.MUQ_MULAN)
        self._currently_loaded_model: TextEmbeddingModel | None = TextEmbeddingModel.MUQ_MULAN
        logger.info("TextEmbeddingService initialized in single-model mode.")

    def _release_models(self) -> None:
        """Releases all loaded models from memory and clears the CUDA cache."""
        if not self._models:
            return

        logger.info(f"Releasing model '{self._currently_loaded_model}'...")
        self._models.clear()
        self._currently_loaded_model = None

        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("Model released and cache cleared.")

    def _load_model(self, model_name: TextEmbeddingModel) -> Any:
        """Loads a specific model by its enum name."""
        if model_name == TextEmbeddingModel.CLAP:
            logger.info(
                f"Initializing CLAP model with checkpoint '{CLAP_MODEL_CHECKPOINT_PATH}'..."
            )
            if not os.path.exists(CLAP_MODEL_CHECKPOINT_PATH):
                raise FileNotFoundError(
                    f"CLAP model checkpoint not found at '{CLAP_MODEL_CHECKPOINT_PATH}'."
                )
            model = laion_clap.CLAP_Module(
                enable_fusion=False, amodel=CLAP_AUDIO_MODEL_TYPE, device=self.device
            )
            model.load_ckpt(CLAP_MODEL_CHECKPOINT_PATH)
            model.eval()
            logger.info("CLAP model for text embedding initialized successfully.")
            return model

        elif model_name == TextEmbeddingModel.MUQ_MULAN:
            logger.info(f"Initializing MuQ-MuLan model with path '{MUQ_MULAN_MODEL_PATH}'...")
            if not os.path.exists(MUQ_MULAN_MODEL_PATH):
                raise FileNotFoundError(
                    f"MuQ-MuLan model directory not found at '{MUQ_MULAN_MODEL_PATH}'."
                )
            model = (
                MuQMuLan.from_pretrained(MUQ_MULAN_REPO_ID, cache_dir=MUQ_MULAN_MODEL_PATH)
                .to(self.device)
                .eval()
            )
            logger.info("MuQ-MuLan model for text embedding initialized successfully.")
            return model

        else:
            raise ValueError(f"Unsupported model type: {model_name}")

    def get_model(self, model_name: TextEmbeddingModel) -> Any:
        """
        Retrieves a model, loading it if necessary. If a different model is already
        loaded, it is released before loading the new one.
        """
        if self._currently_loaded_model and self._currently_loaded_model != model_name:
            logger.info(
                f"Switching models: Unloading '{self._currently_loaded_model}' to load '{model_name}'."
            )
            self._release_models()

        if model_name not in self._models:
            try:
                self._models[model_name] = self._load_model(model_name)
                self._currently_loaded_model = model_name
            except Exception as e:
                logger.error(f"Failed to load model '{model_name}': {e}")
                self._release_models()
                raise

        return self._models[model_name]

    def get_text_embedding(self, text: str, model_name: TextEmbeddingModel) -> np.ndarray:
        """Generates a text embedding for a single string using the specified model."""
        model = self.get_model(model_name)

        if model_name == TextEmbeddingModel.CLAP:
            embedding = model.get_text_embedding([text], use_tensor=False)
        elif model_name == TextEmbeddingModel.MUQ_MULAN:
            with torch.no_grad():
                embedding = model(texts=[text]).cpu().numpy()
        else:
            raise ValueError(f"Unsupported model for text embedding: {model_name}")

        return cast(np.ndarray, embedding.flatten())


@lru_cache(maxsize=1)
def get_text_embedding_service() -> TextEmbeddingService:
    """Returns a cached instance of the TextEmbeddingService."""
    return TextEmbeddingService()
