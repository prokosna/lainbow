import logging
from typing import Literal

import numpy as np
from domain import config

from inference.models.base import InferenceModel
from inference.models.clap_model import CLAPModel
from inference.models.mert_model import MERTModel
from inference.models.muq_model import MuQModel
from inference.models.muq_mulan_model import MuQMuLanModel

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, type[InferenceModel]] = {
    "mert": MERTModel,
    "clap": CLAPModel,
    "muq": MuQModel,
    "muq_mulan": MuQMuLanModel,
}


ModelManagerMode = Literal["single", "multi"]


class ModelManager:
    """
    Manages the lifecycle of inference models, including loading, caching, and execution.
    This class ensures that each model is loaded only once into memory (singleton pattern),
    which is crucial for conserving GPU VRAM.
    It can operate in two modes:
    - 'multi': Allows multiple models to be loaded in memory simultaneously.
    - 'single': Ensures only one model is loaded at any given time to save VRAM.
    """

    def __init__(self, mode: ModelManagerMode = "multi") -> None:
        self.mode: ModelManagerMode = mode
        self._models: dict[str, InferenceModel] = {}
        self._currently_loaded_model: str | None = None
        logger.info(f"ModelManager initialized in '{self.mode}' mode.")
        if len(config.INFERENCE_LOAD_MODELS_ON_STARTUP) > 0:
            for model_name in config.INFERENCE_LOAD_MODELS_ON_STARTUP:
                if model_name in MODEL_REGISTRY:
                    self.get_model(model_name)
                else:
                    logger.warning(f"Model '{model_name}' is not registered. Skipping.")
            logger.info(f"Models loaded on startup: {config.INFERENCE_LOAD_MODELS_ON_STARTUP}")

    def _load_model(self, model_name: str) -> InferenceModel:
        """Loads a model by its registered name."""
        if model_name not in MODEL_REGISTRY:
            logger.error(f"Attempted to load an unregistered model: {model_name}")
            raise ValueError(f"Model '{model_name}' is not registered.")

        logger.info(f"Loading model '{model_name}' into memory...")
        model_class = MODEL_REGISTRY[model_name]
        model_instance = model_class()
        logger.info(f"Model '{model_name}' loaded successfully.")
        return model_instance

    def get_model(self, model_name: str) -> InferenceModel:
        """Retrieves a model, loading it if necessary and handling single/multi mode."""
        if (
            self.mode == "single"
            and self._currently_loaded_model
            and self._currently_loaded_model != model_name
        ):
            logger.info(
                f"Single-model mode: Unloading '{self._currently_loaded_model}' to load '{model_name}'."
            )
            self.release_model(self._currently_loaded_model)

        if model_name not in self._models:
            self._models[model_name] = self._load_model(model_name)

        self._currently_loaded_model = model_name
        return self._models[model_name]

    def release_model(self, model_name: str) -> None:
        """Releases a specific model from memory."""
        if model_name in self._models:
            logger.info(f"Releasing model '{model_name}'...")
            self._models[model_name].release()
            del self._models[model_name]
            if self._currently_loaded_model == model_name:
                self._currently_loaded_model = None
            logger.info(f"Model '{model_name}' released.")
        else:
            logger.warning(f"Attempted to release a model that was not loaded: '{model_name}'.")

    def run_inference(self, model_name: str, input_data: np.ndarray) -> np.ndarray:
        """
        Runs inference using the specified model.

        Args:
            model_name: The name of the model to use (e.g., 'mert', 'clap').
            input_data: A NumPy array where each row is a segment of audio data.

        Returns:
            A NumPy array representing the embedding.
        """
        model = self.get_model(model_name)
        return model.run_inference(input_data)
