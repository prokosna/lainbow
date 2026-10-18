from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class InferenceModel(ABC):
    """An abstract base class for all inference models."""

    @abstractmethod
    def run_inference(self, audio_data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Runs inference on a batch of audio segments.

        Args:
            audio_data: A NumPy array containing the audio segments.

        Returns:
            A NumPy array containing the results of the inference.
        """
        raise NotImplementedError

    def embed_text(self, text: str) -> npt.NDArray[np.float32]:
        """
        Embeds a text string into an embedding vector.

        Args:
            text: The text string to embed.

        Returns:
            A 1-D NumPy array containing the text embedding.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support text embedding")

    @abstractmethod
    def release(self) -> None:
        """
        Releases the model and associated resources from memory, especially from GPU VRAM.
        """
        raise NotImplementedError
