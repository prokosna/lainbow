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

    @abstractmethod
    def release(self) -> None:
        """
        Releases the model and associated resources from memory, especially from GPU VRAM.
        """
        raise NotImplementedError
