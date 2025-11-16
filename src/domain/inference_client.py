import io
import logging
from typing import cast

import httpx
import numpy as np
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_fixed

from domain import config

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(10),
    wait=wait_fixed(30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)  # type: ignore[misc]
def run_inference(model_name: str, audio_data: np.ndarray) -> np.ndarray:
    """
    Sends a batch of audio data to the inference API and returns the embeddings.

    Args:
        model_name: The name of the model to use for inference (e.g., 'mert', 'clap').
        audio_data: A stacked NumPy array of audio segments (shape: [num_segments, num_samples]).

    Returns:
        A stacked NumPy array of embeddings (shape: [num_segments, embedding_dim]).

    Raises:
        httpx.HTTPStatusError: If the API returns an error response.
        ValueError: If the response from the API cannot be parsed.
    """
    api_url = f"{config.INFERENCE_API_URL}/api/v1/inference/{model_name}"

    with io.BytesIO() as buffer:
        np.save(buffer, audio_data)
        buffer.seek(0)
        files = {"file": ("data.npy", buffer, "application/octet-stream")}

        try:
            with httpx.Client() as client:
                logger.info(f"Sending inference request for '{model_name}' to {api_url}")
                response = client.post(api_url, files=files, timeout=config.INFERENCE_REQUEST_TIMEOUT)
                response.raise_for_status()

            with io.BytesIO(response.content) as buffer:
                embedding = cast(np.ndarray, np.load(buffer))
                logger.info(f"Successfully received embedding from '{model_name}' API.")
                return embedding
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error occurred while requesting inference from {api_url}: {e.response.status_code} - {e.response.text}"
            )
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during inference request: {e}")
            raise
