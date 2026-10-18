import io
import logging
from typing import cast

import httpx
import numpy as np
import numpy.typing as npt
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from domain import config

logger = logging.getLogger(__name__)

_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.Client(timeout=config.INFERENCE_REQUEST_TIMEOUT)
    return _client


def _is_retryable_exception(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
        ),
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (502, 503, 504)
    return False


@retry(
    retry=retry_if_exception(_is_retryable_exception),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=5, max=60),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def run_inference(model_name: str, audio_data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
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
            client = _get_client()
            logger.info(f"Sending inference request for '{model_name}' to {api_url}")
            response = client.post(api_url, files=files, timeout=config.INFERENCE_REQUEST_TIMEOUT)
            response.raise_for_status()

            with io.BytesIO(response.content) as resp_buffer:
                embedding = cast(npt.NDArray[np.float32], np.load(resp_buffer))
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
