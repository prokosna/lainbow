import logging
from pathlib import Path
from typing import cast

import librosa
import numpy as np
from domain import config, vector_store_utils
from domain.inference_client import run_inference

from worker import audio

logger = logging.getLogger(__name__)

MODEL_TYPE = "muq_mulan"
COLLECTION_NAME = "muq_mulan_audio_embeddings"
EMBEDDING_DIMENSION = 512
ID_FIELD = "file_path"
VECTOR_FIELD = "embedding"
INDEX_PARAMS = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 30, "efConstruction": 360},
}
MUQ_SAMPLING_RATE = 24000

_is_vector_collection_ready: bool = False


def delete_vectors(file_paths: list[str]) -> None:
    """
    Deletes vectors from the MuQ-MuLan vector collection based on their file paths.
    """
    if not file_paths:
        logger.debug("No file paths provided to delete from vector store.")
        return
    try:
        logger.info(
            f"Attempting to delete {len(file_paths)} vectors from vector collection '{COLLECTION_NAME}'"
        )
        vector_store_utils.delete_vectors(
            collection_name=COLLECTION_NAME,
            ids=file_paths,
            id_field=ID_FIELD,
        )
        logger.info(f"Successfully deleted vectors for {len(file_paths)} file paths.")
    except Exception as e:
        raise Exception(f"Failed to delete vectors from vector store: {e}") from e


def ensure_vector_collection_ready() -> None:
    global _is_vector_collection_ready
    if _is_vector_collection_ready:
        return

    try:
        vector_store_utils.create_collection_if_not_exists(
            collection_name=COLLECTION_NAME,
            dimension=EMBEDDING_DIMENSION,
            id_field=ID_FIELD,
            vector_field=VECTOR_FIELD,
            index_params=INDEX_PARAMS,
        )
        logger.info(f"Vector collection '{COLLECTION_NAME}' is ready.")
        _is_vector_collection_ready = True

    except Exception as e:
        logger.error(f"Failed to create vector collection: {e}")
        _is_vector_collection_ready = False
        raise e


def get_audio_embeddings_batch(audio_paths: list[Path]) -> dict[str, np.ndarray]:
    """
    Computes L2-normalized MuQ-MuLan audio embeddings for a batch of audio files.
    This sends the entire audio file for inference.
    """
    if not audio_paths:
        return {}

    results = {}
    for audio_path in audio_paths:
        try:
            audio_file = audio.load_audio_for_librosa(audio_path, sr=MUQ_SAMPLING_RATE)
            audio_data, _ = librosa.load(
                audio_file, sr=MUQ_SAMPLING_RATE, mono=True, duration=config.MUQ_FILE_DURATION_SEC
            )

            audio_batch = np.expand_dims(audio_data, axis=0)
            embedding = run_inference(MODEL_TYPE, audio_batch)

            # The result is already [1, dim], so we take the first element
            final_embedding = embedding[0]

            norm = np.linalg.norm(final_embedding)
            normalized_embedding = final_embedding / norm if norm > 0 else final_embedding
            results[str(audio_path)] = cast(np.ndarray, normalized_embedding)

        except Exception as e:
            logger.error(f"Failed to process {audio_path} for MuQ-MuLan, skipping: {e}")

    return results


def get_audio_embedding(audio_path: Path) -> np.ndarray:
    """
    Computes an L2-normalized MuQ-MuLan audio embedding for a single audio file.
    """
    results = get_audio_embeddings_batch([audio_path])
    path_str = str(audio_path)

    if path_str not in results:
        raise Exception(f"Failed to get MuQ-MuLan embedding for {audio_path}")

    return results[path_str]
