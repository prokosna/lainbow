import logging
from pathlib import Path
from typing import cast

import librosa
import numpy as np
from domain import milvus_utils
from domain.inference_client import run_inference

from worker import audio

logger = logging.getLogger(__name__)

SEGMENT_DURATION_SEC = 10.0
NUM_SEGMENTS = 3

MODEL_TYPE = "mert"
COLLECTION_NAME = "mert_audio_embeddings"
EMBEDDING_DIMENSION = 1024
ID_FIELD = "file_path"
VECTOR_FIELD = "embedding"
INDEX_PARAMS = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 256},
}
MERT_SAMPLING_RATE = 24000

_is_milvus_collection_created: bool = False


def delete_vectors(file_paths: list[str]) -> None:
    """
    Deletes vectors from the MERT Milvus collection based on their file paths.
    """
    if not file_paths:
        logger.debug("No file paths provided to delete from Milvus.")
        return
    try:
        logger.info(
            f"Attempting to delete {len(file_paths)} vectors from Milvus collection '{COLLECTION_NAME}'"
        )
        milvus_utils.delete_vectors(
            collection_name=COLLECTION_NAME,
            ids=file_paths,
            id_field=ID_FIELD,
        )
        logger.info(f"Successfully deleted vectors for {len(file_paths)} file paths.")
    except Exception as e:
        raise Exception(f"Failed to delete vectors from Milvus: {e}") from e


def create_milvus_collection_if_not_exist() -> None:
    global _is_milvus_collection_created
    if _is_milvus_collection_created:
        return

    try:
        milvus_utils.create_collection_if_not_exists(
            collection_name=COLLECTION_NAME,
            dimension=EMBEDDING_DIMENSION,
            id_field=ID_FIELD,
            vector_field=VECTOR_FIELD,
            index_params=INDEX_PARAMS,
        )
        logger.info(f"Milvus collection '{COLLECTION_NAME}' is ready.")
        _is_milvus_collection_created = True

    except Exception as e:
        logger.error(f"Failed to create Milvus collection: {e}")
        _is_milvus_collection_created = False
        raise e


def get_audio_embeddings_batch(audio_paths: list[Path]) -> dict[str, np.ndarray]:
    """
    Computes L2-normalized MERT audio embeddings for a batch of audio files.

    Args:
        audio_paths: A list of paths to the audio files.

    Returns:
        A dictionary mapping file paths (as strings) to their L2-normalized embeddings.
        Files that fail to process are omitted.
    """
    if not audio_paths:
        return {}

    all_segments = []
    segment_counts = []
    processed_paths = []

    for audio_path in audio_paths:
        try:
            audio_file = audio.load_audio_for_librosa(audio_path, sr=MERT_SAMPLING_RATE)
            audio_data, sr = librosa.load(audio_file, sr=MERT_SAMPLING_RATE, mono=True)
            segments = audio.extract_audio_segments(
                audio_data, sr, SEGMENT_DURATION_SEC, NUM_SEGMENTS + 1, False
            )
            segments_to_use = segments[:NUM_SEGMENTS]
            if segments_to_use:
                all_segments.extend(segments_to_use)
                segment_counts.append(len(segments_to_use))
                processed_paths.append(str(audio_path))
            else:
                logger.warning(f"No segments extracted for {audio_path}, skipping.")
        except Exception as e:
            logger.error(f"Failed to process {audio_path}, skipping: {e}")

    if not all_segments:
        return {}

    try:
        stacked_segments = np.stack(all_segments, axis=0)

        segment_embeddings = run_inference(MODEL_TYPE, stacked_segments)

        results = {}
        current_pos = 0
        for i, count in enumerate(segment_counts):
            path_key = processed_paths[i]
            file_embeds = segment_embeddings[current_pos : current_pos + count]
            current_pos += count
            avg_embedding = file_embeds.mean(axis=0)
            norm = np.linalg.norm(avg_embedding)
            normalized_embedding = avg_embedding / norm if norm > 0 else avg_embedding
            results[path_key] = cast(np.ndarray, normalized_embedding)

        return results

    except Exception as e:
        raise Exception(f"Failed to get MERT embeddings for batch: {e}") from e


def get_audio_embedding(audio_path: Path) -> np.ndarray:
    """
    Computes a L2-normalized MERT audio embedding for a given audio file.
    The normalized embedding is ready to be stored in Milvus for cosine similarity search.

    This function extracts segments from the audio, computes embeddings for each in a single batch,
    and returns the normalized average embedding.

    Args:
        audio_path: Path to the audio file.

    Returns:
        A L2-normalized numpy array representing the audio embedding.
    """
    """Computes a L2-normalized MERT audio embedding for a single audio file.

    This is a convenience wrapper around the batch-processing `get_audio_embeddings_batch`.
    """
    results = get_audio_embeddings_batch([audio_path])
    path_str = str(audio_path)

    if path_str not in results:
        raise Exception(f"Failed to get MERT embedding for {audio_path}")

    return results[path_str]
