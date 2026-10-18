import logging
from pathlib import Path
from typing import cast

import librosa
import numpy as np
import numpy.typing as npt

from domain import config, vector_store_utils
from domain.embedding_models import ModelSpec
from domain.inference_client import run_inference
from worker import audio

logger = logging.getLogger(__name__)

SEGMENT_DURATION_SEC = 10.0
NUM_SEGMENTS = 3

_ready_collections: set[str] = set()


def ensure_vector_collection_ready(spec: ModelSpec) -> None:
    """Creates the vector collection once per process (cached by collection name)."""
    if spec.collection_name in _ready_collections:
        return

    try:
        vector_store_utils.create_collection_if_not_exists(
            collection_name=spec.collection_name,
            dimension=spec.dimension,
            id_field="file_path",
            vector_field="embedding",
            index_params={},
        )
        logger.info(f"Vector collection '{spec.collection_name}' is ready.")
        _ready_collections.add(spec.collection_name)
    except Exception as e:
        logger.error(f"Failed to create vector collection: {e}")
        _ready_collections.discard(spec.collection_name)
        raise e


def delete_vectors(spec: ModelSpec, file_paths: list[str]) -> None:
    """Deletes vectors from the vector collection based on their file paths."""
    if not file_paths:
        logger.debug("No file paths provided to delete from vector store.")
        return
    try:
        logger.info(
            f"Attempting to delete {len(file_paths)} vectors from vector collection '{spec.collection_name}'"
        )
        vector_store_utils.delete_vectors(
            collection_name=spec.collection_name,
            ids=file_paths,
            id_field="file_path",
        )
        logger.info(f"Successfully deleted vectors for {len(file_paths)} file paths.")
    except Exception as e:
        raise Exception(f"Failed to delete vectors from vector store: {e}") from e


def _get_segments_embeddings_batch(
    spec: ModelSpec, audio_paths: list[Path]
) -> dict[str, npt.NDArray[np.float32]]:
    if not audio_paths:
        return {}

    all_segments = []
    segment_counts = []
    processed_paths = []

    for audio_path in audio_paths:
        try:
            audio_file = audio.load_audio_for_librosa(audio_path, sr=spec.sample_rate)
            audio_data, sr = librosa.load(audio_file, sr=spec.sample_rate, mono=True)

            segments = audio.extract_audio_segments(
                audio_data=audio_data,
                sample_rate=int(sr),
                segment_duration=SEGMENT_DURATION_SEC,
                num_segments=NUM_SEGMENTS + 1,
                random_sampling=False,
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

        segment_embeddings = run_inference(spec.name.value, stacked_segments)

        results: dict[str, npt.NDArray[np.float32]] = {}
        current_pos = 0
        for i, count in enumerate(segment_counts):
            path_key = processed_paths[i]
            file_embeds = segment_embeddings[current_pos : current_pos + count]
            current_pos += count
            avg_embedding = file_embeds.mean(axis=0)
            norm = np.linalg.norm(avg_embedding)
            normalized_embedding = avg_embedding / norm if norm > 0 else avg_embedding
            results[path_key] = cast(npt.NDArray[np.float32], normalized_embedding)

        return results

    except Exception as e:
        raise Exception(f"Failed to get {spec.name.value} embeddings for batch: {e}") from e


def _get_full_embeddings_batch(
    spec: ModelSpec, audio_paths: list[Path]
) -> dict[str, npt.NDArray[np.float32]]:
    if not audio_paths:
        return {}

    results: dict[str, npt.NDArray[np.float32]] = {}
    for audio_path in audio_paths:
        try:
            audio_file = audio.load_audio_for_librosa(audio_path, sr=spec.sample_rate)
            audio_data, _ = librosa.load(
                audio_file,
                sr=spec.sample_rate,
                mono=True,
                duration=config.MUQ_FILE_DURATION_SEC,
            )

            audio_batch = np.expand_dims(audio_data, axis=0)
            embedding = run_inference(spec.name.value, audio_batch)

            final_embedding = embedding[0]

            norm = np.linalg.norm(final_embedding)
            normalized_embedding = final_embedding / norm if norm > 0 else final_embedding
            results[str(audio_path)] = cast(npt.NDArray[np.float32], normalized_embedding)

        except Exception as e:
            logger.error(f"Failed to process {audio_path} for {spec.name.value}, skipping: {e}")

    return results


def get_audio_embeddings_batch(
    spec: ModelSpec, audio_paths: list[Path]
) -> dict[str, npt.NDArray[np.float32]]:
    """Computes L2-normalized audio embeddings for a batch of audio files."""
    if spec.strategy == "segments":
        return _get_segments_embeddings_batch(spec, audio_paths)
    elif spec.strategy == "full":
        return _get_full_embeddings_batch(spec, audio_paths)
    else:
        raise ValueError(f"Unsupported strategy for get_audio_embeddings_batch: {spec.strategy}")


def get_audio_embedding(spec: ModelSpec, audio_path: Path) -> npt.NDArray[np.float32]:
    """Computes an L2-normalized audio embedding for a single audio file."""
    results = get_audio_embeddings_batch(spec, [audio_path])
    path_str = str(audio_path)
    if path_str not in results:
        raise Exception(f"Failed to get {spec.name.value} embedding for {audio_path}")
    return results[path_str]
