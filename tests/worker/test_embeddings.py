from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from domain.embedding_models import MODEL_SPECS
from domain.schemas import EmbeddingModel
from worker import embeddings


def test_get_audio_embeddings_batch_segments_and_full(
    make_song: Callable[..., str],
    music_root: Path,
    fake_inference: Any,
) -> None:
    song1_rel = make_song("artist/album/song1.wav", seconds=45.0)
    song2_rel = make_song("artist/album/song2.wav", seconds=45.0)
    audio_paths = [music_root / song1_rel, music_root / song2_rel]

    # Test "segments" strategy (e.g. CLAP)
    segments_spec = MODEL_SPECS[EmbeddingModel.CLAP]
    assert segments_spec.strategy == "segments"
    fake_inference.calls.clear()

    results_segments = embeddings.get_audio_embeddings_batch(segments_spec, audio_paths)

    assert len(results_segments) == 2
    assert fake_inference.call_count == 1
    assert fake_inference.calls[0].model_name == segments_spec.name.value
    # 2 files x 3 segments = 6 rows in the batched inference call
    assert fake_inference.calls[0].audio_data.shape[0] == 6

    for audio_path in audio_paths:
        vec = results_segments[str(audio_path)]
        assert len(vec) == segments_spec.dimension
        assert np.linalg.norm(vec) == pytest.approx(1.0, rel=1e-5)

    # Test "full" strategy (e.g. MUQ)
    full_spec = MODEL_SPECS[EmbeddingModel.MUQ]
    assert full_spec.strategy == "full"
    fake_inference.calls.clear()

    results_full = embeddings.get_audio_embeddings_batch(full_spec, audio_paths)

    assert len(results_full) == 2
    # 2 files produce 2 calls, each with batch size 1
    assert fake_inference.call_count == 2
    for call in fake_inference.calls:
        assert call.model_name == full_spec.name.value
        assert call.audio_data.shape[0] == 1

    for audio_path in audio_paths:
        vec = results_full[str(audio_path)]
        assert len(vec) == full_spec.dimension
        assert np.linalg.norm(vec) == pytest.approx(1.0, rel=1e-5)


def test_ensure_vector_collection_ready(
    fake_vector_store: dict[str, dict[str, list[float]]],
) -> None:
    spec = MODEL_SPECS[EmbeddingModel.CLAP]
    embeddings._ready_collections.clear()

    embeddings.ensure_vector_collection_ready(spec)
    assert spec.collection_name in embeddings._ready_collections
    assert spec.collection_name in fake_vector_store

    # Second call should be a no-op (cached)
    embeddings.ensure_vector_collection_ready(spec)
    assert spec.collection_name in embeddings._ready_collections


def test_delete_vectors(
    fake_vector_store: dict[str, dict[str, list[float]]],
) -> None:
    spec = MODEL_SPECS[EmbeddingModel.CLAP]
    fake_vector_store[spec.collection_name] = {"song1.wav": [0.1] * spec.dimension}

    embeddings.delete_vectors(spec, ["song1.wav"])
    assert "song1.wav" not in fake_vector_store[spec.collection_name]

    # Deleting with empty list is a no-op
    embeddings.delete_vectors(spec, [])
