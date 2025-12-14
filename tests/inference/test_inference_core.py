import logging
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from inference.core.model_manager import ModelManager
from inference.models.clap_model import CLAPModel
from inference.models.mert_model import MERTModel

# Configure logging for tests to see manager's output
logging.basicConfig(level=logging.INFO)


@pytest.fixture  # type: ignore[misc]
def dummy_audio_segment() -> np.ndarray:
    """Creates a single dummy audio segment as a NumPy array."""
    sample_rate = 24000
    duration = 5  # seconds
    frequency = 440  # Hz
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    amplitude = 0.5
    data = amplitude * np.sin(2.0 * np.pi * frequency * t)
    return data.astype(np.float32)


@pytest.fixture  # type: ignore[misc]
def mock_model_inits() -> Iterator[None]:
    """Patches the heavy model initializations within MERTModel and CLAPModel."""
    # Patch MERT's HuggingFace loaders
    mert_model_patch = patch("inference.models.mert_model.AutoModel.from_pretrained")
    mert_processor_patch = patch(
        "inference.models.mert_model.Wav2Vec2FeatureExtractor.from_pretrained"
    )
    # Patch CLAP's laion-clap loader
    clap_module_patch = patch("inference.models.clap_model.laion_clap.CLAP_Module")

    mock_mert_model = mert_model_patch.start()
    _ = mert_processor_patch.start()
    mock_clap_module = clap_module_patch.start()

    # Configure mocks to behave like the real objects
    mock_mert_model.return_value.to.return_value = MagicMock()
    mock_clap_instance = MagicMock()
    mock_clap_instance.load_ckpt.return_value = None
    mock_clap_module.return_value = mock_clap_instance

    yield

    mert_model_patch.stop()
    mert_processor_patch.stop()
    clap_module_patch.stop()


def test_manager_loads_and_retrieves_model(mock_model_inits: None) -> None:
    """Tests that the manager can load and cache a model on first request."""
    manager = ModelManager()
    # Mock the actual inference method to avoid computation
    with patch.object(MERTModel, "run_inference", return_value=np.array([1, 2, 3])) as mock_embed:
        model = manager.get_model("mert")
        assert isinstance(model, MERTModel)
        assert "mert" in manager._models

        # Running inference should use the cached model
        manager.run_inference("mert", np.array([[]]))
        mock_embed.assert_called_once()

        # Getting the model again should return the same instance
        model2 = manager.get_model("mert")
        assert model is model2


def test_manager_run_inference(mock_model_inits: None, dummy_audio_segment: np.ndarray) -> None:
    """Tests the run_inference method for both MERT and CLAP."""
    manager = ModelManager()
    dummy_embedding = np.random.rand(512).astype(np.float32)

    # Test MERT inference
    with patch.object(MERTModel, "run_inference", return_value=dummy_embedding) as mock_mert_embed:
        # Stack the segment to simulate a batch of one
        stacked_segment = np.stack([dummy_audio_segment])
        result = manager.run_inference("mert", stacked_segment)
        mock_mert_embed.assert_called_once()
        # Use np.testing.assert_array_equal for comparing arrays
        np.testing.assert_array_equal(mock_mert_embed.call_args[0][0], stacked_segment)
        np.testing.assert_array_equal(result, dummy_embedding)

    # Test CLAP inference
    with patch.object(CLAPModel, "run_inference", return_value=dummy_embedding) as mock_clap_embed:
        stacked_segment = np.stack([dummy_audio_segment])
        result = manager.run_inference("clap", stacked_segment)
        mock_clap_embed.assert_called_once()
        np.testing.assert_array_equal(mock_clap_embed.call_args[0][0], stacked_segment)
        np.testing.assert_array_equal(result, dummy_embedding)


def test_manager_multi_mode(mock_model_inits: None) -> None:
    """Tests that in multi-mode, both models can be loaded simultaneously."""
    manager = ModelManager(mode="multi")
    manager.get_model("mert")
    manager.get_model("clap")

    assert "mert" in manager._models
    assert "clap" in manager._models
    assert isinstance(manager._models["mert"], MERTModel)
    assert isinstance(manager._models["clap"], CLAPModel)
    assert manager._currently_loaded_model == "clap"  # Last loaded


def test_manager_single_mode(mock_model_inits: None) -> None:
    """Tests that in single-mode, loading a new model unloads the old one."""
    manager = ModelManager(mode="single")

    # Mock the release method to check if it's called
    with patch.object(MERTModel, "release") as mock_mert_release:
        # Load MERT first
        manager.get_model("mert")
        assert "mert" in manager._models
        assert manager._currently_loaded_model == "mert"

        # Now, load CLAP. This should trigger the release of MERT.
        manager.get_model("clap")
        assert "clap" in manager._models
        assert "mert" not in manager._models  # MERT should be unloaded
        assert manager._currently_loaded_model == "clap"

        mock_mert_release.assert_called_once()


def test_manager_release_model(mock_model_inits: None) -> None:
    """Tests that explicitly releasing a model works correctly."""
    manager = ModelManager()
    model = manager.get_model("mert")

    with patch.object(model, "release") as mock_release:
        manager.release_model("mert")
        mock_release.assert_called_once()
        assert "mert" not in manager._models
        assert manager._currently_loaded_model is None


def test_individual_model_inference_and_release(
    mock_model_inits: None, dummy_audio_segment: np.ndarray
) -> None:
    """Tests that model classes can run inference and release resources."""
    # MERT Model Test
    mert_model = MERTModel()
    # mert_model.model is already a mock from the fixture. Configure its return value.
    dummy_output = MagicMock()
    dummy_output.hidden_states = [torch.randn(1, 100, 1024)] * 25
    mert_model.model.return_value = dummy_output

    stacked_segment = np.stack([dummy_audio_segment])
    embedding = mert_model.run_inference(stacked_segment)
    mert_model.model.assert_called_once()
    assert isinstance(embedding, np.ndarray)
    # The output shape is (batch_size, num_features)
    assert embedding.shape == (1, 1024)  # MERT-330M has 1024 dims

    mert_model.release()
    # Check if release deleted the model attribute
    assert not hasattr(mert_model, "model")

    # CLAP Model Test
    clap_model = CLAPModel()
    # clap_model.model is also a mock. Configure its method's return value.
    clap_model.model.get_audio_embedding_from_data.return_value = np.random.rand(1, 512)
    stacked_segment = np.stack([dummy_audio_segment])
    embedding = clap_model.run_inference(stacked_segment)

    clap_model.model.get_audio_embedding_from_data.assert_called_once()
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (1, 512)

    clap_model.release()
    assert not hasattr(clap_model, "model")
