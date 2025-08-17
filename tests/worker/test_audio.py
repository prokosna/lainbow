from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from worker.audio import extract_audio_segments


class TestExtractAudioSegments:
    """Test cases for extract_audio_segments function."""

    def test_empty_audio_data(self) -> None:
        """Test with empty audio data."""
        audio_data = np.array([])
        with pytest.raises(ValueError, match="Audio data is empty"):
            extract_audio_segments(audio_data, 44100, 1.0, 1)

    def test_invalid_segment_duration(self) -> None:
        """Test with invalid segment duration."""
        audio_data = np.random.randn(44100)  # 1 second of audio
        with pytest.raises(ValueError, match="Segment duration must be positive"):
            extract_audio_segments(audio_data, 44100, 0, 1)
        with pytest.raises(ValueError, match="Segment duration must be positive"):
            extract_audio_segments(audio_data, 44100, -1, 1)

    def test_invalid_num_segments(self) -> None:
        """Test with invalid number of segments."""
        audio_data = np.random.randn(44100)  # 1 second of audio
        with pytest.raises(ValueError, match="Number of segments must be positive"):
            extract_audio_segments(audio_data, 44100, 1.0, 0)
        with pytest.raises(ValueError, match="Number of segments must be positive"):
            extract_audio_segments(audio_data, 44100, 1.0, -1)

    def test_segment_duration_longer_than_audio_padding(self) -> None:
        """Test padding when segment duration is longer than total audio duration."""
        # Create 0.5 second audio data
        sample_rate = 44100
        audio_duration = 0.5
        audio_data = np.random.randn(int(sample_rate * audio_duration))

        # Request 1 second segment
        segment_duration = 1.0
        segments = extract_audio_segments(audio_data, sample_rate, segment_duration, 1)

        assert len(segments) == 1
        assert len(segments[0]) == int(sample_rate * segment_duration)
        # Check that the original audio is preserved at the beginning
        np.testing.assert_array_equal(segments[0][: len(audio_data)], audio_data)
        # Check that padding is zeros
        np.testing.assert_array_equal(
            segments[0][len(audio_data) :], np.zeros(len(segments[0]) - len(audio_data))
        )

    def test_single_segment_center_extraction(self) -> None:
        """Test single segment extraction from center."""
        sample_rate = 44100
        audio_data = np.arange(sample_rate * 2)  # 2 seconds of audio with values 0, 1, 2, ...

        segments = extract_audio_segments(audio_data, sample_rate, 1.0, 1, random_sampling=False)

        assert len(segments) == 1
        assert len(segments[0]) == sample_rate
        # Should extract from center: start at 0.5 seconds
        expected_start = int(0.5 * sample_rate)
        np.testing.assert_array_equal(
            segments[0], audio_data[expected_start : expected_start + sample_rate]
        )

    def test_multiple_segments_evenly_spaced(self) -> None:
        """Test multiple segments with evenly spaced sampling."""
        sample_rate = 44100
        audio_data = np.arange(sample_rate * 4)  # 4 seconds of audio

        segments = extract_audio_segments(audio_data, sample_rate, 1.0, 3, random_sampling=False)

        assert len(segments) == 3
        for segment in segments:
            assert len(segment) == sample_rate

        # Check that segments are evenly spaced
        # Available duration for spacing: 4 - 1 = 3 seconds
        # Step between segments: 3 / (3-1) = 1.5 seconds
        expected_starts = [0, int(1.5 * sample_rate), int(3.0 * sample_rate)]
        for i, segment in enumerate(segments):
            expected_start = expected_starts[i]
            np.testing.assert_array_equal(
                segment, audio_data[expected_start : expected_start + sample_rate]
            )

    @patch("numpy.random.seed")
    @patch("numpy.random.randint")
    def test_single_segment_random_sampling(
        self, mock_randint: MagicMock, mock_seed: MagicMock
    ) -> None:
        """Test single segment with random sampling."""
        sample_rate = 44100
        audio_data = np.arange(sample_rate * 2)  # 2 seconds of audio

        # Mock random start position
        mock_randint.return_value = int(0.3 * sample_rate)  # Start at 0.3 seconds

        segments = extract_audio_segments(audio_data, sample_rate, 1.0, 1, random_sampling=True)

        assert len(segments) == 1
        assert len(segments[0]) == sample_rate
        mock_seed.assert_called_with(42)
        expected_start = int(0.3 * sample_rate)
        np.testing.assert_array_equal(
            segments[0], audio_data[expected_start : expected_start + sample_rate]
        )

    def test_too_many_segments_requested(self) -> None:
        """Test when more segments are requested than possible."""
        sample_rate = 44100
        audio_data = np.random.randn(int(sample_rate * 2.5))  # 2.5 seconds of audio

        # Request 5 segments of 1 second each (impossible with 2.5s audio)
        with patch("worker.audio.logger") as mock_logger:
            segments = extract_audio_segments(
                audio_data, sample_rate, 1.0, 5, random_sampling=False
            )

            # Should return maximum possible segments (2)
            assert len(segments) == 2
            mock_logger.warning.assert_called()

    def test_no_segments_possible(self) -> None:
        """Test when no segments are possible due to very short audio."""
        sample_rate = 44100
        audio_data = np.random.randn(int(sample_rate * 0.1))  # 0.1 seconds of audio

        # Request segments of 1 second each
        with patch("worker.audio.logger") as mock_logger:
            segments = extract_audio_segments(
                audio_data, sample_rate, 1.0, 3, random_sampling=False
            )

            # Should return entire audio as single segment (with padding)
            assert len(segments) == 1
            assert len(segments[0]) == sample_rate  # 1 second with padding
            mock_logger.warning.assert_called()

    def test_segment_boundary_handling(self) -> None:
        """Test proper handling of segment boundaries."""
        sample_rate = 44100
        audio_data = np.arange(sample_rate * 3)  # 3 seconds of audio

        segments = extract_audio_segments(audio_data, sample_rate, 1.0, 2, random_sampling=False)

        assert len(segments) == 2
        for segment in segments:
            assert len(segment) == sample_rate

        # Segments should not overlap and should be within bounds
        for segment in segments:
            assert np.all(segment >= 0)
            assert np.all(segment < len(audio_data))

    def test_exact_duration_match(self) -> None:
        """Test when segment duration exactly matches audio duration."""
        sample_rate = 44100
        audio_data = np.random.randn(sample_rate)  # Exactly 1 second

        segments = extract_audio_segments(audio_data, sample_rate, 1.0, 1)

        assert len(segments) == 1
        assert len(segments[0]) == sample_rate
        np.testing.assert_array_equal(segments[0], audio_data)

    def test_different_sample_rates(self) -> None:
        """Test with different sample rates."""
        for sample_rate in [22050, 44100, 48000]:
            audio_data = np.random.randn(sample_rate * 2)  # 2 seconds

            segments = extract_audio_segments(audio_data, sample_rate, 1.0, 1)

            assert len(segments) == 1
            assert len(segments[0]) == sample_rate
