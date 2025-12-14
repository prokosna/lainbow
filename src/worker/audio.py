import datetime
import io
import json
import logging
import os
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mutagen
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = [".flac", ".wv", ".mp3", ".wav", ".ogg", ".m4a", ".mp4"]
# Threshold for what we consider a DSD-like high sample rate (e.g., > 192kHz)
DSD_RATE_THRESHOLD = 192000


@dataclass
class AudioMetadata:
    """Dataclass to hold extracted audio metadata."""

    file_path: str
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    genre: str | None = None
    duration_seconds: int | None = None
    mtime: datetime.datetime | None = None


@dataclass
class AudioFileStatus:
    """Dataclass to hold the status of an audio file."""

    file_path: str
    status: Literal["OK", "NOT_EXIST", "STALE"]
    metadata: AudioMetadata | None = None


def find_audio_files(directory: Path) -> Iterator[Path]:
    """Recursively find all audio files in a directory."""
    for root, _, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in SUPPORTED_EXTENSIONS:
                yield Path(root) / file


def check_audio_file_status(file_path: Path, mtime: datetime.datetime | None) -> AudioFileStatus:
    """Check a single song against the filesystem."""
    if not file_path.exists():
        return AudioFileStatus(file_path=str(file_path), status="NOT_EXIST")

    try:
        mtime_dt_fs = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
        is_stale = False
        if mtime is None:
            is_stale = True
            logger.info(f"File has no mtime in DB, marking as STALE: {file_path}")
        else:
            # Ensure both datetimes are naive before comparison
            mtime_fs_naive = mtime_dt_fs.replace(tzinfo=None)
            mtime_db_naive = mtime.replace(tzinfo=None)
            time_difference = abs(mtime_fs_naive - mtime_db_naive)

            if time_difference > datetime.timedelta(seconds=10):
                is_stale = True
                logger.info(
                    f"File is STALE. Path: {file_path}, FS mtime: {mtime_fs_naive}, DB mtime: {mtime_db_naive}, Difference: {time_difference}"
                )

        if is_stale:
            metadata = extract_metadata(file_path)
            if metadata:
                return AudioFileStatus(file_path=str(file_path), status="STALE", metadata=metadata)
            return AudioFileStatus(file_path=str(file_path), status="NOT_EXIST")
    except FileNotFoundError:
        return AudioFileStatus(file_path=str(file_path), status="NOT_EXIST")

    return AudioFileStatus(file_path=str(file_path), status="OK")


def extract_metadata_bulk(file_paths: list[Path]) -> list[AudioMetadata]:
    """Extract metadata from a list of audio files."""
    results = []
    for file_path in file_paths:
        metadata = extract_metadata(file_path)
        if metadata:
            results.append(metadata)
    return results


def extract_metadata(file_path: Path) -> AudioMetadata | None:
    """Extract metadata from a single audio file."""
    mtime_ts = os.path.getmtime(file_path)
    mtime_dt = datetime.datetime.fromtimestamp(mtime_ts)

    audio = mutagen.File(file_path, easy=True)
    if audio is None:
        return None

    duration = int(audio.info.length) if hasattr(audio.info, "length") else None

    return AudioMetadata(
        file_path=str(file_path),
        title=audio.get("title", [None])[0],
        artist=audio.get("artist", [None])[0],
        album=audio.get("album", [None])[0],
        genre=audio.get("genre", [None])[0],
        duration_seconds=duration,
        mtime=mtime_dt,
    )


def load_audio_for_librosa(file_path: Path, sr: int | None = 48000) -> Path | io.BytesIO:
    """
    Loads an audio file. If it's a WavPack file (.wv), it checks for a very high
    sample rate (indicative of DSD/DXD). If found, it converts the file to PCM
    in-memory using ffmpeg before loading.
    """
    if file_path.suffix.lower() == ".wv":
        try:
            # Use ffprobe to get detailed stream info as JSON
            probe_command = [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-of",
                "json",
                str(file_path),
            ]
            process = subprocess.run(probe_command, check=True, capture_output=True, text=True)
            stream_info = json.loads(process.stdout)

            if not stream_info.get("streams"):
                raise ValueError("No streams found in file.")

            # Check the sample rate of the first audio stream
            audio_stream = stream_info["streams"][0]
            sample_rate_str = audio_stream.get("sample_rate", "0")
            sample_rate = int(sample_rate_str)

            if sample_rate > DSD_RATE_THRESHOLD:
                convert_command = [
                    "ffmpeg",
                    "-i",
                    str(file_path),
                    "-af",
                    f"aresample={sr}:resampler=soxr:dither_method=low_shibata:precision=28",
                    "-acodec",
                    "pcm_s16le",
                    "-f",
                    "wav",
                    "-",
                ]
                convert_process = subprocess.run(
                    convert_command,
                    check=True,
                    capture_output=True,
                )
                wav_in_memory = io.BytesIO(convert_process.stdout)
                return wav_in_memory
            else:
                logger.debug(
                    f"Standard sample rate ({sample_rate} Hz) detected. Loading directly: {file_path}"
                )
        except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError) as e:
            raise Exception(
                f"Failed to probe or convert {file_path}: {e}. Falling back to direct load."
            ) from e
        except FileNotFoundError as e:
            raise Exception(
                "ffmpeg/ffprobe not found. Please ensure it is installed and in your PATH."
            ) from e

    return file_path


def extract_audio_segments(
    audio_data: np.ndarray,
    sample_rate: int,
    segment_duration: float,
    num_segments: int,
    random_sampling: bool = False,
) -> list[np.ndarray]:
    """
    Extract multiple segments from audio data.

    Args:
        audio_data: Audio data as numpy array from librosa.load()
        sample_rate: Sample rate of the audio data
        segment_duration: Duration of each segment in seconds
        num_segments: Number of segments to extract
        random_sampling: If True, extract segments randomly. If False, extract evenly spaced segments.

    Returns:
        List of audio segments as numpy arrays

    Raises:
        ValueError: If the audio is too short or parameters are invalid
    """
    if audio_data.size == 0:
        raise ValueError("Audio data is empty")

    if segment_duration <= 0:
        raise ValueError("Segment duration must be positive")

    if num_segments <= 0:
        raise ValueError("Number of segments must be positive")

    total_duration = len(audio_data) / sample_rate
    segment_samples = int(segment_duration * sample_rate)

    if segment_duration > total_duration:
        logger.warning(
            f"Segment duration ({segment_duration}s) is longer than total audio duration ({total_duration:.2f}s). "
            f"Padding audio to match segment duration."
        )
        padding_samples = segment_samples - len(audio_data)
        padded_audio = np.pad(audio_data, (0, padding_samples), mode="constant", constant_values=0)
        return [padded_audio]

    max_possible_segments = int(total_duration / segment_duration)

    if num_segments > max_possible_segments:
        logger.warning(
            f"Requested {num_segments} segments of {segment_duration}s each would require "
            f"{num_segments * segment_duration:.1f}s, but audio is only {total_duration:.2f}s long. "
            f"Reducing to maximum possible: {max_possible_segments} segments."
        )
        num_segments = max_possible_segments

        if num_segments == 0:
            logger.warning("No segments possible. Returning entire audio as single segment.")
            return [audio_data]

    if num_segments == 1:
        if random_sampling:
            np.random.seed(42)
            max_start_sample = len(audio_data) - segment_samples
            start_sample = np.random.randint(0, max_start_sample + 1)
        else:
            # Single segment: extract from the center
            start_sample = int((len(audio_data) - segment_samples) / 2)

        end_sample = start_sample + segment_samples
        return [audio_data[start_sample:end_sample]]

    segments = []

    if random_sampling:
        np.random.seed(42)

        available_start = 0
        available_end = len(audio_data) - segment_samples

        for i in range(num_segments):
            remaining_segments = num_segments - i
            available_length = available_end - available_start

            if available_length <= 0:
                logger.warning(f"Insufficient space for segment {i + 1}")
                break

            interval_size = available_length // remaining_segments
            random_offset = np.random.randint(0, interval_size + 1) if interval_size > 0 else 0
            start_sample = available_start + random_offset
            end_sample = start_sample + segment_samples
            segments.append(audio_data[start_sample:end_sample])
            available_start = end_sample

    if not random_sampling:
        # Evenly spaced sampling
        available_duration = total_duration - segment_duration

        if num_segments == 1:
            pass
        else:
            step = available_duration / (num_segments - 1)

            for i in range(num_segments):
                start_time = i * step
                start_sample = int(start_time * sample_rate)
                end_sample = start_sample + segment_samples

                if end_sample > len(audio_data):
                    end_sample = len(audio_data)
                    start_sample = end_sample - segment_samples

                segments.append(audio_data[start_sample:end_sample])

    return segments
