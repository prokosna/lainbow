"""Audio decoding (ffmpeg) and clip slicing."""

import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt

from lainbow.core.method import Span

Audio = npt.NDArray[np.float32]


class DecodeError(Exception):
    """Raised when ffmpeg is missing or fails to decode a file."""


def decode(path: Path, sample_rate: int, max_sec: float | None) -> Audio:
    """Decode a file to mono float32 at ``sample_rate``, up to ``max_sec`` seconds."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path)]
    if max_sec is not None:
        cmd += ["-t", f"{max_sec:.3f}"]
    cmd += ["-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1"]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError as e:
        raise DecodeError("ffmpeg is not installed") from e
    if proc.returncode != 0:
        raise DecodeError(proc.stderr.decode(errors="replace").strip() or "ffmpeg failed")
    usable = len(proc.stdout) - len(proc.stdout) % 4
    return np.frombuffer(proc.stdout[:usable], dtype=np.float32)


def slice_clips(audio: Audio, sample_rate: int, clips: list[Span]) -> list[Audio]:
    """Cut the audio into the given clips."""
    out: list[Audio] = []
    for c in clips:
        start = round(c.start * sample_rate)
        end = round(c.end * sample_rate)
        out.append(audio[start:end])
    return out
