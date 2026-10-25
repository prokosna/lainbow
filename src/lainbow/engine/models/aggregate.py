"""Pure numpy helpers for combining clip embeddings."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.floating[Any]]


def weighted_mean(vectors: Sequence[Vector], weights: Sequence[float]) -> Vector:
    """Return the mean of ``vectors`` weighted by ``weights``."""
    if not vectors or len(vectors) != len(weights):
        raise ValueError("vectors and weights must be non-empty and of equal length")
    w = np.asarray(weights, dtype=np.float64)
    total = float(w.sum())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    stacked = np.stack([np.asarray(v, dtype=np.float64) for v in vectors])
    result: Vector = (stacked * w[:, None]).sum(axis=0) / total
    return result


def l2_normalize(v: Vector) -> Vector:
    """Scale ``v`` to unit length; a zero vector stays zero."""
    arr = np.asarray(v, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr
    result: Vector = arr / norm
    return result
