"""Tests for the aggregation helpers."""

import numpy as np
import pytest

from lainbow.engine.models.aggregate import l2_normalize, pad_by_wrapping, weighted_mean


def test_weighted_mean() -> None:
    out = weighted_mean([np.array([1.0, 0.0]), np.array([0.0, 1.0])], [3.0, 1.0])
    assert out == pytest.approx([0.75, 0.25])


def test_weighted_mean_validates() -> None:
    with pytest.raises(ValueError, match="equal length"):
        weighted_mean([np.zeros(2)], [1.0, 2.0])
    with pytest.raises(ValueError, match="positive"):
        weighted_mean([np.zeros(2)], [0.0])


def test_l2_normalize() -> None:
    assert l2_normalize(np.array([3.0, 4.0])) == pytest.approx([0.6, 0.8])


def test_l2_normalize_zero_vector() -> None:
    out = l2_normalize(np.zeros(3))
    assert out.tolist() == [0.0, 0.0, 0.0]


def test_pad_by_wrapping_fills_from_span_start() -> None:
    source = np.arange(10, dtype=np.float32)
    out = pad_by_wrapping(source[6:], 8, source)
    assert out.tolist() == [6, 7, 8, 9, 0, 1, 2, 3]
    assert out.dtype == np.float32


def test_pad_by_wrapping_repeats_short_source_and_truncates() -> None:
    source = np.array([1.0, 2.0], dtype=np.float32)
    assert pad_by_wrapping(source, 5, source).tolist() == [1, 2, 1, 2, 1]
    assert pad_by_wrapping(np.arange(6, dtype=np.float32), 4, source).tolist() == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="empty"):
        pad_by_wrapping(np.zeros(1, dtype=np.float32), 3, np.zeros(0, dtype=np.float32))
