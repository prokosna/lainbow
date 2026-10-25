"""Tests for analysis method, tail calculations, spans, clip splitting, and profile labels."""

import re
from dataclasses import replace

import pytest

from lainbow.core.method import (
    TAIL_FRACTION,
    TAIL_MAX_SEC,
    TAIL_MIN_TRACK_SEC,
    AnalysisMethod,
    Span,
    analysis_span,
    profile_of,
    split_clips,
    tail_sec,
)
from lainbow.core.models import MODELS, ModelId


def test_tail_rule_constants() -> None:
    assert TAIL_MIN_TRACK_SEC == 60.0
    assert TAIL_MAX_SEC == 30.0
    assert TAIL_FRACTION == 0.15


def test_analysis_method_validation() -> None:
    method = AnalysisMethod(max_sec=180, drop_tail=False)
    assert method.max_sec == 180
    assert not method.drop_tail

    method_none = AnalysisMethod(max_sec=None, drop_tail=True)
    assert method_none.max_sec is None
    assert method_none.drop_tail

    with pytest.raises(ValueError, match="max_sec must be >= 1 or None"):
        AnalysisMethod(max_sec=0, drop_tail=False)

    with pytest.raises(ValueError, match="max_sec must be >= 1 or None"):
        AnalysisMethod(max_sec=-10, drop_tail=False)


def test_tail_sec() -> None:
    assert tail_sec(59.9) == 0.0
    assert tail_sec(60.0) == 9.0
    assert tail_sec(100.0) == 15.0
    assert tail_sec(200.0) == 30.0
    assert tail_sec(1000.0) == 30.0


def test_span() -> None:
    span = Span(start=10.0, end=35.5)
    assert span.start == 10.0
    assert span.end == 35.5
    assert span.length == 25.5


@pytest.mark.parametrize(
    ("duration", "max_sec", "drop_tail", "expected_end"),
    [
        (50.0, None, True, 50.0),
        (59.9, None, True, 59.9),
        (60.0, None, True, 51.0),
        (100.0, None, True, 85.0),
        (300.0, None, True, 270.0),
        (120.0, 180, False, 120.0),
        (300.0, 180, False, 180.0),
        (1000.0, 600, True, 600.0),
        (650.0, 600, True, 600.0),
        (620.0, 600, True, 590.0),
        (1000.0, None, True, 970.0),
    ],
)
def test_analysis_span_evaluation_cases(
    duration: float,
    max_sec: int | None,
    drop_tail: bool,
    expected_end: float,
) -> None:
    method = AnalysisMethod(max_sec=max_sec, drop_tail=drop_tail)
    span = analysis_span(duration, method)
    assert span.start == 0.0
    assert pytest.approx(span.end) == expected_end


def test_analysis_span_invalid_duration() -> None:
    method = AnalysisMethod(max_sec=180, drop_tail=False)
    with pytest.raises(ValueError, match="duration must be positive"):
        analysis_span(0.0, method)
    with pytest.raises(ValueError, match="duration must be positive"):
        analysis_span(-5.0, method)


def test_split_clips() -> None:
    # (0, 25) with 10 gives (0, 10), (10, 20), (20, 25)
    clips = split_clips(Span(0.0, 25.0), 10.0)
    assert clips == [
        Span(0.0, 10.0),
        Span(10.0, 20.0),
        Span(20.0, 25.0),
    ]

    # (0, 20) with 10 gives exactly two clips
    clips_exact = split_clips(Span(0.0, 20.0), 10.0)
    assert clips_exact == [
        Span(0.0, 10.0),
        Span(10.0, 20.0),
    ]

    # (0, 5) with 10 gives one clip, (0, 5)
    clips_short = split_clips(Span(0.0, 5.0), 10.0)
    assert clips_short == [Span(0.0, 5.0)]

    # None gives one clip
    clips_none = split_clips(Span(0.0, 150.0), None)
    assert clips_none == [Span(0.0, 150.0)]

    # Positive length span never returns an empty list
    assert len(split_clips(Span(0.0, 0.01), 10.0)) > 0

    # clip_sec <= 0 raises ValueError
    with pytest.raises(ValueError, match="clip_sec must be positive"):
        split_clips(Span(0.0, 20.0), 0.0)
    with pytest.raises(ValueError, match="clip_sec must be positive"):
        split_clips(Span(0.0, 20.0), -1.0)


def test_profile_of() -> None:
    muq_spec = MODELS[ModelId.MUQ]
    method_180 = AnalysisMethod(max_sec=180, drop_tail=False)
    method_600 = AnalysisMethod(max_sec=600, drop_tail=False)
    method_none_droptail = AnalysisMethod(max_sec=None, drop_tail=True)

    label_180 = profile_of(muq_spec, method_180)
    assert label_180 == "r1-max180-keeptail"

    label_600 = profile_of(muq_spec, method_600)
    assert label_600 == "r1-max600-keeptail"

    label_none = profile_of(muq_spec, method_none_droptail)
    assert label_none == "r1-maxnone-droptail"

    # Must match ^[a-z0-9-]+$
    pattern = re.compile(r"^[a-z0-9-]+$")
    for label in (label_180, label_600, label_none):
        assert pattern.match(label) is not None

    # Bumping the revision changes the label
    bumped_spec = replace(muq_spec, revision=2)
    label_bumped = profile_of(bumped_spec, method_180)
    assert label_bumped == "r2-max180-keeptail"
    assert label_bumped != label_180
