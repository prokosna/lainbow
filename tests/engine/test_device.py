"""Tests for device selection."""

from types import SimpleNamespace
from typing import Literal

import pytest
import torch

from lainbow.engine.device import select_device

Accel = Literal["auto", "cpu", "cuda", "rocm"]


def _gpu(monkeypatch: pytest.MonkeyPatch, present: bool) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: present)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _i: SimpleNamespace(name="Fake GPU", total_memory=8 * 2**30),
    )


@pytest.mark.parametrize(
    ("accel", "expected"),
    [("auto", "cuda"), ("cpu", "cpu"), ("cuda", "cuda"), ("rocm", "cuda")],
)
def test_with_gpu(monkeypatch: pytest.MonkeyPatch, accel: Accel, expected: str) -> None:
    _gpu(monkeypatch, True)
    assert select_device(accel) == expected


@pytest.mark.parametrize(("accel", "expected"), [("auto", "cpu"), ("cpu", "cpu")])
def test_without_gpu(monkeypatch: pytest.MonkeyPatch, accel: Accel, expected: str) -> None:
    _gpu(monkeypatch, False)
    assert select_device(accel) == expected


def test_gpu_required_but_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _gpu(monkeypatch, False)
    with pytest.raises(RuntimeError, match="--gpus all"):
        select_device("cuda")
    with pytest.raises(RuntimeError, match="/dev/kfd"):
        select_device("rocm")
