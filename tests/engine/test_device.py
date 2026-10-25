"""Tests for device selection."""

from types import SimpleNamespace
from typing import Literal

import pytest
import torch

from lainbow.engine import device as device_mod
from lainbow.engine.device import (
    DriverTooOldError,
    UnsupportedGpuError,
    preflight,
    select_device,
)

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


def _preflight_env(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int], archs: list[str]
) -> list[str]:
    _gpu(monkeypatch, True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_a: capability)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: archs)
    monkeypatch.setattr(device_mod, "_driver_cuda_version", lambda: 12080)
    calls: list[str] = []
    monkeypatch.setattr(device_mod, "_self_test", calls.append)
    return calls


ARCHS = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120", "compute_120"]


def test_old_driver_with_cuda_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _gpu(monkeypatch, False)
    monkeypatch.setattr(device_mod, "_driver_cuda_version", lambda: 12020)
    monkeypatch.setattr(device_mod, "_built_cuda_version", lambda: 12080)
    with pytest.raises(DriverTooOldError, match=r"supports CUDA 12\.2 .* CUDA 12\.8 or newer"):
        select_device("cuda")
    with pytest.raises(DriverTooOldError, match=r"570 or newer"):
        select_device("cuda")


def test_old_driver_with_auto_warns_and_uses_cpu(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _gpu(monkeypatch, False)
    monkeypatch.setattr(device_mod, "_driver_cuda_version", lambda: 12020)
    monkeypatch.setattr(device_mod, "_built_cuda_version", lambda: 12080)
    with caplog.at_level("WARNING"):
        assert select_device("auto") == "cpu"
    assert "update the NVIDIA driver (570 or newer)" in caplog.text


def test_no_driver_keeps_the_old_message(monkeypatch: pytest.MonkeyPatch) -> None:
    _gpu(monkeypatch, False)
    monkeypatch.setattr(device_mod, "_driver_cuda_version", lambda: 0)
    monkeypatch.setattr(device_mod, "_built_cuda_version", lambda: 12080)
    with pytest.raises(RuntimeError, match="--gpus all"):
        select_device("cuda")


def test_unsupported_capability_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _preflight_env(monkeypatch, (6, 1), ARCHS)
    with pytest.raises(UnsupportedGpuError) as e:
        preflight("cuda")
    assert "Fake GPU (compute capability 6.1) is not supported by this engine build" in str(e.value)
    assert "supports: sm_75" in str(e.value)
    assert str(e.value).endswith("join with --accel cpu")
    assert calls == []


@pytest.mark.parametrize("capability", [(7, 5), (8, 9), (12, 0), (9, 0)])
def test_supported_capability_runs_self_test(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int]
) -> None:
    calls = _preflight_env(monkeypatch, capability, ARCHS)
    preflight("cuda")
    assert calls == ["cuda"]


def test_compute_entries_do_not_count(monkeypatch: pytest.MonkeyPatch) -> None:
    _preflight_env(monkeypatch, (12, 1), ["sm_75", "compute_120"])
    with pytest.raises(UnsupportedGpuError):
        preflight("cuda")


def test_failing_self_test_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _preflight_env(monkeypatch, (8, 6), ARCHS)

    def boom(_device: str) -> None:
        raise RuntimeError("no kernel image is available")

    monkeypatch.setattr(device_mod, "_self_test", boom)
    with pytest.raises(UnsupportedGpuError, match="GPU self-test failed: no kernel image"):
        preflight("cuda")


def test_preflight_on_cpu_does_nothing() -> None:
    preflight("cpu")
