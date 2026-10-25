"""Tests for Lainbow engine main entry point."""

import os
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeModel, MockServer, make_plan, make_settings

from lainbow.engine import __main__ as engine_main
from lainbow.engine.device import DriverTooOldError, UnsupportedGpuError
from lainbow.engine.models import registry
from lainbow.engine.worker import InsufficientGpuMemoryError


@pytest.fixture(autouse=True)
def clean_lainbow_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure no LAINBOW_* environment variables leak into tests."""
    for key in list(os.environ.keys()):
        if key.startswith("LAINBOW_"):
            monkeypatch.delenv(key, raising=False)


def test_engine_main_missing_settings(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        engine_main.main()

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "LAINBOW_SERVER_URL: missing" in captured.err
    assert "LAINBOW_TOKEN: missing" in captured.err


def test_engine_main_valid_settings(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(registry, "FACTORIES", {})
    monkeypatch.setenv("LAINBOW_SERVER_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("LAINBOW_TOKEN", "super_secret_engine_token_123")

    with pytest.raises(SystemExit) as exc_info:
        engine_main.main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "no analysis models are available in this engine build" in captured.err
    # Settings table logged without raw token
    assert "LAINBOW_SERVER_URL" in captured.err
    assert "LAINBOW_TOKEN" in captured.err
    assert "********" in captured.err
    assert "super_secret_engine_token_123" not in captured.err


def test_engine_returns_stale_leases_before_claiming(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    server = MockServer(tmp_path, make_plan("muq"))
    calls: list[str] = []
    real = server.client()

    class RecordingClient:
        def plan(self) -> Any:
            calls.append("plan")
            return real.plan()

        def release(self) -> int:
            calls.append("release")
            return 3

    class StubWorker:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        def run_forever(self, _stop: Any) -> None:
            calls.append("claim")

    monkeypatch.setattr(engine_main, "ServerClient", lambda *_a: RecordingClient())
    monkeypatch.setattr(engine_main, "Worker", StubWorker)
    monkeypatch.setattr(registry, "FACTORIES", {"muq": lambda _d: FakeModel("muq")})
    settings = make_settings(accel="cpu", models_dir=tmp_path, model_loading="together")
    with caplog.at_level("INFO"):
        assert engine_main.run(settings) == 0
    assert calls == ["plan", "release", "claim"]
    assert "returned 3 tracks this engine held before it last stopped" in caplog.text


@pytest.mark.parametrize("failing", ["select_device", "preflight"])
def test_unsupported_gpu_exits_3_without_claiming(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    failing: str,
) -> None:
    server = MockServer(tmp_path, make_plan("muq"))
    claims: list[str] = []

    class StubWorker:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            claims.append("worker")

    def unsupported(*_a: Any) -> Any:
        raise UnsupportedGpuError("GTX 1060 (compute capability 6.1) is not supported")

    def old_driver(*_a: Any) -> Any:
        raise DriverTooOldError("the NVIDIA driver supports CUDA 12.2")

    if failing == "select_device":
        monkeypatch.setattr(engine_main, "select_device", old_driver)
    else:
        monkeypatch.setattr(engine_main, "select_device", lambda _a: "cuda")
        monkeypatch.setattr(engine_main, "preflight", unsupported)
    monkeypatch.setattr(engine_main, "ServerClient", lambda *_a: server.client())
    monkeypatch.setattr(engine_main, "Worker", StubWorker)
    monkeypatch.setattr(registry, "FACTORIES", {"muq": lambda _d: FakeModel("muq")})
    settings = make_settings(accel="auto", models_dir=tmp_path)
    with caplog.at_level("ERROR"):
        assert engine_main.run(settings) == 3
    assert claims == []
    assert server.claims == []
    assert "GTX 1060" in caplog.text or "driver supports" in caplog.text


def test_insufficient_gpu_memory_exits_3(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    server = MockServer(tmp_path, make_plan("muq"))

    class StubWorker:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        def run_forever(self, _stop: Any) -> None:
            raise InsufficientGpuMemoryError("GPU memory is too small for muq")

    monkeypatch.setattr(engine_main, "ServerClient", lambda *_a: server.client())
    monkeypatch.setattr(engine_main, "Worker", StubWorker)
    monkeypatch.setattr(registry, "FACTORIES", {"muq": lambda _d: FakeModel("muq")})
    settings = make_settings(accel="cpu", models_dir=tmp_path, model_loading="together")
    with caplog.at_level("ERROR"):
        assert engine_main.run(settings) == 3
    assert "GPU memory is too small" in caplog.text
