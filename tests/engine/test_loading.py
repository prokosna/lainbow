"""Tests for the model loading mode: the decision, sequential processing and the OOM fallback."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from fakes import FakeModel, MockServer, make_plan, make_settings
from fastapi.testclient import TestClient

from lainbow.engine import audio, loading
from lainbow.engine.loading import available_memory_gb, choose_loading
from lainbow.engine.models.base import AudioModel
from lainbow.engine.text_api import create_app
from lainbow.engine.worker import InsufficientGpuMemoryError, Worker


@pytest.fixture(autouse=True)
def no_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decode to silence so that these tests do not need ffmpeg."""
    monkeypatch.setattr(audio, "decode", lambda _p, sr, _m: np.zeros(sr, dtype=np.float32))


GB = 2**30
MODELS = ["muq", "muq_mulan"]  # needs ~9 GB


def _free(monkeypatch: pytest.MonkeyPatch, gb: float) -> None:
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_a: (int(gb * GB), 16 * GB))


def test_auto_cuda_enough_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    _free(monkeypatch, 16)
    assert choose_loading("auto", "cuda", MODELS) == "together"


def test_auto_cuda_little_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    _free(monkeypatch, 6)
    assert choose_loading("auto", "cuda", MODELS) == "sequential"


def _cpu_memory(monkeypatch: pytest.MonkeyPatch, gb: float | None) -> None:
    monkeypatch.setattr(loading, "available_memory_gb", lambda: gb)


def test_auto_cpu_enough_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    _cpu_memory(monkeypatch, 32)
    assert choose_loading("auto", "cpu", MODELS) == "together"


def test_auto_cpu_little_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    _cpu_memory(monkeypatch, 6)
    assert choose_loading("auto", "cpu", MODELS) == "sequential"


def test_auto_cpu_unknown_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    _cpu_memory(monkeypatch, None)
    assert choose_loading("auto", "cpu", MODELS) == "together"


def test_auto_cpu_warns_with_docker_hint(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _cpu_memory(monkeypatch, 3)  # the largest model (5 GB) does not fit in 2.7 GB
    with caplog.at_level("WARNING"):
        assert choose_loading("auto", "cpu", MODELS) == "sequential"
    assert "Docker Desktop: Settings > Resources > Memory" in caplog.text


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_available_memory_cgroup_v2(tmp_path: Path) -> None:
    _write(tmp_path / "cg" / "memory.max", f"{8 * GB}\n")
    _write(tmp_path / "cg" / "memory.current", f"{3 * GB}\n")
    _write(tmp_path / "meminfo", f"MemAvailable: {64 * 1024 * 1024} kB\n")
    assert available_memory_gb(tmp_path / "cg", tmp_path / "meminfo") == pytest.approx(5)


def test_available_memory_cgroup_v1(tmp_path: Path) -> None:
    _write(tmp_path / "cg" / "memory" / "memory.limit_in_bytes", str(4 * GB))
    _write(tmp_path / "cg" / "memory" / "memory.usage_in_bytes", str(GB))
    _write(tmp_path / "meminfo", f"MemAvailable: {64 * 1024 * 1024} kB\n")
    assert available_memory_gb(tmp_path / "cg", tmp_path / "meminfo") == pytest.approx(3)


def test_available_memory_unlimited_uses_meminfo(tmp_path: Path) -> None:
    _write(tmp_path / "cg" / "memory.max", "max\n")
    _write(tmp_path / "meminfo", f"MemTotal: 1 kB\nMemAvailable: {2 * 1024 * 1024} kB\n")
    assert available_memory_gb(tmp_path / "cg", tmp_path / "meminfo") == pytest.approx(2)
    _write(tmp_path / "cg1" / "memory" / "memory.limit_in_bytes", str(2**63))
    assert available_memory_gb(tmp_path / "cg1", tmp_path / "meminfo") == pytest.approx(2)


def test_available_memory_meminfo_smaller_than_cgroup(tmp_path: Path) -> None:
    _write(tmp_path / "cg" / "memory.max", str(8 * GB))
    _write(tmp_path / "meminfo", f"MemAvailable: {1024 * 1024} kB\n")
    assert available_memory_gb(tmp_path / "cg", tmp_path / "meminfo") == pytest.approx(1)


def test_available_memory_unreadable(tmp_path: Path) -> None:
    assert available_memory_gb(tmp_path / "none", tmp_path / "none") is None


def test_auto_pinned_models_fit_sequentially(monkeypatch: pytest.MonkeyPatch) -> None:
    _free(monkeypatch, 7)  # together needs 9 GB; pinned muq_mulan (4) + muq (5) = 9 GB > 6.3 GB
    assert choose_loading("auto", "cuda", MODELS, {"muq_mulan"}) == "sequential"
    _free(monkeypatch, 11)  # together needs 9 GB <= 9.9 GB
    assert choose_loading("auto", "cuda", MODELS, {"muq_mulan"}) == "together"


def test_auto_warns_when_even_sequential_does_not_fit(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _free(monkeypatch, 6)  # pinned muq_mulan (4) + muq (5) does not fit in 5.4 GB
    with caplog.at_level("WARNING"):
        assert choose_loading("auto", "cuda", MODELS, {"muq_mulan"}) == "sequential"
    assert "may not fit" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING"):
        assert choose_loading("auto", "cuda", MODELS) == "sequential"  # largest alone: 5 <= 5.4
    assert "may not fit" not in caplog.text


def test_explicit_settings_are_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    _free(monkeypatch, 1)
    assert choose_loading("together", "cuda", MODELS) == "together"
    assert choose_loading("sequential", "cpu", MODELS) == "sequential"


def test_sequential_worker(tmp_path: Path) -> None:
    server = MockServer(tmp_path, make_plan("a", "b", clip_sec=0.5))
    for i in range(1, 5):
        server.add_track(i)
    log: list[str] = []
    a, b = FakeModel("a"), FakeModel("b")
    a.log = b.log = log
    models: dict[str, AudioModel] = {"a": a, "b": b}
    worker = Worker(server.client(), models, make_settings(), server.plan, loading="sequential")

    assert [worker.run_once() for _ in range(2)] == [4, 4]
    assert worker.run_once() == 0
    assert server.claims == [["a"], ["a"], ["b"], ["b"], ["a"]]
    assert log == ["load a", "unload a", "load b", "unload b", "load a"]
    posted = [(r["track_id"], r["model"]) for body in server.posted for r in body["results"]]
    assert sorted(posted) == [(t, m) for t in range(1, 5) for m in ("a", "b")]
    assert all(body["failures"] == [] for body in server.posted)


class OutOfMemoryError(RuntimeError):
    """Stands in for ``torch.cuda.OutOfMemoryError`` (matched by class name)."""


class OomModel(FakeModel):
    def embed(self, clips: Any) -> Any:
        raise OutOfMemoryError("CUDA out of memory")


def test_oom_switches_to_sequential(tmp_path: Path) -> None:
    server = MockServer(tmp_path, make_plan("a", "b", clip_sec=0.5))
    server.add_track(1)
    a, b = FakeModel("a"), OomModel("b")
    models: dict[str, AudioModel] = {"a": a, "b": b}
    worker = Worker(server.client(), models, make_settings(), server.plan)

    assert worker.run_once() >= 1
    assert worker._loading == "sequential"
    assert a.unloads == 1
    assert b.unloads == 0
    for body in server.posted:
        assert body["results"] == []
        assert body["failures"] == []


def _text_rotation(tmp_path: Path, pinned: frozenset[str]) -> tuple[FakeModel, FakeModel]:
    server = MockServer(tmp_path, make_plan("a", "b", clip_sec=0.5))
    server.add_track(1)
    a, b = FakeModel("a"), FakeModel("b")
    models: dict[str, AudioModel] = {"a": a, "b": b}
    worker = Worker(
        server.client(),
        models,
        make_settings(),
        server.plan,
        loading="sequential",
        pinned=pinned,
    )
    while worker.run_once():
        pass
    return a, b


def test_sequential_keeps_pinned_model_loaded(tmp_path: Path) -> None:
    a, b = _text_rotation(tmp_path, frozenset({"b"}))
    assert b.unloads == 0
    assert a.unloads >= 1
    client = TestClient(create_app({"a": a, "b": b}))
    r = client.post("/text-embedding", json={"model": "b", "text": "hello"})
    assert r.status_code == 200


def test_sequential_unloads_text_model_without_pin(tmp_path: Path) -> None:
    _, b = _text_rotation(tmp_path, frozenset())
    assert b.unloads >= 1


def test_oom_fallback_keeps_pinned(tmp_path: Path) -> None:
    server = MockServer(tmp_path, make_plan("a", "b", "c", clip_sec=0.5))
    server.add_track(1)
    a, b, c = FakeModel("a"), FakeModel("b"), OomModel("c")
    models: dict[str, AudioModel] = {"a": a, "b": b, "c": c}
    worker = Worker(server.client(), models, make_settings(), server.plan, pinned=frozenset({"b"}))
    worker.run_once()
    assert (a.unloads, b.unloads, c.unloads) == (1, 0, 0)


class OomAfterOneModel(FakeModel):
    def embed(self, clips: Any) -> Any:
        if self.calls:
            raise OutOfMemoryError("CUDA out of memory")
        return super().embed(clips)


def test_sequential_oom_with_one_model_is_fatal(tmp_path: Path) -> None:
    server = MockServer(tmp_path, make_plan("a", clip_sec=0.5))
    server.add_track(1)
    server.add_track(2)
    models: dict[str, AudioModel] = {"a": OomAfterOneModel("a")}
    worker = Worker(server.client(), models, make_settings(), server.plan, loading="sequential")

    with pytest.raises(InsufficientGpuMemoryError, match="GPU memory is too small for a"):
        worker.run_once()
    assert server.released == 1
    assert [r["track_id"] for body in server.posted for r in body["results"]] == [1]
    assert all(body["failures"] == [] for body in server.posted)
