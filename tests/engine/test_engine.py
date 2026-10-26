"""Engine tests: fake models, mock server, no torch and no network."""

import shutil
import threading
from pathlib import Path

import httpx
import numpy as np
import pytest
from fakes import (
    DIM,
    RATE,
    FakeModel,
    MockServer,
    make_plan,
    make_settings,
    write_wav,
)
from fastapi.testclient import TestClient

import lainbow
from lainbow.core.method import Span
from lainbow.engine import audio
from lainbow.engine.client import (
    EngineOutdatedError,
    EngineUnauthorizedError,
    ServerClient,
    ServerUnreachableError,
)
from lainbow.engine.models import registry
from lainbow.engine.models.base import AudioModel
from lainbow.engine.protocol import PlanOut
from lainbow.engine.text_api import create_app
from lainbow.engine.worker import Worker


@pytest.fixture
def server_root(tmp_path: Path) -> Path:
    root = tmp_path / "server"
    root.mkdir()
    return root


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _worker(server: MockServer, models: dict[str, AudioModel], **settings: object) -> Worker:
    return Worker(server.client(), models, make_settings(**settings), server.plan)


@needs_ffmpeg
def test_run_once_posts_result_per_task(server_root: Path) -> None:
    server = MockServer(server_root, make_plan("a", "b", clip_sec=0.5))
    server.add_track(1)
    server.add_track(2)
    models: dict[str, AudioModel] = {"a": FakeModel("a"), "b": FakeModel("b")}
    assert _worker(server, models).run_once() == 2
    assert len(server.posted) == 1
    body = server.posted[0]
    assert len(body["results"]) == 4
    assert body["failures"] == []
    assert all(len(r["embedding"]) == DIM for r in body["results"])
    assert server.downloads == [1, 2]


@needs_ffmpeg
def test_new_profile_refreshes_plan_once(
    server_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = MockServer(server_root, make_plan("a"))
    worker = _worker(server, {"a": FakeModel("a")})
    new_plan = make_plan("a")
    new_plan.tasks[0].profile = "r1-maxnone-keephead"
    server.plan = new_plan
    server.add_track(1)
    server.add_track(2)
    calls: list[int] = []
    real = worker.client.plan

    def counting_plan() -> PlanOut:
        calls.append(1)
        return real()

    monkeypatch.setattr(worker.client, "plan", counting_plan)
    assert worker.run_once() == 2
    assert len(calls) == 1
    body = server.posted[0]
    assert body["failures"] == []
    assert {r["profile"] for r in body["results"]} == {"r1-maxnone-keephead"}


@needs_ffmpeg
def test_decoded_once_per_sample_rate(server_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = MockServer(server_root, make_plan("a", "b"))
    server.add_track(1)
    calls: list[tuple[int, float | None]] = []
    real = audio.decode

    def spy(path: Path, sample_rate: int, max_sec: float | None) -> audio.Audio:
        calls.append((sample_rate, max_sec))
        return real(path, sample_rate, max_sec)

    monkeypatch.setattr(audio, "decode", spy)
    _worker(server, {"a": FakeModel("a"), "b": FakeModel("b")}).run_once()
    assert [(sr, round(m or 0, 3)) for sr, m in calls] == [(RATE, 1.0)]


@needs_ffmpeg
def test_music_dir_skips_download(server_root: Path, tmp_path: Path) -> None:
    server = MockServer(server_root, make_plan("a"))
    server.add_track(1)
    music = tmp_path / "music"
    music.mkdir()
    shutil.copy(server_root / "t1.wav", music / "t1.wav")
    assert _worker(server, {"a": FakeModel("a")}, music_dir=music).run_once() == 1
    assert server.downloads == []
    assert len(server.posted[0]["results"]) == 1


@needs_ffmpeg
def test_model_exception_becomes_failure(server_root: Path) -> None:
    server = MockServer(server_root, make_plan("a", "b"))
    server.add_track(1)
    models: dict[str, AudioModel] = {"a": FakeModel("a", fail=True), "b": FakeModel("b")}
    _worker(server, models).run_once()
    body = server.posted[0]
    assert [r["model"] for r in body["results"]] == ["b"]
    assert [f["model"] for f in body["failures"]] == ["a"]
    assert body["failures"][0]["error"] == "RuntimeError: boom"


def test_no_work_returns_zero(server_root: Path) -> None:
    server = MockServer(server_root, make_plan("a"))
    assert _worker(server, {"a": FakeModel("a")}).run_once() == 0
    assert server.posted == []


def test_unauthorized(server_root: Path) -> None:
    server = MockServer(server_root, make_plan("a"))
    server.unauthorized = True
    with pytest.raises(EngineUnauthorizedError, match="lainbow add-engine"):
        server.client().claim(["a"], 1)


def test_unreachable() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    client = ServerClient("http://server", "tok", transport=httpx.MockTransport(refuse))
    with pytest.raises(ServerUnreachableError):
        client.plan()


def test_client_sends_version_header_on_every_request(tmp_path: Path) -> None:
    seen_headers: list[dict[str, str]] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(dict(request.headers))
        url = request.url.path
        if url.endswith("/plan"):
            return httpx.Response(200, json={"tasks": []})
        if url.endswith("/claim"):
            return httpx.Response(200, json={"items": []})
        if url.endswith("/results"):
            return httpx.Response(200, json={"accepted": 0, "rejected": 0, "failed": 0})
        if url.endswith("/release"):
            return httpx.Response(200, json={"released": 0})
        if "/audio" in url:
            return httpx.Response(200, content=b"audio-data")
        return httpx.Response(404)

    client = ServerClient("http://server", "tok", transport=httpx.MockTransport(mock_handler))
    client.plan()
    client.claim(["m"], 1)
    client.post_results([], [])
    client.release()
    client.download(1, tmp_path / "t.wav")

    assert len(seen_headers) == 5
    for headers in seen_headers:
        assert headers.get("x-lainbow-version") == lainbow.__version__


def test_client_raises_engine_outdated_error_on_409() -> None:
    detail_msg = "server is 0.2.0 but this engine is 0.1.0; on the engine machine run: git..."

    def handler_409(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"detail": detail_msg})

    client = ServerClient("http://server", "tok", transport=httpx.MockTransport(handler_409))
    with pytest.raises(EngineOutdatedError) as exc_info:
        client.plan()
    assert exc_info.value.detail == detail_msg
    assert str(exc_info.value) == detail_msg


def test_worker_run_forever_propagates_engine_outdated_error(
    server_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    server = MockServer(server_root, make_plan("a"))
    detail_msg = "server is 0.2.0 but this engine is 0.1.0"

    def handler_409(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"detail": detail_msg})

    client = ServerClient("http://server", "tok", transport=httpx.MockTransport(handler_409))
    worker = Worker(client, {"a": FakeModel("a")}, make_settings(), server.plan)
    stop = threading.Event()
    with caplog.at_level("WARNING"), pytest.raises(EngineOutdatedError) as exc_info:
        worker.run_forever(stop)

    assert exc_info.value.detail == detail_msg
    assert "could not release leases" in caplog.text


def test_stop_calls_release(server_root: Path) -> None:
    server = MockServer(server_root, make_plan("a"))
    stop = threading.Event()
    stop.set()
    _worker(server, {"a": FakeModel("a")}).run_forever(stop)
    assert server.released == 1


@needs_ffmpeg
def test_decode_and_slice(tmp_path: Path) -> None:
    wav = tmp_path / "s.wav"
    write_wav(wav, 1.0)
    data = audio.decode(wav, 8000, None)
    assert data.dtype == np.float32
    assert len(data) == 8000
    assert len(audio.decode(wav, 8000, 0.5)) == 4000
    clips = audio.slice_clips(data, 8000, [Span(0, 0.25), Span(0.25, 1.0)])
    assert [len(c) for c in clips] == [2000, 6000]


def test_decode_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    with pytest.raises(audio.DecodeError):
        audio.decode(bad, 8000, None)


def test_text_api() -> None:
    http = TestClient(create_app({"a": FakeModel("a")}))
    r = http.post("/text-embedding", json={"model": "a", "text": "hello"})
    assert r.status_code == 200
    assert r.json()["embedding"] == list(FakeModel().embed_text("hello"))
    assert http.post("/text-embedding", json={"model": "zzz", "text": "x"}).status_code == 404


def test_registry_excludes_import_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(_dir: Path) -> AudioModel:
        raise ImportError("no torch")

    monkeypatch.setattr(registry, "FACTORIES", {})
    monkeypatch.setitem(registry.FACTORIES, "ok", lambda _dir: FakeModel("ok"))
    monkeypatch.setitem(registry.FACTORIES, "broken", broken)
    assert registry.available(Path("/nonexistent")) == {"ok"}
