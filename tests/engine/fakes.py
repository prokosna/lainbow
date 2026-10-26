"""Shared fixtures: a fake model and a mock engine server backed by WAV files."""

import json
import math
import struct
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import numpy.typing as npt

from lainbow.core.method import AnalysisMethod
from lainbow.core.settings import EngineSettings
from lainbow.engine.client import ServerClient
from lainbow.engine.models.base import EmbedResult
from lainbow.engine.protocol import PlanOut, PlanTask

DIM = 8
RATE = 16000


class FakeModel:
    """Records calls; embedding is the mean of the clips padded/truncated to DIM."""

    def __init__(self, model_id: str = "fake", fail: bool = False) -> None:
        self.id = model_id
        self.fail = fail
        self.calls: list[list[int]] = []
        self.loaded: str | None = None
        self.unloads = 0
        self.log: list[str] | None = None  # optional shared event log

    def load(self, device: str) -> None:
        self.loaded = device
        if self.log is not None:
            self.log.append(f"load {self.id}")

    def unload(self) -> None:
        self.loaded = None
        self.unloads += 1
        if self.log is not None:
            self.log.append(f"unload {self.id}")

    def embed(self, clips: Sequence[npt.NDArray[np.float32]]) -> EmbedResult:
        self.calls.append([len(c) for c in clips])
        if self.fail:
            raise RuntimeError("boom")
        vec = np.zeros(DIM, dtype=np.float64)
        mean = np.concatenate(clips).astype(np.float64) if clips else vec
        vec[: min(DIM, len(mean))] = mean[:DIM]
        vec[0] += 1.0
        return EmbedResult(embedding=vec / np.linalg.norm(vec), details={"clips": len(clips)})

    def embed_text(self, text: str) -> npt.NDArray[np.floating[Any]]:
        if self.unloads and self.loaded is None:
            raise RuntimeError("model is not loaded")
        out = np.zeros(DIM)
        out[len(text) % DIM] = 1.0
        return out


def write_wav(path: Path, seconds: float = 1.0, rate: int = RATE) -> None:
    n = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(
            b"".join(
                struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate)))
                for i in range(n)
            )
        )


def make_plan(*models: str, clip_sec: float | None = None, sample_rate: int = RATE) -> PlanOut:
    return PlanOut(
        tasks=[
            PlanTask(
                model=m,
                profile="r1-maxnone-keeptail",
                method=AnalysisMethod(),
                sample_rate=sample_rate,
                clip_sec=clip_sec,
                dimension=DIM,
            )
            for m in models
        ]
    )


class MockServer:
    """Implements claim, results, release and audio over httpx.MockTransport."""

    def __init__(self, root: Path, plan: PlanOut) -> None:
        self.root = root
        self.plan = plan
        self.tracks: list[dict[str, Any]] = []
        self.posted: list[dict[str, Any]] = []
        self.released = 0
        self.downloads: list[int] = []
        self.unauthorized = False
        self.claims: list[list[str]] = []
        self._pending: dict[int, set[str]] = {}

    def add_track(self, track_id: int, duration: float = 1.0) -> None:
        path = f"t{track_id}.wav"
        write_wav(self.root / path, duration)
        self.tracks.append({"id": track_id, "path": path, "duration": duration})
        self._pending[track_id] = {p.model for p in self.plan.tasks}

    def handle(self, request: httpx.Request) -> httpx.Response:
        if self.unauthorized:
            return httpx.Response(401, json={"detail": "nope"})
        url = request.url.path
        if url.endswith("/plan"):
            return httpx.Response(200, json=self.plan.model_dump(mode="json"))
        if url.endswith("/claim"):
            wanted = set(json.loads(request.content)["models"])
            self.claims.append(sorted(wanted))
            items = []
            for t in self.tracks:
                models = self._pending[t["id"]] & wanted
                if not models:
                    continue
                self._pending[t["id"]] -= models
                items.append(
                    {
                        "track_id": t["id"],
                        "path": t["path"],
                        "duration_sec": t["duration"],
                        "lease_until": "2030-01-01T00:00:00+00:00",
                        "tasks": [
                            {"model": p.model, "profile": p.profile}
                            for p in self.plan.tasks
                            if p.model in models
                        ],
                    }
                )
            self.tracks = [t for t in self.tracks if self._pending[t["id"]]]
            return httpx.Response(200, json={"items": items})
        if url.endswith("/results"):
            body = json.loads(request.content)
            self.posted.append(body)
            return httpx.Response(
                200,
                json={
                    "accepted": len(body["results"]),
                    "rejected": 0,
                    "failed": len(body["failures"]),
                },
            )
        if url.endswith("/release"):
            self.released += 1
            return httpx.Response(200, json={"released": 0})
        if "/tracks/" in url and url.endswith("/audio"):
            track_id = int(url.split("/")[-2])
            self.downloads.append(track_id)
            path = next(t["path"] for t in self._all if t["id"] == track_id)
            return httpx.Response(200, content=(self.root / path).read_bytes())
        return httpx.Response(404)

    @property
    def _all(self) -> list[dict[str, Any]]:
        return [{"id": int(p.stem[1:]), "path": p.name} for p in self.root.glob("t*.wav")]

    def client(self) -> ServerClient:
        return ServerClient("http://server", "tok", transport=httpx.MockTransport(self.handle))


def make_settings(**kwargs: Mapping[str, Any] | Any) -> EngineSettings:
    return EngineSettings.model_validate(
        {"server_url": "http://server", "token": "tok", "text_port": 0, **kwargs}
    )
