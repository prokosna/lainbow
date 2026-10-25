"""The engine work loop: claim, download, decode, embed, post."""

import logging
import os
import tempfile
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from lainbow.core.method import Span, clip_plan
from lainbow.core.models import MODELS
from lainbow.core.settings import EngineSettings
from lainbow.engine import audio
from lainbow.engine.client import ServerClient, ServerUnreachableError
from lainbow.engine.loading import Loading
from lainbow.engine.models.base import AudioModel
from lainbow.engine.protocol import ClaimItem, FailureIn, PlanOut, PlanTask, ResultIn

logger = logging.getLogger(__name__)

BACKOFF_START_SEC = 5.0
BACKOFF_MAX_SEC = 60.0
MAX_DECODERS = 8

_Decoded = dict[int, audio.Audio]


def _error_text(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


def _failure(track_id: int, model: str, profile: str, e: BaseException) -> FailureIn:
    return FailureIn(track_id=track_id, model=model, profile=profile, error=_error_text(e))


def _is_out_of_memory(e: BaseException) -> bool:
    """Match torch's ``OutOfMemoryError`` by name, so that CPU-only builds can test it."""
    return type(e).__name__ == "OutOfMemoryError"


class InsufficientGpuMemoryError(RuntimeError):
    """The GPU cannot hold even one model; sequential loading cannot help."""


class _OutOfMemoryError(Exception):
    """A model ran out of device memory while embedding; carries the model id."""

    def __init__(self, model: str) -> None:
        super().__init__(model)
        self.model = model


@dataclass(frozen=True)
class _JobTask:
    model: str
    profile: str
    plan: PlanTask
    spans: list[Span]


@dataclass(frozen=True)
class _Job:
    """A claimed item with its known tasks and the decode horizon per sample rate."""

    item: ClaimItem
    tasks: tuple[_JobTask, ...]
    invalid: tuple[FailureIn, ...]
    horizons: dict[int, float]


class Worker:
    """Processes leased tracks with the loaded models."""

    def __init__(
        self,
        client: ServerClient,
        models: Mapping[str, AudioModel],
        settings: EngineSettings,
        plan: PlanOut | None = None,
        *,
        device: str = "cpu",
        loading: Loading = "together",
        pinned: frozenset[str] = frozenset(),
    ) -> None:
        self.client = client
        self.models = models
        self.settings = settings
        self._plan = plan
        self._device = device
        self._loading: Loading = loading
        # Models that are never unloaded (they serve the text endpoint).
        self._pinned = pinned
        # Models currently loaded (the caller loads all of them, or just the pinned ones).
        self._loaded: set[str] = set(models) if loading == "together" else set(pinned)
        # Sequential mode: where in the plan order the next claim starts.
        self._cursor = 0
        self._decoders = settings.decoders or min(MAX_DECODERS, os.cpu_count() or 1)

    def _plan_tasks(self, refresh: bool = False) -> dict[tuple[str, str], PlanTask]:
        if self._plan is None or refresh:
            self._plan = self.client.plan()
        return {(t.model, t.profile): t for t in self._plan.tasks}

    def _prepare(self, item: ClaimItem, plan: Mapping[tuple[str, str], PlanTask]) -> _Job:
        """Resolve the item's tasks against the plan and compute the decode horizons."""
        tasks: list[_JobTask] = []
        invalid: list[FailureIn] = []
        horizons: dict[int, float] = {}
        for ref in item.tasks:
            task = plan.get((ref.model, ref.profile))
            if task is None:
                continue
            try:
                spans = clip_plan(item.duration_sec, task.method, task.clip_sec)
            except ValueError as e:
                invalid.append(_failure(item.track_id, ref.model, ref.profile, e))
                continue
            tasks.append(_JobTask(ref.model, ref.profile, task, spans))
            end = max((s.end for s in spans), default=0.0)
            horizons[task.sample_rate] = max(horizons.get(task.sample_rate, 0.0), end)
        return _Job(item, tuple(tasks), tuple(invalid), horizons)

    def _load(self, job: _Job, tmp: Path) -> _Decoded:
        """Fetch the track's file and decode it once per sample rate."""
        item = job.item
        music_dir = self.settings.music_dir
        local = music_dir / item.path if music_dir is not None else None
        downloaded: Path | None = None
        try:
            if local is not None and local.is_file():
                source = local
            else:
                downloaded = tmp / f"{item.track_id}{Path(item.path).suffix}"
                self.client.download(item.track_id, downloaded)
                source = downloaded
            return {sr: audio.decode(source, sr, max_sec) for sr, max_sec in job.horizons.items()}
        finally:
            if downloaded is not None:
                downloaded.unlink(missing_ok=True)

    def _analyze(
        self, job: _Job, decoded: _Decoded | Exception
    ) -> tuple[list[ResultIn], list[FailureIn]]:
        """Embed every known task of the job; a load error fails all of them."""
        track_id = job.item.track_id
        results: list[ResultIn] = []
        failures: list[FailureIn] = list(job.invalid)
        for jt in job.tasks:
            try:
                if isinstance(decoded, Exception):
                    raise decoded
                clips = audio.slice_clips(
                    decoded[jt.plan.sample_rate], jt.plan.sample_rate, jt.spans
                )
                out = self.models[jt.model].embed(clips)
                results.append(
                    ResultIn(
                        track_id=track_id,
                        model=jt.model,
                        profile=jt.profile,
                        embedding=[float(x) for x in out.embedding],
                        details=dict(out.details) if out.details is not None else None,
                    )
                )
            except Exception as e:
                if _is_out_of_memory(e):
                    raise _OutOfMemoryError(jt.model) from e
                logger.warning("track %d %s failed: %s", track_id, jt.model, e)
                failures.append(_failure(track_id, jt.model, jt.profile, e))
        return results, failures

    def _known_plan(self, items: list[ClaimItem]) -> dict[tuple[str, str], PlanTask]:
        """Return the plan, refetching it once if the claim has tasks it does not know."""
        plan = self._plan_tasks()
        if any((r.model, r.profile) not in plan for it in items for r in it.tasks):
            plan = self._plan_tasks(refresh=True)
            unknown = {
                (r.model, r.profile)
                for it in items
                for r in it.tasks
                if (r.model, r.profile) not in plan
            }
            if unknown:
                logger.warning(
                    "skipping tasks missing from the server plan (leases will expire): %s",
                    sorted(unknown),
                )
        return plan

    def _activate(self, model_id: str) -> None:
        """Sequential mode: load ``model_id`` and unload the others, except pinned ones."""
        for other in sorted(self._loaded - self._pinned - {model_id}):
            self.models[other].unload()
            self._loaded.discard(other)
        if model_id not in self._loaded:
            self.models[model_id].load(self._device)
            self._loaded.add(model_id)

    def _switch_to_sequential(self, keep: str) -> None:
        """Unload every model except ``keep`` and the pinned ones; continue one at a time."""
        logger.warning("out of GPU memory with all models loaded; switching to sequential loading")
        for model_id in sorted(self._loaded - self._pinned - {keep}):
            self.models[model_id].unload()
            self._loaded.discard(model_id)
        self._loading = "sequential"
        self._cursor = list(self.models).index(keep)

    def _run_sequential(self) -> int:
        """Claim for one model at a time, moving on when it has no work.

        Idle (0) only after a full cycle over all models found nothing; the next cycle starts
        again from the first model.
        """
        ids = list(self.models)
        for k in range(len(ids)):
            i = (self._cursor + k) % len(ids)
            self._activate(ids[i])
            handled = self._process_claim([ids[i]])
            if handled:
                self._cursor = i
                return handled
        self._cursor = 0
        return 0

    def run_once(self) -> int:
        """Process one claim; return the number of tracks handled (0 = no work)."""
        if self._loading == "sequential":
            return self._run_sequential()
        return self._process_claim(list(self.models))

    def _give_up_on_memory(self, model_id: str) -> None:
        """One model alone does not fit: hand the leases back and stop instead of looping."""
        try:
            self.client.release()
        except Exception as e:
            logger.warning("could not release leases: %s", e)
        info = MODELS.get(model_id)
        need = f" (~{info.memory_gb:g} GB needed)" if info is not None else ""
        raise InsufficientGpuMemoryError(
            f"GPU memory is too small for {model_id}{need}; join with --accel cpu"
        )

    def _process_claim(self, model_ids: list[str]) -> int:
        """Claim tracks for ``model_ids`` and process them; return the number handled."""
        claim = self.client.claim(model_ids, self.settings.claim_size)
        if not claim.items:
            return 0
        started = time.monotonic()
        plan = self._known_plan(claim.items)
        jobs = [self._prepare(item, plan) for item in claim.items]
        results: list[ResultIn] = []
        failures: list[FailureIn] = []
        done = 0
        fatal: _OutOfMemoryError | None = None
        with (
            tempfile.TemporaryDirectory(prefix="lainbow-engine-") as tmp_name,
            ThreadPoolExecutor(self._decoders) as pool,
        ):
            futures = [pool.submit(self._load, job, Path(tmp_name)) for job in jobs]
            for job, future in zip(jobs, futures, strict=True):
                try:
                    decoded: _Decoded | Exception = future.result()
                except Exception as e:
                    decoded = e
                try:
                    r, f = self._analyze(job, decoded)
                except _OutOfMemoryError as oom:
                    # Not a failure: the unfinished tracks' leases expire and they are retried.
                    if self._loading == "sequential":
                        fatal = oom
                    else:
                        self._switch_to_sequential(oom.model)
                    break
                results += r
                failures += f
                done += 1

        self.client.post_results(results, failures)
        if fatal is not None:
            self._give_up_on_memory(fatal.model)
        elapsed = time.monotonic() - started
        seconds = sum(job.item.duration_sec for job in jobs[:done])
        rate = f", {done * 60.0 / elapsed:.1f} tracks/min" if elapsed >= 1.0 else ""
        logger.info("claim done: %d tracks, %.0f s of audio%s", done, seconds, rate)
        return done or 1

    def run_forever(self, stop: threading.Event) -> None:
        """Loop on ``run_once`` until ``stop`` is set, then release the leases."""
        backoff = BACKOFF_START_SEC
        warned = False
        try:
            while not stop.is_set():
                try:
                    handled = self.run_once()
                except ServerUnreachableError as e:
                    if not warned:
                        logger.warning("%s; retrying", e)
                        warned = True
                    stop.wait(backoff)
                    backoff = min(backoff * 2, BACKOFF_MAX_SEC)
                    continue
                warned = False
                backoff = BACKOFF_START_SEC
                if handled == 0:
                    stop.wait(self.settings.idle_sec)
        finally:
            try:
                self.client.release()
            except Exception as e:
                logger.warning("could not release leases: %s", e)
