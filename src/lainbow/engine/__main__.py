"""Runnable entry point for the Lainbow engine."""

import contextlib
import logging
import os
import signal
import sys
import threading

from lainbow.core.models import MODELS, Purpose
from lainbow.core.settings import EngineSettings, start
from lainbow.engine import text_api
from lainbow.engine.client import (
    EngineOutdatedError,
    EngineUnauthorizedError,
    ServerClient,
    ServerUnreachableError,
)
from lainbow.engine.device import (
    DriverTooOldError,
    UnsupportedGpuError,
    preflight,
    select_device,
)
from lainbow.engine.loading import choose_loading
from lainbow.engine.models import registry
from lainbow.engine.models.base import AudioModel
from lainbow.engine.worker import InsufficientGpuMemoryError, Worker

logger = logging.getLogger(__name__)

# This engine cannot work as configured (GPU, driver, or a version that does not match the server).
EXIT_CONFIGURATION = 3
CONFIGURATION_HOLD_SEC: float = 600.0


def run(
    settings: EngineSettings,
    *,
    stop: threading.Event | None = None,
    hold_sec: float | None = None,
) -> int:
    """Run the engine until SIGTERM or SIGINT; return the exit code."""
    if not registry.FACTORIES:
        logger.error("no analysis models are available in this engine build")
        return 1

    if stop is None:
        stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(ValueError, AttributeError):
            signal.signal(sig, lambda _s, _f: stop.set())

    def _exit_configuration() -> int:
        # Before returning EXIT_CONFIGURATION, wait on the stop event so the container
        # does not restart in a tight loop when run under `--restart unless-stopped`.
        logger.info(
            "waiting 10 min before exiting so the container does not restart in a loop; "
            "fix the problem above, or stop it with ./lainbow leave"
        )
        duration = hold_sec if hold_sec is not None else CONFIGURATION_HOLD_SEC
        if duration > 0:
            stop.wait(duration)
        return EXIT_CONFIGURATION

    # The one place where the engine sets an environment variable: the MuQ-MuLan text encoder
    # downloads through huggingface_hub, which must cache under models_dir like the other weights.
    os.environ.setdefault("HF_HOME", str(settings.models_dir / "hf"))

    client = ServerClient(str(settings.server_url), settings.token.get_secret_value())
    try:
        plan = client.plan()
        # A restarted engine keeps its id, so leases from before it stopped would only expire later.
        returned = client.release()
    except (EngineUnauthorizedError, ServerUnreachableError) as e:
        logger.error("%s", e)
        return 1
    except EngineOutdatedError as e:
        logger.error("%s", e)
        return _exit_configuration()

    if returned:
        logger.info("returned %d tracks this engine held before it last stopped", returned)

    usable = registry.available(settings.models_dir)
    wanted = {t.model for t in plan.tasks}
    missing = sorted(wanted - usable)
    if missing:
        logger.warning("the plan wants models this engine cannot run: %s", ", ".join(missing))
    models: dict[str, AudioModel] = {}
    try:
        device = select_device(settings.accel) if wanted & usable else "cpu"
        preflight(device)
    except (DriverTooOldError, UnsupportedGpuError) as e:
        logger.error("%s", e)
        return _exit_configuration()
    for model_id in sorted(wanted & usable):
        models[model_id] = registry.FACTORIES[model_id](settings.models_dir)
    if not models:
        logger.error("no analysis models are available in this engine build")
        return 1

    # In sequential mode the worker loads and unloads models itself, one at a time.
    # Text-capable models stay loaded while the text endpoint is on, so queries keep working.
    pinned = (
        frozenset(m for m in models if Purpose.TEXT in MODELS[m].purposes)
        if settings.text_port > 0
        else frozenset()
    )
    loading = choose_loading(settings.model_loading, device, models, pinned)
    for model_id, model in models.items():
        if loading == "together" or model_id in pinned:
            model.load(device)

    if settings.text_port > 0:
        text_api.serve_in_thread(text_api.create_app(models), settings.text_port)

    worker = Worker(client, models, settings, plan, device=device, loading=loading, pinned=pinned)
    try:
        worker.run_forever(stop)
    except (InsufficientGpuMemoryError, EngineOutdatedError) as e:
        logger.error("%s", e)
        return _exit_configuration()
    return 0


def main() -> None:
    """Load settings, configure logging, and start the Lainbow engine."""
    settings = start(EngineSettings)
    code = run(settings)
    if code:
        sys.exit(code)


if __name__ == "__main__":
    main()
