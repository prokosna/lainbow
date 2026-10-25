"""Runnable entry point for the Lainbow engine."""

import logging
import signal
import sys
import threading

from lainbow.core.settings import EngineSettings, start
from lainbow.engine import text_api
from lainbow.engine.client import EngineUnauthorizedError, ServerClient, ServerUnreachableError
from lainbow.engine.models import registry
from lainbow.engine.models.base import AudioModel
from lainbow.engine.worker import Worker

logger = logging.getLogger(__name__)


def _device(accel: str) -> str:
    return "cpu" if accel == "auto" else accel


def run(settings: EngineSettings) -> int:
    """Run the engine until SIGTERM or SIGINT; return the exit code."""
    if not registry.FACTORIES:
        logger.error("no analysis models are available in this engine build")
        return 1

    client = ServerClient(str(settings.server_url), settings.token.get_secret_value())
    try:
        plan = client.plan()
    except (EngineUnauthorizedError, ServerUnreachableError) as e:
        logger.error("%s", e)
        return 1

    usable = registry.available()
    wanted = {t.model for t in plan.tasks}
    missing = sorted(wanted - usable)
    if missing:
        logger.warning("the plan wants models this engine cannot run: %s", ", ".join(missing))
    models: dict[str, AudioModel] = {}
    for model_id in sorted(wanted & usable):
        model = registry.FACTORIES[model_id]()
        model.load(_device(settings.accel))
        models[model_id] = model
    if not models:
        logger.error("no analysis models are available in this engine build")
        return 1

    if settings.text_port > 0:
        text_api.serve_in_thread(text_api.create_app(models), settings.text_port)

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda _s, _f: stop.set())
    Worker(client, models, settings, plan).run_forever(stop)
    return 0


def main() -> None:
    """Load settings, configure logging, and start the Lainbow engine."""
    settings = start(EngineSettings)
    code = run(settings)
    if code:
        sys.exit(code)


if __name__ == "__main__":
    main()
