"""Registry of model factories; a factory that raises ImportError is unavailable."""

from collections.abc import Callable

from lainbow.engine.models.base import AudioModel

FACTORIES: dict[str, Callable[[], AudioModel]] = {}


def available() -> set[str]:
    """Return the ids whose factory can be built in this environment."""
    ok: set[str] = set()
    for model_id, factory in FACTORIES.items():
        try:
            factory()
        except ImportError:
            continue
        ok.add(model_id)
    return ok
