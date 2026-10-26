"""Registry of model factories; a factory that raises ImportError is unavailable."""

from collections.abc import Callable
from pathlib import Path

from lainbow.engine.models.base import AudioModel


def _muq(models_dir: Path) -> AudioModel:
    from lainbow.engine.models.muq import MuQModel

    return MuQModel(models_dir)


def _muq_mulan(models_dir: Path) -> AudioModel:
    from lainbow.engine.models.muq_mulan import MuQMuLanModel

    return MuQMuLanModel(models_dir)


def _clap(models_dir: Path) -> AudioModel:
    from lainbow.engine.models.clap import ClapModel

    return ClapModel(models_dir)


def _mert(models_dir: Path) -> AudioModel:
    from lainbow.engine.models.mert import MertModel

    return MertModel(models_dir)


FACTORIES: dict[str, Callable[[Path], AudioModel]] = {
    "muq": _muq,
    "muq_mulan": _muq_mulan,
    "clap": _clap,
    "mert": _mert,
}


def available(models_dir: Path) -> set[str]:
    """Return the ids whose factory can be built in this environment."""
    ok: set[str] = set()
    for model_id, factory in FACTORIES.items():
        try:
            factory(models_dir)
        except ImportError:
            continue
        ok.add(model_id)
    return ok
