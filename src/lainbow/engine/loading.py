"""Decide whether the engine keeps all models loaded at once or one at a time."""

import logging
from collections.abc import Collection, Iterable
from typing import Literal

from lainbow.core.models import MODELS

logger = logging.getLogger(__name__)

Loading = Literal["together", "sequential"]

# Share of the free device memory the loaded models may use in ``auto`` mode.
FREE_MEMORY_SHARE = 0.9


def choose_loading(
    setting: Literal["auto", "together", "sequential"],
    device: str,
    models: Iterable[str],
    pinned: Collection[str] = (),
) -> Loading:
    """Resolve ``LAINBOW_MODEL_LOADING`` for the given device and model ids.

    Pinned models stay loaded in sequential mode, so they count towards its memory need.
    """
    if pinned:
        logger.info("pinned models (never unloaded): %s", ", ".join(sorted(pinned)))
    if setting != "auto":
        logger.info("model loading: %s (set explicitly)", setting)
        return setting
    if not device.startswith("cuda"):
        logger.info("model loading: together (device %s)", device)
        return "together"

    import torch

    ids = list(models)
    needed = sum(MODELS[m].memory_gb for m in ids)
    free_gb = torch.cuda.mem_get_info()[0] / 2**30
    budget = FREE_MEMORY_SHARE * free_gb
    if needed <= budget:
        logger.info("model loading: together (needs ~%.1f GB, %.1f GB free)", needed, free_gb)
        return "together"
    unpinned = [MODELS[m].memory_gb for m in ids if m not in pinned]
    seq_needed = sum(MODELS[m].memory_gb for m in ids if m in pinned) + max(unpinned, default=0.0)
    logger.info(
        "model loading: sequential (needs ~%.1f GB together, ~%.1f GB sequentially, %.1f GB free)",
        needed,
        seq_needed,
        free_gb,
    )
    if seq_needed > budget:
        logger.warning("even sequential loading may not fit in GPU memory")
    return "sequential"
