"""Decide whether the engine keeps all models loaded at once or one at a time."""

import logging
from collections.abc import Collection, Iterable
from pathlib import Path
from typing import Literal

from lainbow.core.models import MODELS

logger = logging.getLogger(__name__)

Loading = Literal["together", "sequential"]

# Share of the free device memory the loaded models may use in ``auto`` mode.
FREE_MEMORY_SHARE = 0.9

# A cgroup memory limit at or above this means "no limit" (cgroup v1 reports a huge number).
_UNLIMITED_BYTES = 2**60


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
    ids = list(models)
    needed = sum(MODELS[m].memory_gb for m in ids)
    if device.startswith("cuda"):
        import torch

        free_gb: float | None = torch.cuda.mem_get_info()[0] / 2**30
        where = "free"
    else:
        free_gb = available_memory_gb()
        where = "available in this container"
    if free_gb is None:
        logger.info("model loading: together (device %s, memory unknown)", device)
        return "together"
    budget = FREE_MEMORY_SHARE * free_gb
    if needed <= budget:
        logger.info("model loading: together (needs ~%.1f GB, %.1f GB %s)", needed, free_gb, where)
        return "together"
    unpinned = [MODELS[m].memory_gb for m in ids if m not in pinned]
    seq_needed = sum(MODELS[m].memory_gb for m in ids if m in pinned) + max(unpinned, default=0.0)
    logger.info(
        "model loading: sequential (needs ~%.1f GB together, ~%.1f GB sequentially, %.1f GB %s)",
        needed,
        seq_needed,
        free_gb,
        where,
    )
    if seq_needed > budget:
        if device.startswith("cuda"):
            logger.warning("even sequential loading may not fit in GPU memory")
        else:
            logger.warning(
                "even sequential loading may not fit in memory; "
                "give the container more memory (Docker Desktop: Settings > Resources > Memory)"
            )
    return "sequential"


def _read_number(path: Path) -> int | None:
    """Return the integer in a file, or None if it is missing, unreadable or not a number."""
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _cgroup_free_bytes(cgroup: Path) -> int | None:
    """Return the memory left under the cgroup limit (v2, then v1), or None if there is none."""
    candidates = (
        (cgroup / "memory.max", cgroup / "memory.current"),
        (cgroup / "memory" / "memory.limit_in_bytes", cgroup / "memory" / "memory.usage_in_bytes"),
    )
    for limit_file, usage_file in candidates:
        limit = _read_number(limit_file)  # "max" (v2, unlimited) is not a number
        if limit is None or limit >= _UNLIMITED_BYTES:
            continue
        return max(limit - (_read_number(usage_file) or 0), 0)
    return None


def _meminfo_available_bytes(meminfo: Path) -> int | None:
    """Return ``MemAvailable`` from a meminfo file in bytes, or None."""
    try:
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def available_memory_gb(
    cgroup: Path = Path("/sys/fs/cgroup"), meminfo: Path = Path("/proc/meminfo")
) -> float | None:
    """Return the RAM this process can still use in GB, or None if it cannot be determined.

    It is the smaller of the cgroup limit minus its usage (when a limit is set) and the
    host's ``MemAvailable``.
    """
    values = [
        v for v in (_cgroup_free_bytes(cgroup), _meminfo_available_bytes(meminfo)) if v is not None
    ]
    return min(values) / 2**30 if values else None
