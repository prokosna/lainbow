"""Compute device selection."""

import logging
from typing import Literal

logger = logging.getLogger(__name__)


def select_device(accel: Literal["auto", "cpu", "cuda", "rocm"]) -> str:
    """Return the torch device string for the ``LAINBOW_ACCEL`` setting."""
    import torch

    has_gpu = bool(torch.cuda.is_available())  # ROCm builds also report through torch.cuda
    if accel in ("cuda", "rocm") and not has_gpu:
        hint = (
            "run the container with `docker run --gpus all`"
            if accel == "cuda"
            else "pass /dev/kfd and /dev/dri to the container"
        )
        raise RuntimeError(f"LAINBOW_ACCEL={accel} but no GPU is visible to torch; {hint}")
    device = "cuda" if accel != "cpu" and has_gpu else "cpu"

    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        logger.info(
            "torch %s, device cuda (%s, %.1f GiB)",
            torch.__version__,
            props.name,
            props.total_memory / 2**30,
        )
    else:
        logger.info("torch %s, device cpu", torch.__version__)
    return device
