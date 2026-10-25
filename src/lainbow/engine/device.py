"""Compute device selection and the GPU preflight."""

import logging
import re
from typing import Literal

logger = logging.getLogger(__name__)

UPDATE_DRIVER_HINT = "update the NVIDIA driver (570 or newer)"


class UnsupportedGpuError(RuntimeError):
    """The GPU cannot run this engine build (unsupported architecture or failing kernels)."""


class DriverTooOldError(RuntimeError):
    """The NVIDIA driver supports an older CUDA than this torch build needs."""


def _driver_cuda_version() -> int:
    """Return the CUDA version the driver supports as an int like 12080 (0 = no driver)."""
    import torch

    try:
        return int(torch._C._cuda_getDriverVersion())  # type: ignore[attr-defined]
    except Exception:
        return 0


def _built_cuda_version() -> int:
    """Return the CUDA version torch was built for as an int like 12080 (0 = not a CUDA build)."""
    import torch

    built = getattr(torch.version, "cuda", None)
    if not built:
        return 0
    major, _, minor = str(built).partition(".")
    return int(major) * 1000 + int(minor or 0) * 10


def _format_cuda(version: int) -> str:
    return f"{version // 1000}.{version % 1000 // 10}"


def _old_driver_message() -> str | None:
    """Return the message for a driver that is present but older than torch's CUDA build."""
    driver, built = _driver_cuda_version(), _built_cuda_version()
    if 0 < driver < built:
        return (
            f"the NVIDIA driver supports CUDA {_format_cuda(driver)} but this engine needs "
            f"CUDA {_format_cuda(built)} or newer; {UPDATE_DRIVER_HINT}"
        )
    return None


def select_device(accel: Literal["auto", "cpu", "cuda", "rocm"]) -> str:
    """Return the torch device string for the ``LAINBOW_ACCEL`` setting."""
    import torch

    has_gpu = bool(torch.cuda.is_available())  # ROCm builds also report through torch.cuda
    if accel != "cpu" and not has_gpu:
        old_driver = _old_driver_message()
        if old_driver is not None:
            if accel == "auto":
                logger.warning("%s; running on the CPU", old_driver)
            else:
                raise DriverTooOldError(f"LAINBOW_ACCEL={accel} but {old_driver}")
        elif accel in ("cuda", "rocm"):
            hint = (
                "run the container with `docker run --gpus all`"
                if accel == "cuda"
                else "pass /dev/kfd and /dev/dri to the container"
            )
            raise RuntimeError(f"LAINBOW_ACCEL={accel} but no GPU is visible to torch; {hint}")
    device = "cuda" if accel != "cpu" and has_gpu else "cpu"
    logger.info("torch %s, device %s", torch.__version__, device)
    return device


def _arch_supported(major: int, minor: int, arch_list: list[str]) -> bool:
    """Whether a binary for ``sm_<major><minor>`` or a lower ``sm_`` of the same major exists."""
    for arch in arch_list:
        m = re.fullmatch(r"sm_(\d+)[a-z]*", arch)
        if m is None or len(m.group(1)) < 2:
            continue
        digits = m.group(1)
        if int(digits[:-1]) == major and int(digits[-1]) <= minor:
            return True
    return False


def _self_test(device: str) -> None:
    """Run a small matmul and conv1d on ``device`` and wait for them to finish."""
    import torch

    x = torch.randn(256, 256, device=device)
    (x @ x).sum().item()
    signal = torch.randn(1, 1, 64, device=device)
    kernel = torch.randn(1, 1, 3, device=device)
    torch.nn.functional.conv1d(signal, kernel)
    torch.cuda.synchronize()


def preflight(device: str) -> None:
    """Check that the chosen device can really run kernels; call before loading any model."""
    if device != "cuda":
        logger.info("preflight: cpu")
        return
    import torch

    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability()
    arch_list = list(torch.cuda.get_arch_list())
    if not _arch_supported(major, minor, arch_list):
        raise UnsupportedGpuError(
            f"{props.name} (compute capability {major}.{minor}) is not supported by this engine "
            f"build (supports: {', '.join(arch_list)}); join with --accel cpu"
        )
    try:
        _self_test(device)
    except Exception as e:
        raise UnsupportedGpuError(f"GPU self-test failed: {e}") from e
    driver = _driver_cuda_version()
    logger.info(
        "preflight: GPU %s, compute capability %d.%d, %.1f GiB, driver CUDA %s",
        props.name,
        major,
        minor,
        props.total_memory / 2**30,
        _format_cuda(driver) if driver else "unknown",
    )
