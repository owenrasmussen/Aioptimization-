"""NVML helpers: environment stamp and wait-for-cooldown."""
from __future__ import annotations

import subprocess
import time


def _nvml():
    import pynvml  # imported lazily so the rest of the package works without a GPU

    pynvml.nvmlInit()
    return pynvml


def temperature(index: int = 0) -> int:
    nv = _nvml()
    h = nv.nvmlDeviceGetHandleByIndex(index)
    return nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)


def wait_for_cool(max_temp_c: int, index: int = 0, timeout_s: float = 600) -> int:
    """Block until the GPU is at or below max_temp_c. Returns the start temperature."""
    deadline = time.monotonic() + timeout_s
    t = temperature(index)
    while t > max_temp_c:
        if time.monotonic() > deadline:
            raise TimeoutError(f"GPU still at {t}C after {timeout_s}s (target {max_temp_c}C)")
        time.sleep(5)
        t = temperature(index)
    return t


def environment(index: int = 0) -> dict:
    """Snapshot of what a run depends on. Goes into every result record."""
    nv = _nvml()
    h = nv.nvmlDeviceGetHandleByIndex(index)
    name = nv.nvmlDeviceGetName(h)
    cuda = nv.nvmlSystemGetCudaDriverVersion()
    return {
        "gpu": name.decode() if isinstance(name, bytes) else name,
        "vram_mb": nv.nvmlDeviceGetMemoryInfo(h).total // 2**20,
        "driver": _s(nv.nvmlSystemGetDriverVersion()),
        "cuda": f"{cuda // 1000}.{(cuda % 1000) // 10}",
        "sm_clock_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
        "mem_clock_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
        "power_limit_w": nv.nvmlDeviceGetPowerManagementLimit(h) / 1000,
    }


def llamacpp_commit(repo_dir: str) -> str:
    return subprocess.run(
        ["git", "-C", repo_dir, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _s(v):
    return v.decode() if isinstance(v, bytes) else v
