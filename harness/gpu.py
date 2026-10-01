"""NVML helpers: environment stamp, wait-for-cooldown, and resource sampling."""
from __future__ import annotations

import statistics
import subprocess
import threading
import time
from dataclasses import dataclass


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


def stable_env(env: dict, commit: str) -> dict:
    """The subset of environment() that doesn't change between runs on the same
    machine. Clocks are excluded (they drift at idle/under load) so env_id
    doesn't churn on every run; actual clocks at run start belong on the run row."""
    return {
        "gpu": env["gpu"], "vram_mb": env["vram_mb"], "driver": env["driver"],
        "cuda": env["cuda"], "engine_commit": commit, "power_limit_w": env["power_limit_w"],
    }


def _s(v):
    return v.decode() if isinstance(v, bytes) else v


@dataclass
class Sample:
    t: float  # time.monotonic()
    vram_used_mb: float
    power_w: float
    energy_mj: int  # NVML's running counter, millijoules since driver load
    temp_c: int
    sm_clock_mhz: int


class Sampler:
    """Samples NVML at ~10Hz in a background thread for the duration of a `with` block.

    Note: on Windows (WDDM), nvmlDeviceGetMemoryInfo().used reports total device
    VRAM in use by every process, not just the one being measured -- there's no
    per-process VRAM query under WDDM. baseline_vram_mb is captured on __enter__
    so callers can report a delta, which is what actually tracks the measured
    process's own usage.
    """

    def __init__(self, index: int = 0, hz: float = 10.0):
        self.index = index
        self.interval = 1.0 / hz
        self.samples: list[Sample] = []
        self.baseline_vram_mb: float = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "Sampler":
        nv = _nvml()
        h = nv.nvmlDeviceGetHandleByIndex(self.index)
        self.baseline_vram_mb = nv.nvmlDeviceGetMemoryInfo(h).used / 2**20
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        nv = _nvml()
        h = nv.nvmlDeviceGetHandleByIndex(self.index)
        while not self._stop.is_set():
            self.samples.append(Sample(
                t=time.monotonic(),
                vram_used_mb=nv.nvmlDeviceGetMemoryInfo(h).used / 2**20,
                power_w=nv.nvmlDeviceGetPowerUsage(h) / 1000,
                energy_mj=nv.nvmlDeviceGetTotalEnergyConsumption(h),
                temp_c=nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
                sm_clock_mhz=nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
            ))
            self._stop.wait(self.interval)

    def summary(self, n_tokens: int | None = None) -> dict:
        return summarize_samples(self.samples, self.baseline_vram_mb, n_tokens=n_tokens)


def summarize_samples(samples: list[Sample], baseline_vram_mb: float = 0.0,
                       n_tokens: int | None = None) -> dict:
    """Pure function, unit-testable without a GPU. energy_j is the NVML energy
    counter's delta across the sampling window (first to last sample) -- this
    includes whatever process startup/teardown fell inside the `with` block,
    so treat joules_per_token as approximate, not a tight measurement."""
    if not samples:
        return {"peak_vram_mb": None, "peak_vram_delta_mb": None, "mean_power_w": None,
                "energy_j": None, "joules_per_token": None, "max_temp_c": None, "n_samples": 0}
    peak_vram = max(s.vram_used_mb for s in samples)
    mean_power = statistics.fmean(s.power_w for s in samples)
    max_temp = max(s.temp_c for s in samples)
    energy_j = (samples[-1].energy_mj - samples[0].energy_mj) / 1000 if len(samples) > 1 else None
    jpt = energy_j / n_tokens if (energy_j is not None and n_tokens) else None
    return {
        "peak_vram_mb": peak_vram,
        "peak_vram_delta_mb": peak_vram - baseline_vram_mb,
        "mean_power_w": mean_power,
        "energy_j": energy_j,
        "joules_per_token": jpt,
        "max_temp_c": max_temp,
        "n_samples": len(samples),
    }
