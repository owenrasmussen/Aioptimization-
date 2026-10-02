"""The "speed of light" check: a measured speed that exceeds what the
hardware can physically do is a bug or a cheat, per docs/plan.md section 4.

Decode (token generation) is memory-bandwidth-bound on a single GPU at batch
size 1: every new token requires reading essentially all of the weights plus
the whole KV cache once. Prompt processing is compute-bound instead (it
batches many tokens through the same weights), so the SAME roofline formula
does not apply to it -- a separate, much looser FLOP-based bound is used for
pp instead, catching only gross errors (4x+), not subtle ones.
"""
from __future__ import annotations

from pathlib import Path

from harness import memory

# NVIDIA's published spec, used as a cross-check (and fallback when NVML's
# bus-width query isn't available). GB/s.
KNOWN_PEAK_BW_GBPS = {"NVIDIA GeForce RTX 3080 Ti": 912.4}
# Dense INT8 Tensor Core TOPS, used only for the loose prefill compute bound.
KNOWN_PEAK_DENSE_TOPS = {"NVIDIA GeForce RTX 3080 Ti": 272.8}


def peak_bandwidth_gbps(index: int = 0, override: float | None = None) -> dict:
    """{"gbps", "source": "override"|"nvml"|"table", "nvml_gbps", "table_gbps"}.

    NVML value = max_mem_clock_mhz * 2 (DDR) * bus_width_bits / 8 / 1000 --
    verified live against this card: 9501 MHz * 2 * 384 / 8 / 1000 = 912.1
    GB/s, matching NVIDIA's published 912.4 GB/s spec almost exactly. Uses
    the MAX memory clock (nvmlDeviceGetMaxClockInfo), not the current one:
    the current clock is ~405MHz at idle, which would make the bound ~20x
    too tight and flag every honest run.
    """
    if override is not None:
        return {"gbps": override, "source": "override", "nvml_gbps": None, "table_gbps": None}

    import pynvml

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(index)
    name = pynvml.nvmlDeviceGetName(h)
    name = name.decode() if isinstance(name, bytes) else name

    nvml_gbps = None
    try:
        mem_clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_MEM)
        bus_width = pynvml.nvmlDeviceGetMemoryBusWidth(h)
        nvml_gbps = mem_clock_mhz * 2 * bus_width / 8 / 1000
    except Exception:
        pass  # nvmlDeviceGetMemoryBusWidth isn't on every pynvml build; fall back to the table

    table_gbps = KNOWN_PEAK_BW_GBPS.get(name)

    if nvml_gbps and table_gbps and abs(nvml_gbps - table_gbps) / table_gbps > 0.03:
        return {"gbps": table_gbps, "source": "table", "nvml_gbps": nvml_gbps, "table_gbps": table_gbps}
    if nvml_gbps:
        return {"gbps": nvml_gbps, "source": "nvml", "nvml_gbps": nvml_gbps, "table_gbps": table_gbps}
    if table_gbps:
        return {"gbps": table_gbps, "source": "table", "nvml_gbps": None, "table_gbps": table_gbps}
    return {"gbps": None, "source": "unknown", "nvml_gbps": None, "table_gbps": None}


def decode_bytes_per_token(gguf_path: Path, depth: int, kv_type_k: str, kv_type_v: str) -> dict | None:
    """{"weight_bytes", "kv_bytes", "total"}, or None for MoE models (the
    dense roofline formula doesn't apply to active-parameter MoE decode and
    would falsely flag every MoE run).

    weight_bytes excludes token_embd.weight: decode only looks up a single
    row of it per token (negligible), unlike output.weight (the LM head),
    which is read in full every token regardless of whether the two are
    tied. Verified live on Qwen2.5-3B's GGUF: token_embd.weight and
    output.weight are stored as two distinct, differently-quantized
    tensors (8.3% and 12.2% of total weight bytes respectively) -- not
    tied in this export, and excluding token_embd tightens the bound by
    that 8.3%, which is not negligible.
    """
    from gguf import GGUFReader

    r = GGUFReader(str(gguf_path))
    arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
    if f"{arch}.expert_count" in r.fields:
        return None

    weight_bytes = sum(t.n_bytes for t in r.tensors if t.name != "token_embd.weight")
    shape = memory.shape_from_gguf(str(gguf_path))
    kv_bytes = memory.kv_cache_bytes(shape, depth, kv_type_k, kv_type_v)
    return {"weight_bytes": weight_bytes, "kv_bytes": kv_bytes, "total": weight_bytes + kv_bytes}


def check_decode(tg_tps: float, bytes_per_token: float, peak_gbps: float, margin: float = 1.0) -> dict:
    """A hard physical ceiling -- margin stays at 1.0 because every
    approximation in decode_bytes_per_token/peak_bandwidth_gbps already
    makes the bound looser, never tighter, so timing jitter can't
    legitimately push a real measurement past it."""
    bound_tps = (peak_gbps * 1e9 / bytes_per_token) if (peak_gbps and bytes_per_token) else None
    exceeded = bound_tps is not None and tg_tps > margin * bound_tps
    utilization = (tg_tps * bytes_per_token / (peak_gbps * 1e9)) if (peak_gbps and bytes_per_token) else None
    return {"name": "physics_decode", "passed": not exceeded, "value": tg_tps, "bound": bound_tps,
            "details": {"utilization": utilization, "bytes_per_token": bytes_per_token}}


def check_prefill(pp_tps: float, n_params: int, peak_tops: float, margin: float = 1.0) -> dict:
    """A deliberately loose compute bound: bound = peak_tops / (2 * n_params)
    (one multiply-add per parameter per token, ignoring attention FLOPs,
    which is already generous). Prompt processing is compute-bound, not
    memory-bound -- the decode formula does NOT apply here. This catches
    gross errors (e.g. counting depth tokens in the throughput denominator,
    which can inflate pp by 10x+) but not subtle ones; said plainly rather
    than oversold."""
    bound_tps = (peak_tops * 1e12 / (2 * n_params)) if (peak_tops and n_params) else None
    exceeded = bound_tps is not None and pp_tps > margin * bound_tps
    return {"name": "physics_prefill", "passed": not exceeded, "value": pp_tps, "bound": bound_tps,
            "details": {}}
