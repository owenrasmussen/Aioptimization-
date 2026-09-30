"""Analytical VRAM estimate: weights + KV cache + measured overhead.

KV bytes = layers_with_kv * kv_heads * (key_dim * k_bytes + value_dim * v_bytes) * ctx

Validate against measured peak VRAM before trusting it. Hybrid models (some
layers with linear attention / no KV) need --kv-layers set to the real count.

Example:
  python -m harness.memory --gguf models/qwen-9b-Q4_K_M.gguf --ctx 32768 --k q8_0 --v q8_0
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

# Bytes per element, including block scales (q8_0: 34 B / 32 elems, q4_0: 18 B / 32).
KV_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q5_1": 24 / 32,
            "q5_0": 22 / 32, "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32}


@dataclass
class ModelShape:
    kv_layers: int
    kv_heads: int
    key_dim: int
    value_dim: int


def kv_cache_bytes(shape: ModelShape, ctx: int, k_type: str = "f16", v_type: str = "f16") -> float:
    per_token = shape.kv_layers * shape.kv_heads * (
        shape.key_dim * KV_BYTES[k_type] + shape.value_dim * KV_BYTES[v_type])
    return per_token * ctx


def total_bytes(weight_bytes: int, shape: ModelShape, ctx: int, k_type: str, v_type: str,
                overhead_bytes: int) -> float:
    return weight_bytes + kv_cache_bytes(shape, ctx, k_type, v_type) + overhead_bytes


def shape_from_gguf(path: str) -> ModelShape:
    from gguf import GGUFReader  # pip install gguf

    r = GGUFReader(path)

    def field(key):
        f = r.fields.get(key)
        return None if f is None else int(f.parts[f.data[0]][0])

    arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
    n_head = field(f"{arch}.attention.head_count")
    head_dim = field(f"{arch}.embedding_length") // n_head
    return ModelShape(
        kv_layers=field(f"{arch}.block_count"),
        kv_heads=field(f"{arch}.attention.head_count_kv") or n_head,
        key_dim=field(f"{arch}.attention.key_length") or head_dim,
        value_dim=field(f"{arch}.attention.value_length") or head_dim,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", help="read shape and weight size from this file")
    ap.add_argument("--weights-gb", type=float, help="weight size if no --gguf")
    ap.add_argument("--kv-layers", type=int)
    ap.add_argument("--kv-heads", type=int)
    ap.add_argument("--head-dim", type=int)
    ap.add_argument("--ctx", type=int, nargs="+", default=[8192, 16384, 32768])
    ap.add_argument("--k", default="f16", choices=KV_BYTES)
    ap.add_argument("--v", default="f16", choices=KV_BYTES)
    ap.add_argument("--overhead-gb", type=float, default=0.8,
                    help="compute buffers + CUDA context; measure once and set it")
    ap.add_argument("--budget-gb", type=float, default=11.0)
    args = ap.parse_args()

    if args.gguf:
        shape = shape_from_gguf(args.gguf)
        weights = Path(args.gguf).stat().st_size
    else:
        shape = ModelShape(args.kv_layers, args.kv_heads, args.head_dim, args.head_dim)
        weights = int(args.weights_gb * 1e9)
    if args.kv_layers:
        shape.kv_layers = args.kv_layers

    gb = 1e9
    print(f"shape {shape}  weights {weights / gb:.2f} GB  K={args.k} V={args.v}")
    for ctx in args.ctx:
        kv = kv_cache_bytes(shape, ctx, args.k, args.v)
        tot = weights + kv + args.overhead_gb * gb
        fits = "fits" if tot <= args.budget_gb * gb else "OVER"
        print(f"ctx {ctx:>6}: KV {kv / gb:5.2f} GB  total {tot / gb:5.2f} GB  [{fits} {args.budget_gb} GB]")


if __name__ == "__main__":
    main()
