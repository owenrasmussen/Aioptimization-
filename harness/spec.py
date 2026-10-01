"""The candidate spec: one JSON file describes exactly what to build and measure.

Two separate hashes matter here, not one:

- `artifact_hash` covers only what changes the .gguf file llama-quantize
  produces (base model + quant + any per-tensor overrides/imatrix). It keys
  the artifact cache, so "Q4_K_M + KV f16" and "Q4_K_M + KV q8_0" share the
  same built weights file instead of each triggering their own rebuild.
- `spec_hash` covers every field, including the KV cache and context
  settings that change what gets *measured* but not what gets *built*. It
  keys the `specs`/`runs` tables.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

KV_TYPES = {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"}
# KV types that don't need flash attention to work (full/half precision, stored non-transposed).
KV_TYPES_NO_FA_REQUIRED = {"f32", "f16", "bf16"}

_DEFAULTS = {
    "kv_type_k": "f16",
    "kv_type_v": "f16",
    "flash_attn": False,
    "ctx": 4096,
    "batch_size": None,
    "ubatch_size": None,
    "engine_flags": (),
    "tensor_overrides": {},
    "imatrix": None,
}


@dataclass(frozen=True)
class CandidateSpec:
    base_model: str
    quant: str
    kv_type_k: str = "f16"
    kv_type_v: str = "f16"
    flash_attn: bool = False
    ctx: int = 4096
    batch_size: int | None = None
    ubatch_size: int | None = None
    engine_flags: tuple[str, ...] = ()
    # Reserved for Level 2 (per-tensor mixed precision); harness/run.py rejects
    # non-empty values for now rather than silently ignoring them.
    tensor_overrides: dict = field(default_factory=dict)
    imatrix: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "quant", self.quant.upper())
        object.__setattr__(self, "kv_type_k", self.kv_type_k.lower())
        object.__setattr__(self, "kv_type_v", self.kv_type_v.lower())
        object.__setattr__(self, "engine_flags", tuple(self.engine_flags))
        if self.kv_type_k not in KV_TYPES:
            raise ValueError(f"unknown kv_type_k: {self.kv_type_k!r}")
        if self.kv_type_v not in KV_TYPES:
            raise ValueError(f"unknown kv_type_v: {self.kv_type_v!r}")
        if self.kv_type_v not in KV_TYPES_NO_FA_REQUIRED and not self.flash_attn:
            raise ValueError(
                f"kv_type_v={self.kv_type_v!r} needs flash_attn=true in llama.cpp"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    def _canonical(self, fields: dict) -> str:
        pruned = {k: v for k, v in fields.items() if k in _DEFAULTS and v == _DEFAULTS[k]}
        kept = {k: v for k, v in fields.items() if k not in pruned}
        return json.dumps(kept, sort_keys=True, separators=(",", ":"))

    @property
    def spec_hash(self) -> str:
        d = self.to_dict()
        d["base_model"] = Path(d["base_model"]).name  # path-independent
        return hashlib.sha256(self._canonical(d).encode()).hexdigest()[:16]

    @property
    def artifact_hash(self) -> str:
        d = {
            "base_model": Path(self.base_model).name,
            "quant": self.quant,
            "tensor_overrides": self.tensor_overrides,
            "imatrix": self.imatrix,
        }
        return hashlib.sha256(self._canonical(d).encode()).hexdigest()[:16]

    @classmethod
    def from_json(cls, path: str | Path) -> "CandidateSpec":
        d = json.loads(Path(path).read_text())
        known = {f for f in _DEFAULTS} | {"base_model", "quant"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown spec field(s): {sorted(unknown)}")
        return cls(**d)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
