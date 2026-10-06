"""Shared llama.cpp subprocess helpers: binary resolution, logging, build/bench/KL.

kld.py and noise.py are thin CLI wrappers around the functions here, so the
Week 1 one-off scripts and the Week 2 orchestrator (run.py) share one
subprocess/logging path instead of duplicating it.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path

from harness.spec import CandidateSpec

DEFAULT_BIN = "third_party/llama.cpp/build/bin"


class RunFailed(RuntimeError):
    def __init__(self, cmd: list[str], log_path: Path, returncode: int):
        super().__init__(f"{cmd[0]} failed (exit {returncode}), see {log_path}")
        self.cmd = cmd
        self.log_path = log_path
        self.returncode = returncode


def bin_path(name: str, bin_dir: str | None = None) -> str:
    d = bin_dir or os.environ.get("LLAMA_BIN", DEFAULT_BIN)
    exe = Path(d) / name
    for candidate in (exe, exe.with_suffix(".exe")):
        if candidate.exists():
            return str(candidate)
    return str(exe)  # let the OS raise a clear "not found" on exec


def _cuda_bin_dirs() -> list[str]:
    """CUDA's DLL directory, so ggml-cuda.dll's runtime deps (cudart64_*,
    cublas64_*, ...) resolve even in a shell that never sourced the CUDA
    installer's PATH change -- confirmed live: a fresh shell launching these
    binaries directly failed with STATUS_DLL_NOT_FOUND even though the exact
    same binaries had just quantized a model successfully moments earlier in
    a shell that did have the PATH set. CUDA 12.x ships its DLLs in bin/
    directly; 13.x moved them to bin/x64 (same layout change
    scripts/setup_llamacpp.sh already works around at build time) -- check
    both rather than assuming one."""
    roots = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        roots.append(Path(cuda_path))
    base = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
    if base.is_dir():
        roots += sorted(base.glob("v*"), reverse=True)
    out = []
    for root in roots:
        for sub in (root / "bin" / "x64", root / "bin"):
            if sub.is_dir() and str(sub) not in out:
                out.append(str(sub))
    return out


def subprocess_env() -> dict:
    """os.environ with CUDA's DLL directory prepended to PATH -- every
    launch of a llama.cpp binary (llama.py's run() and server.py's
    LlamaServer Popen alike) must go through this, not os.environ directly,
    or it silently works in whichever shell happens to have CUDA on PATH
    and fails everywhere else."""
    env = os.environ.copy()
    extra = _cuda_bin_dirs()
    if extra:
        env["PATH"] = os.pathsep.join(extra) + os.pathsep + env.get("PATH", "")
    return env


def run(cmd: list[str], log_path: Path) -> subprocess.CompletedProcess:
    """Run cmd, always writing a log (command line + stdout + stderr). Raises RunFailed on nonzero exit."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(cmd, capture_output=True, text=True, env=subprocess_env())
    log_path.write_text("$ " + shlex.join(cmd) + "\n" + p.stdout + p.stderr)
    if p.returncode != 0:
        raise RunFailed(cmd, log_path, p.returncode)
    return p


def kv_flags(spec: CandidateSpec) -> list[str]:
    """-ctk/-ctv/-fa flags, same syntax for llama-bench, llama-perplexity and llama-server."""
    fa = "on" if spec.flash_attn else "off"
    return ["-ctk", spec.kv_type_k, "-ctv", spec.kv_type_v, "-fa", fa]


def spec_flags(spec: CandidateSpec) -> list[str]:
    """kv_flags() plus -b/-ub when set -- the one place every llama.cpp tool
    invocation builds its spec-derived flags, so a KV type or batch size can't
    drift between bench/KL/server the way KL alone once missed KV flags.

    spec.engine_flags are deliberately NOT included: append them separately
    at the call site, since their syntax isn't guaranteed valid on every
    binary (llama-bench accepts flags the server doesn't, and vice versa)."""
    flags = kv_flags(spec)
    if spec.batch_size:
        flags += ["-b", str(spec.batch_size)]
    if spec.ubatch_size:
        flags += ["-ub", str(spec.ubatch_size)]
    return flags


def quantize(src: Path, dst: Path, quant: str, log_path: Path, bin_dir: str | None = None) -> float:
    """Quantize src -> dst. Writes to a .tmp file and renames on success, so a
    crashed/killed quantize never leaves a file that looks cached but isn't."""
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    exe = bin_path("llama-quantize", bin_dir)
    start = time.monotonic()
    run([exe, str(src), str(tmp), quant], log_path)
    elapsed = time.monotonic() - start
    os.replace(tmp, dst)
    return elapsed


def bench(model: Path, pp: int, tg: int, reps: int, flags: list[str], log_path: Path,
          depth: int = 0, bin_dir: str | None = None) -> list[dict]:
    """One llama-bench invocation. Pass pp=0 or tg=0 to measure only the other phase.

    `depth` is llama-bench's -d/--n-depth: it pre-fills the KV cache to that many
    tokens before timing pp/tg, which is how llama-bench simulates "speed at a
    given context length" -- there's no plain -c/--ctx-size flag on this tool."""
    exe = bin_path("llama-bench", bin_dir)
    cmd = [exe, "-m", str(model), "-p", str(pp), "-n", str(tg), "-r", str(reps)]
    if depth:
        cmd += ["-d", str(depth)]
    cmd += ["-ngl", "99", "-o", "json", *flags]
    p = run(cmd, log_path)
    return json.loads(p.stdout)


def reference_logits(ref: Path, text: Path, ctx: int, out_path: Path, chunks: int | None = None,
                      bin_dir: str | None = None) -> Path:
    """Save reference logits for KL comparison, if not already saved at out_path.
    Writes to a .tmp path and renames on success (same pattern as quantize()):
    reference files are multi-GB and this is a long-running subprocess, so a
    crash or kill partway through must not leave a partial file sitting at
    out_path looking like a valid, complete cache entry."""
    if out_path.exists():
        return out_path
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    exe = bin_path("llama-perplexity", bin_dir)
    cmd = [exe, "-m", str(ref), "-f", str(text), "-c", str(ctx), "-ngl", "99"]
    if chunks:
        cmd += ["--chunks", str(chunks)]
    cmd += ["--kl-divergence-base", str(tmp)]
    run(cmd, out_path.with_suffix(".ref.log"))
    os.replace(tmp, out_path)
    return out_path


def kl_divergence(model: Path, base: Path, text: Path, ctx: int, flags: list[str], log_path: Path,
                   chunks: int | None = None, bin_dir: str | None = None) -> dict:
    from harness.kld import parse  # local import: avoids a module-load cycle, kld.py also imports this module

    exe = bin_path("llama-perplexity", bin_dir)
    cmd = [exe, "-m", str(model), "-f", str(text), "-c", str(ctx), "-ngl", "99", *flags]
    if chunks:
        cmd += ["--chunks", str(chunks)]
    cmd += ["--kl-divergence-base", str(base), "--kl-divergence"]
    p = run(cmd, log_path)
    return parse(p.stdout + p.stderr)
