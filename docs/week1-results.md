# Week 1 Results: Reproduce, Don't Build

Sep 30, 2026

Ran the full Week 1 checklist from [README.md](../README.md): pin and build llama.cpp,
measure the noise floor, check KL divergence across quant levels, and validate the
analytical memory model against a real measurement.

## Setup

- llama.cpp pinned at commit [`f7b384c1e`](../LLAMACPP_COMMIT), built with CUDA
  (`GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=86` for the 3080 Ti).
- Model: Qwen2.5-3B-Instruct, downloaded as F16 GGUF and quantized locally to
  Q8_0, Q6_K, Q4_K_M, Q3_K_M with `llama-quantize`.
- GPU: RTX 3080 Ti, 12GB VRAM, Windows 10, driver 596.36, CUDA 13.4.

One Windows-specific build fix was needed and is committed: the Visual Studio
generator puts binaries in `build/bin/<Config>/` by default, which doesn't match
`harness/*.py`'s `--bin-dir` default of `build/bin`. Fixed in
[scripts/setup_llamacpp.sh](../scripts/setup_llamacpp.sh) by forcing a flat
`CMAKE_RUNTIME_OUTPUT_DIRECTORY`.

## Noise floor

10 repeats of one config (Q4_K_M, unlocked clocks), via `harness/noise.py`:

| Metric | Median | CV | 95% CI |
| --- | --- | --- | --- |
| Prompt processing | 8362 t/s | 4.72% | [8274, 8764] |
| Generation | 224.0 t/s | 4.45% | [218.0, 237.5] |

Temps held at 67-70C across all repeats, no thermal throttling. This is the
noise floor: any later speed difference smaller than ~4.5-4.7% doesn't count.

GPU clock/power locking (`scripts/lock_gpu.sh lock`) requires an elevated
terminal on Windows and wasn't run this session, so the locked-vs-unlocked
comparison the README asks for is still open.

## KL divergence vs Q8_0

Reference: Q8_0. Eval text: ~130k tokens pulled from 10 diverse Wikipedia
articles (not a public benchmark), via `harness/kld.py` at ctx 512.

| Quant | Size | Perplexity | Mean KLD | 99th %ile KLD | Top-1 agreement |
| --- | --- | --- | --- | --- | --- |
| Q6_K | 2.66 GB | 7.54 | 0.0067 | 0.060 | 95.9% |
| Q4_K_M | 2.00 GB | 7.74 | 0.0459 | 0.536 | 90.7% |
| Q3_K_M | 1.64 GB | 8.40 | 0.1442 | 1.519 | 83.8% |

Quality degrades monotonically from Q6_K to Q3_K_M, as expected. If this had
come out flat or non-monotonic, that would have signaled a harness bug rather
than a real finding.

## Memory model validation

`harness/memory.py`'s analytical estimate (weights + KV cache bytes) vs a real
measurement, for Q4_K_M at 32k context with Q8_0 K/V cache and flash attention:

| | Weights | KV cache | Overhead | Total |
| --- | --- | --- | --- | --- |
| Analytical (default `--overhead-gb 0.8`) | 2.10 GB | 0.64 GB | 0.80 GB | 3.55 GB |
| Measured (`llama-server`, idle baseline subtracted) | — | — | — | **2.965 GB** |
| Analytical (calibrated `--overhead-gb 0.22`) | 2.10 GB | 0.64 GB | 0.22 GB | 2.97 GB |

Measured by diffing `nvidia-smi` VRAM usage before (1548 MiB) and after
(4376 MiB) loading `llama-server -c 32768 -ctk q8_0 -ctv q8_0 -fa on`. The
core formula (weights + KV bytes) is accurate to ~20MB once the overhead
constant is calibrated — the script's default of 0.8 GB is a conservative
placeholder that overestimates total VRAM need by over half a GB for a model
this size. Worth recalibrating `--overhead-gb` per model/context rather than
trusting the default.

## Reading

Per the README, two pieces of prior art to read before Week 2:

- **[KV Pareto](https://arxiv.org/abs/2512.01953)** (EACL 2026 Industry Track,
  Gokhale/Das/Patwari/Sirasao/Delaye) — closest prior work to the project's
  first finding. Evaluates Qwen/Llama/Mistral with KV quantization
  (int2/4/8, mixed-precision, multiple granularities) plus prefill chunking
  and 4-bit AWQ weight quantization; finds Pareto-optimal configs with
  68-78% memory reduction at 1-3% accuracy loss, verified on NIAH/GSM8K/MMLU
  up to 128k context. Uses AWQ in research code, not llama.cpp's GGUF stack
  and KV cache types — that gap is still open.
  (The link in `docs/plan.md` §6 originally pointed at a dead
  `preview.aclanthology.org` staging URL; it's now fixed to the arXiv link.)
- **[PrismaQuant](https://pypi.org/project/prismaquant)** — the mixed-precision
  tool writeup `docs/plan.md` flags. Solves per-layer bit allocation as a
  knapsack problem using KL and Fisher probes, gated by held-out KL, already
  shipped as llama.cpp and vLLM files. Read so as not to repeat this as a
  headline result — the project's angle is the weights-vs-KV-cache split, not
  per-layer weight mixing.

## What's left

Week 1 is otherwise complete: pinned commit, noise-floor number, KL table
across three quants, and a memory estimate checked against a real
measurement. Remaining before Week 2 (per `docs/plan.md` §10):

- [ ] Locked-clock noise floor comparison (needs an elevated terminal)
- [ ] Read KV Pareto and PrismaQuant
