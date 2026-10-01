# Week 2 Results: Minimum Harness

Oct 1, 2026

Built the harness's actual contract from [docs/plan.md](plan.md) section 1
-- a candidate spec in, a result record out -- replacing the Week 1 one-off
scripts with a single orchestrator, a queryable DuckDB store, and an
artifact cache. Scope matched the README's Week 2 checklist exactly: no
Optuna, no lm-evaluation-harness, no long-context suite, no per-tensor
mixed precision.

## What got built

- [harness/spec.py](../harness/spec.py) -- `CandidateSpec`: a frozen
  dataclass with two hashes, not one. `artifact_hash` (base model + quant +
  tensor overrides/imatrix) keys the artifact cache; `spec_hash` (everything,
  including KV cache type and context length) keys the `specs`/`runs` tables.
- [harness/llama.py](../harness/llama.py) -- shared subprocess/logging layer
  (quantize, bench, KL) pulled out of `kld.py`/`noise.py`, which are now thin
  CLI wrappers over it. Both scripts' CLIs are unchanged.
- [harness/db.py](../harness/db.py) -- DuckDB schema (9 tables: specs,
  artifacts, environments, workloads, runs, speed, resources, quality,
  failures), idempotent inserts, `python -m harness.db --rebuild` to replay
  every `results/runs/*/record.json` from scratch.
- [harness/run.py](../harness/run.py) -- the orchestrator:
  `python -m harness.run specs/foo.json --repeats 5 --kld-ref ... --kld-text ...`
  builds (or reuses) the artifact, measures pp/tg speed as separate
  llama-bench invocations with the KV cache pre-filled to the spec's target
  context length (`-d`/`--n-depth` -- llama-bench has no plain `-c` flag),
  runs KL divergence with the spec's own KV flags applied, and writes both a
  raw JSON record and DuckDB rows.
- [harness/gpu.py](../harness/gpu.py) -- a `Sampler`: background thread
  polling NVML at 10Hz, reporting peak VRAM delta over a pre-run baseline,
  mean power, and energy from NVML's running counter rather than averaging
  instantaneous power draw.

## Two bugs caught before they shipped

A design review pass (run in parallel with planning, before any code was
written) caught two correctness issues in the original draft:

1. **The artifact cache was going to be keyed by the full spec hash.**
   Since the spec also includes KV cache type and context length -- neither
   of which changes the `.gguf` file `llama-quantize` produces -- every KV
   variant of a quant would have triggered its own rebuild of the same
   weights file. That would have quietly multiplied build time on exactly
   the sweep (weights vs. KV cache) this project exists to run. Fixed by
   splitting into `artifact_hash` and `spec_hash` (see `spec.py` above).
2. **KL divergence wasn't going to see the candidate's KV-cache flags.**
   Without `-ctk`/`-ctv`/`-fa` on the `llama-perplexity` call, every KV
   variant of a quant would get an identical KL score, silently erasing half
   of what the harness is supposed to measure. Fixed in `llama.kv_flags()`,
   applied to both the bench and KL calls.

Both are now recorded as implementation notes in `docs/plan.md` section 1,
not just fixed in code.

## Verification

Two real specs, both `Q4_K_M` at ctx 8192, differing only in KV cache
(f16/f16 vs. q8_0/q8_0 + flash attention), run end to end against the
RTX 3080 Ti:

| | KV f16/f16 | KV q8_0/q8_0 |
| --- | --- | --- |
| Artifact | `26a11f88...gguf` (shared, built once) | same file |
| pp speed | 2426 t/s | 3825 t/s |
| tg speed | 143.9 t/s | 166.1 t/s |
| Peak VRAM delta (measured) | 2.98 GB | 2.68 GB |
| Predicted total (memory model) | 3.21 GB | 3.07 GB |
| KL mean vs. Q8_0 | 0.0467 | 0.0465 |
| Top-1 agreement | 90.5% | 90.6% |

Confirmed:
- **One artifact, two specs** -- `models/cache/` held exactly one `.gguf`
  after both specs ran; the second spec's build step was a cache hit.
- **Real, different signal per KV config** -- q8_0 KV cache is both faster
  and lighter on VRAM than f16, at essentially the same quality. That's a
  live instance of the exact tradeoff this project is built to find.
- All 21 unit tests pass (`tests/test_harness.py`, `tests/test_db.py`,
  plus the existing `tests/test_memory_stats.py`), covering spec hash
  determinism, the KV-cache-needs-flash-attention validation, resource
  sampling math, and DB insert idempotency -- all without needing a GPU.

One incidental fix: added `pytest.ini` (`testpaths = tests`) since bare
`pytest` was otherwise collecting `third_party/llama.cpp`'s own vendored
test suites (and failing on their missing dependencies) once the repo was
cloned in.

## What's next

Week 3 per `docs/plan.md` section 10: wire `lm-evaluation-harness` to
`llama-server` for a couple of held-out tasks (GSM8K, one code task), build
the long-context suite (needle-in-a-haystack and multi-hop retrieval at
8k/16k/32k), and create search/dev/held-out text splits from separate
sources with the held-out set locked.
