# Week 5 Results: Sweep on the Small Model

Oct 5, 2026

Per [docs/plan.md](plan.md) section 10's Week 5 checklist: enumerate every
weights+KV config that fits the budget on the small (3B) model, run the
grid with multi-fidelity pruning, and produce the first Pareto plots plus a
check of whether short-context KL predicts long-context quality.

## Prediction, written before running anything

Short-context KL should rank configs almost entirely by **weight quant**,
and barely separate KV types at all -- Week 2-4's KL numbers already showed
f16 vs. q8_0 KV cache landing within noise of each other at short context,
while weight quant differences (Q3_K_M vs Q4_K_M vs Q8_0) were clearly
distinguishable. Long-context KL should widen the gap for the cheapest KV
type (q4_0) specifically, since KV cache quantization error compounds over
more cached tokens in a way short-context KL can't see. My guess: the
Spearman correlation between short and long KL across the full 15-config
grid lands moderately high (rho roughly 0.6-0.85, wide CI given n=15) --
real agreement on weight-quant ranking, with q4_0-KV configs as the
systematic outliers that pull the correlation down from "near 1."

## Scope and deferrals

5 quants (Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0) x 3 matched KV pairs
(f16/f16, q8_0/q8_0, q4_0/q4_0, flash_attn on for all three -- Week 4's
"fair baseline" lesson applied here) x ctx=32768 = **15 configs**. Three
things from `docs/plan.md` section 7's original design are explicitly
deferred, not silently dropped:

- **IQ4_XS and the published mixed-precision recipe** need
  `tensor_overrides`/`imatrix`, which `ensure_artifact` still raises
  `NotImplementedError` on (Level 2, not built yet).
- **Mixed K/V pairs** (e.g. q8_0 K / q4_0 V). Checked
  `third_party/llama.cpp/build/CMakeCache.txt`:
  `GGML_CUDA_FA_ALL_QUANTS:BOOL=OFF`, meaning CUDA flash-attention kernels
  only exist for matched K/V type pairs in this build. A smoke test of a
  mixed pair (q8_0 K / q4_0 V, `-fa on`) did **not** crash and returned a
  plausible-looking 215 tok/s -- which is the more concerning outcome, not
  the safer one: it means there's an unverified fallback path whose actual
  performance characteristics aren't confirmed against a
  `FA_ALL_QUANTS=ON` build. Rather than risk reporting a build artifact as
  a finding (the same class of mistake as Week 4's `flash_attn` confound),
  mixed pairs wait until the build is rebuilt with that flag and the
  fallback is actually characterized.
- **Context length as a sweep axis.** The 11GB budget never binds on this
  3B model -- worst case in the whole grid (Q8_0 weights, f16 KV, 32k ctx)
  totals ~5.6GB (checked via `harness.memory`). A config that fits at 32k
  fits everywhere, so this week sweeps ctx=32768 only (where KV type matters
  most for speed) and measures quality at 8k/16k/32k via the long-context
  suite regardless. The budget filter itself is still exercised by
  `harness.grid.enumerate_grid` and will matter for real in Week 6's 9B
  sweep.

Also: the KL reference model is **f16**, not Q8_0 (Weeks 1-4's default) --
Q8_0 weights are themselves in this week's grid, and scoring them against a
Q8_0 reference would make Q8_0 look artificially perfect (self-comparison),
biasing the whole Pareto front toward it.

## Multi-fidelity design

**Tier 1 (gate, all 15 configs, search split):** cheap short-KL
(`kld 512:20`) and long-KL (`kld 16384:2`) plus the long-context
needle/multihop suite (4 seeds), run on every config -- not just eventual
survivors. This is deliberate: the agreement question this week is partly
about validating the gate itself, and gating the agreement analysis on the
gate's own output would bias it.

**Promotion:** non-dominated sorting on (short-KL mean, predicted VRAM),
both minimized (`harness.grid.nondominated_fronts`) -- not top-K by KL
alone, which would systematically exclude the cheap/low-quality corner of
the tradeoff (e.g. Q3_K_M + q4_0 KV, exactly the kind of point a real
memory-constrained deployment might need). The baseline spec
(`specs/qwen3b-q4km-kvf16-fa.json`) is always promoted, first.

**Tier 2 (full suite, survivors only, dev split):** the full 10-round
interleaved bench at depth 32k, both KL fidelities again (now on dev, to
avoid the winner's-curse of gating and reporting on the same split), the
long-context suite, and GSM8K/HumanEval.

## Results

`[pending -- live run in progress]`

## What's next

Week 6 per `docs/plan.md` section 10: repeat on the Qwen 9B-class model and
a second 7B-8B family, where the 11GB budget will actually bind and the
mixed-K/V-pair question can be revisited once the build is updated.
