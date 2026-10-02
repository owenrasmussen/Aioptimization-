# Week 4 Results: Red-Team Your Own Verifier

Oct 1, 2026

Per [docs/plan.md](plan.md) section 10's Week 4 checklist: plant known-bad
configs and confirm the harness flags them, add the physics ("speed of
light") check, add interleaved repeats with bootstrap intervals to every
comparison, and write down every hole found. A design review (same process
as Weeks 2-3) found real bugs before any of this was built -- including one
that means the Week 3 comparative numbers already published have a hidden
confound. Fixed, not hidden: see "Corrections to prior weeks" below.

## Bugs fixed (data-integrity issues, caught before or during this week)

1. **The Week 2/3 "KV f16 vs q8_0" comparison was confounded.**
   `specs/qwen3b-q4km-kvf16.json` has `flash_attn: false` (the dataclass
   default); `qwen3b-q4km-kvq8.json` has `flash_attn: true` (required for
   its q8_0 V cache). Every number in `docs/week2-results.md` and
   `docs/week3-results.md` comparing these two specs actually compared
   (f16, FA off) vs (q8_0, FA on) -- two variables, not one. See
   "Corrections to prior weeks" for the re-run with a real, single-variable
   baseline.
2. **`engine_flags` could silently override what the spec declared, and
   different tools would see them inconsistently.** `run_spec` passed
   `spec_flags(spec) + engine_flags` to the bench, but KL and the server
   only ever got `spec_flags(spec)` -- an `engine_flags` entry like
   `-ctk q4_0 -ctv q4_0 -fa on` would measure speed at one KV config and
   quality at another under one honest-looking `spec_hash`. Fixed:
   `CandidateSpec` now rejects any `engine_flags` entry outside a tiny
   allowlist (`-t`/`--threads`); anything that could change KV type,
   batch/ubatch size, or depth has to be a real spec field.
3. **`measure_bench` trusted `recs[0]` without checking it.** llama-bench
   can return more than one record, or one with different KV/depth/GPU-layer
   settings than requested. Fixed: `_check_bench_record` validates every
   field that matters (`n_prompt`, `n_gen`, `n_depth`, `type_k`, `type_v`,
   `flash_attn`, `n_gpu_layers`, `model_filename`) and marks the run
   `status="invalid"` on any mismatch, rather than silently passing.
4. **The bench workload hash omitted `depth`.** A ctx=8192 spec and a
   ctx=32768 spec got the same `workload_id` despite decoding behind very
   different KV cache sizes. Fixed: `depth` is now part of the hash.
5. **The artifact cache trusted any file at the expected path.** Nothing
   checked that `models/cache/<artifact_hash>.gguf` was actually the quant
   it claimed to be. Fixed: `ensure_artifact` now reads the cached file's
   real `general.file_type` and compares it against `spec.quant` on every
   cache hit.
6. **A physics violation needs to change `status`, not just add a
   `failures` row** -- otherwise a `WHERE status='ok'` query would still
   use the impossible number. Fixed: physics failures now set
   `status="invalid"`.
7. **Two arms running the identical spec (the A/A calibration test) would
   collide on `run_dir`**, since it was keyed by spec_hash + timestamp-to-
   the-second. Fixed: run directories now include a session id and arm
   index (`<spec_hash>_<session_id>_a<idx>`).
8. **New Week 4 data went into new tables (`schedule`, `checks`), not new
   columns on existing ones** -- `CREATE TABLE IF NOT EXISTS` can't alter
   an existing table, and this avoided needing another `--rebuild` just to
   land this week's additions.

## What got built

- [harness/physics.py](../harness/physics.py) -- the speed-of-light check.
  `peak_bandwidth_gbps()` computes 912.1 GB/s from NVML's **max** memory
  clock (not current -- current clock is ~405MHz at idle, which would make
  the bound ~20x too tight and flag every honest run), matching NVIDIA's
  published 912.4 GB/s spec for the RTX 3080 Ti almost exactly.
  `decode_bytes_per_token` excludes `token_embd.weight` from the byte count
  (verified live: it's a distinct, differently-quantized tensor from
  `output.weight` in Qwen2.5-3B's GGUF -- not tied -- and is only
  single-row-looked-up per token, unlike the LM head, which is read in
  full every step; excluding it tightens the bound by the ~8.3% it would
  otherwise overcount). Returns `None` for MoE models rather than
  misapplying the dense formula. A **separate**, deliberately loose compute
  bound covers prefill (which is compute-bound, not memory-bound -- the
  decode formula doesn't apply there).
- `harness/run.py` restructured so interleaved bench measurement across
  specs is the **only** execution path (`Arm` / `prepare_arm` /
  `bench_interleaved` / `run_quality` / `write_record`) -- a sequential,
  non-interleaved run (the Week 2/3 mistake) is no longer something that
  can happen by accident. Arm order rotates every round (a Latin square,
  not fixed A,B,A,B) and runs kind-major within a round.
- `harness/stats.py` gained `paired_rel_diff_ci` (resamples paired round
  indices together, so interleaving's pairing isn't thrown away the way the
  existing unpaired `bootstrap_diff_ci` would) and `bootstrap_prop_diff_ci`
  (for comparing task-suite scores from counts alone, works retroactively
  on data already in the DB).
- [harness/compare.py](../harness/compare.py) -- reads the DB, never a
  second execution path. Compares every arm against one baseline (not all
  pairs), reports a quality delta alongside every speed delta, and prints
  `QUALITY UNVERIFIED` instead of a clean speed win when an arm has no
  quality measurement.
- `harness/splits.py` gained a 13-gram shingle overlap check alongside the
  existing paragraph-hash check -- catches leaks that merge, lightly edit,
  or partially copy held-out content, which exact paragraph matching
  misses.

## The red-team exercise

| # | Attack | Result |
|---|---|---|
| R1 | A/A: identical spec as two interleaved arms, 10 rounds | **Passed.** `compare` reported "no detectable difference" for both pp (-8.3%,+7.1%) and tg (-3.5%,+1.1%). This is the check that the statistics themselves aren't producing false positives -- the single strongest result here. |
| R2 | Known-slower: Q8_0 vs Q4_K_M weights, same KV f16 + FA on | **Passed.** `compare` detected `tg: slower (-24.7%, -23.7%)`, CI clearly excludes 0; `pp: no detectable difference` (plausible -- pp is compute-bound and Q8_0 dequant isn't necessarily pricier there). Confirms the test has real power to detect a genuine difference, not just correctly stay silent on R1. |
| R3 | Known-worse quality: Q3_K_M vs Q4_K_M, same KV f16 + FA on | **Passed, with an interesting surprise.** KL mean jumped from 0.045 to 0.156 as expected (real quality cost, correctly surfaced in `compare`'s quality column). Speed: Q3_K_M came back **slower** (pp -6.6%/-0.5%, tg -13.0%/-10.5%), not faster, despite being a smaller file (1.72GB vs 2.10GB) with a *higher* physics bound (483.5 vs ~410 tok/s) -- verified this isn't a harness bug by checking `checks`: both configs stayed well within their physical bounds (utilization 0.35-0.38 for Q3_K_M), just with Q3_K's more complex per-block dequantization apparently costing more in practice than its bandwidth savings buy back, on this model/hardware. The physics check correctly didn't flag this as invalid -- it's surprising, not impossible, and the check knows the difference. `QUALITY UNVERIFIED` with `--no-kld` was already confirmed separately: every arm in R1 and R2 (both run with `--no-kld`) correctly showed it. |
| R4 | Flag smuggling: spec declares f16/FA-off, `engine_flags` tries `-ctk q4_0 -ctv q4_0 -fa on` | **Caught at spec load** -- `CandidateSpec` raises `ValueError` before anything runs. Regression test: `test_engine_flags_allowlist_rejects_kv_smuggling`. |
| R5 | Depth smuggling: `engine_flags=["-d","0"]` | **Caught at spec load**, same allowlist. With the allowlist temporarily removed to confirm the second layer: `_check_bench_record` also independently catches a depth mismatch (`test_check_bench_record_catches_wrong_depth`). |
| R6 | Cache poisoning: Q3_K_M file copied over a Q4_K_M `artifact_hash` path (scratch cache dir, real cache never touched) | **Caught.** `ensure_artifact` raised: `file_type=12, expected 15 for quant='Q4_K_M'`. |
| R7 | Inflated speed: `llama.bench` monkeypatched to return 50,000 tok/s | **Caught.** `status="invalid"`, `physics_decode` check failed with `utilization=122.0` (122x the physical bound). Permanent regression test: `test_measure_bench_marks_invalid_on_impossible_speed`. |
| R8 | Held-out leak variants, run live against the real `data/held_out.txt`: (a) renamed whitespace-changed copy, (b) two real held-out paragraphs merged with no blank line, (c) one word edited in an otherwise-copied real paragraph, (d) a real held-out sentence pasted into the middle of a dev paragraph | **All four caught.** (a) via whole-file content-hash classification. (b)-(d) via the new 13-gram shingle check -- confirmed **0 paragraph-hash matches** for all three (exact-paragraph matching alone would have missed every one), but 389, 301, and 6 shingle matches respectively. |
| R9 | Hot start: `wait_for_cool(max_temp_c=0)`, an unreachable target | **Passed.** Raised `TimeoutError` after the configured timeout rather than proceeding with a hot card. |

## Verification

`pytest tests/` -- 96 tests pass (8 new in `tests/test_run_checks.py` for
`_check_bench_record` and the physics wiring, 9 in `tests/test_physics.py`,
7 in `tests/test_compare.py`, plus new coverage in `test_splits.py` for the
shingle check and `test_harness.py` for the engine_flags allowlist). All
without needing a GPU, except the handful marked `requires_model` that read
real GGUF metadata.

## Corrections to prior weeks

**The Week 2/3 conclusion that "q8_0 KV cache is faster" was wrong -- it was
flash attention, not KV cache type, driving the speed difference.** Re-ran
the comparison properly: `specs/qwen3b-q4km-kvf16-fa.json` (f16 KV, FA **on**
-- the real, single-variable baseline) vs. `qwen3b-q4km-kvq8.json` (q8_0 KV,
FA on), interleaved, 10 rounds, through `harness.compare`:

| | KV f16 (FA on) | KV q8_0 (FA on) | Week 2/3's number (confounded: FA off vs on) |
| --- | --- | --- | --- |
| pp speed | 6333.9 t/s (baseline) | **slower**, CI (-9.9%, -4.3%) | reported "faster" |
| tg speed | 208.7 t/s (baseline) | **slower**, CI (-16.9%, -14.7%) | reported "faster" |
| GSM8K (n=30) | 0.633 | 0.633 (identical) | 0.667 vs 0.633 (also confounded) |
| KL mean | 0.0446 | 0.0446 (~identical) | 5.996 vs 6.014 (different ctx/chunks settings between sessions, not comparable to this table) |

With flash attention properly held constant across both arms, **q8_0 KV
cache is measurably slower than f16**, both CIs clearly excluding 0 -- the
opposite of what Week 2/3 reported. The earlier "faster" result was
entirely an artifact of comparing (f16, FA off) against (q8_0, FA on): FA
on is well known to speed up attention regardless of KV cache
quantization, and that's almost certainly all Week 2/3's speed numbers
were actually measuring. Quality (GSM8K, KL) still shows no detectable
difference between KV types, consistent with the (still-standing) Week 2/3
finding that KV cache precision barely affects quality at this size.

`docs/week2-results.md` and `docs/week3-results.md` are left as originally
written -- not retroactively edited -- since the raw numbers they report
are accurate for what was actually run; only the comparison and its
conclusion were confounded. Read their KV-comparison sections alongside
this correction, not standalone.

The methodology itself held up under its own test: R1 (A/A) correctly
found no difference between identical configs, and R2 correctly found a
real, known difference (Q8_0 vs Q4_K_M weights) -- so this reversal is a
real finding about the confound, not noise in the new measurement.

## Holes documented without a fix

- KL's default `--kld-ctx 512 --kld-chunks 50` (~25.6k tokens) is well
  under `docs/plan.md`'s "at least a few hundred thousand tokens, checked
  twice on different samples" rule. Flagged for Week 5, not changed here --
  changing it would make every future KL number incomparable to what's
  already measured without a clear before/after line.
- The prefill compute bound only catches gross errors (4x+), not subtle
  ones -- it ignores attention FLOPs entirely, deliberately, to stay a safe
  lower bound rather than a precise one.
- `ctx` (bench depth) already varies per spec; it's now visible in the
  workload hash, but whether depth should be a first-class sweep axis for
  Week 5's grid is a design question this week doesn't answer.
- LLM-agent red-teaming (`docs/plan.md` section 4's "give an LLM agent the
  goal and see what shortcut it finds") is deferred until Week 5's actual
  search loop exists.

## What's next

Week 5 per `docs/plan.md` section 10: enumerate every weights+KV config
that fits the 11GB budget at each target context length on the small
(3-4B) model, run the full grid through the now-interleaved harness, and
produce the first real Pareto plots.
