# Week 3 Results: Quality Suite

Oct 1, 2026

> **Correction (Week 4):** the "first real comparison" section below (f16
> vs. q8_0 KV cache) was confounded by `flash_attn` differing between the
> two specs, not just KV type. With that controlled for, q8_0 KV cache is
> actually *slower* on speed; the quality findings (no detectable
> difference) still stand. See
> [docs/week4-results.md](week4-results.md)'s "Corrections to prior weeks"
> for the corrected, single-variable, properly-interleaved comparison.

Added the other two quality layers from [docs/plan.md](plan.md) section 1
(KL divergence shipped in Week 1/2): a fixed task suite (GSM8K + HumanEval)
via `lm-evaluation-harness` against `llama-server`, and a long-context suite
(needle-in-a-haystack + multi-hop retrieval) at 8k/16k/32k. Also formalized
the search/dev/held-out text splits [docs/plan.md](plan.md) section 4's
anti-cheating rules require, with the held-out set's hash locked.

A design review done in parallel with planning (before any code was written,
same process as Week 2) caught 13 issues, two of them serious enough that
building against the original plan would have produced silently wrong data.
Both were independently verified against the real, running binaries before
being accepted -- not taken on the reviewer's word.

## What got built

- [harness/server.py](../harness/server.py) -- `LlamaServer`: starts
  `llama-server`, polls `/health`, cross-checks `/props` against the model
  actually requested, exposes `.tokenize()` and `.chat()`. Pins `-np 1
  --fit off` explicitly (see "bugs" below) and redirects the server's
  stdout/stderr to a log file instead of a pipe (an undrained `PIPE` fills
  its OS buffer and hangs a long-running child process).
- [harness/splits.py](../harness/splits.py) -- `data/splits.json` is the
  source of truth for search/dev/held-out, both for free text (KL, the
  long-context haystack) and for lm-eval task items (GSM8K/HumanEval doc
  indices, partitioned deterministically by hashing `(seed, task, index)`).
  Two chokepoints -- `load_eval_text` and `task_doc_indices` -- are the only
  places anything reads a split; both refuse the held-out split without
  `--allow-held-out`.
- [harness/longctx.py](../harness/longctx.py) -- needle-in-a-haystack and
  multi-hop retrieval, haystack sized to an *exact* token count via the
  server's own `/tokenize` (not a word-count estimate -- see "bugs"), fixed
  code-format answers (`KX-\d{7}`) so scoring is an exact match, not a
  substring check.
- [harness/lmeval.py](../harness/lmeval.py) -- GSM8K via
  `lm_eval.simple_evaluate` (library call, not CLI+output-parsing). HumanEval
  does **not** go through lm-eval's built-in task at all -- see below.
- [harness/fetch_wiki.py](../harness/fetch_wiki.py) -- built `data/search.txt`
  (15 articles) and `data/held_out.txt` (8 articles), disjoint from `dev.txt`'s
  existing 10 and from each other; records titles and Wikipedia revision ids.
- `harness/db.py`: `quality` table gained `n_items`/`stderr`/`details`; new
  `split_files` table mirrors `data/splits.json` so a schema `--rebuild`
  (which deletes the `.duckdb` file) can't lose the held-out lock -- the
  manifest is a separate file that survives.

## Bugs caught before or while building (not style preferences)

1. **HumanEval cannot run through lm-eval-harness's built-in task on
   Windows, at all, under any flag.** Verified live: loading
   `humaneval`/`humaneval_instruct` raises `NotImplementedError: This metric
   is currently not supported on Windows` inside HuggingFace `evaluate`'s
   `code_eval`, and this fires at **task import time** (triggered by
   PyYAML's constructor loading `lm_eval/tasks/humaneval/utils.py`, which
   runs a self-test `compute()` call at module scope) -- before
   `simple_evaluate` even starts, so `predict_only=True` does not help.
   `harness/lmeval.py` instead loads `openai/openai_humaneval` directly and
   scores pass@1 itself via `subprocess.run(..., timeout=...)`, which (unlike
   `evaluate`'s `signal.alarm`-based timeout) works identically on Windows
   and POSIX.
2. **`llama-server`'s `--fit` flag defaults to `on`.** Confirmed via
   `--help`. Unset, the server silently shrinks parameters -- including
   context size -- to fit available VRAM. Without `--fit off`, a "32768
   context" long-context test could silently run at a smaller context with
   no error, invalidating exactly what that suite measures. `-np 1` is
   pinned for the same reason (`-np`/`--parallel` defaults to auto, which
   can split the KV cache across multiple slots).
3. **A Windows newline bug broke held-out drift detection in the first
   version of `splits.py`**, caught by actually running the unit tests, not
   by inspection: `classify_text` hashed `path.read_bytes()` while
   `lock_held_out` hashed `path.read_text().encode()`. On Windows, text-mode
   writes translate `\n` to `\r\n`, so the two hashing paths disagreed on
   the exact same file's hash. A drifted (edited-after-locking) held-out
   file would have come back `"adhoc"` instead of `"held_out"`, silently
   skipping the whole point of the lock. Fixed by hashing decoded-then-
   re-encoded UTF-8 text everywhere, and by also checking by *path* (not
   just content hash) so an in-place edit is still caught even though its
   hash no longer matches anything.
4. **A regex bug in the needle/multi-hop scorer**, also caught by running
   the tests: `KX-\d{7}` matched as a *prefix* of an 8-digit number (e.g.
   `KX-1234567` "matched" inside `KX-12345678`), which would have scored
   wrong-answer responses as correct. Fixed with a negative lookahead,
   `KX-\d{7}(?!\d)`.
5. **157 "shared paragraphs" between `dev.txt` and `held_out.txt` turned out
   to be Wikipedia boilerplate** (`== References ==`, `== See also ==`,
   stray infobox table fragments like `|` and lone digits), not real
   content leakage -- the 13-gram overlap check showed 0.0 between the same
   two files, which didn't match. `harness/splits.paragraphs()` now drops
   chunks under 5 words, which cleanly removes every observed fragment
   (longest was 4 words) without touching real sentences. This also
   improved the long-context haystack, which no longer wastes filler slots
   on garbage one-word "paragraphs."
6. **The KL reference-logits cache key didn't include the text file or
   chunk count** -- a pre-existing bug from Week 1/2, not introduced this
   week, but one that Week 3's multiple-text-sources feature would have
   triggered immediately: pointing `--kld-text` at a different file with
   the same `--kld-ref` would have silently reused the wrong cached
   reference logits. Fixed: the cache filename now includes a hash of the
   text and the chunk count.
7. **KV-cache flags could drift between llama-bench, llama-perplexity, and
   now the server** -- the same bug class as Week 2's KL-missing-KV-flags
   issue. Consolidated into one `llama.spec_flags()` used everywhere.
8. **The HumanEval subprocess scorer doubled its own working directory on
   the first live run** -- `subprocess.run(..., cwd=work_dir)` plus a
   relative `program_path` resolved to `work_dir/work_dir/<file>.py`, so
   every one of the first 5 live items "failed" with "can't open file,"
   not a code-correctness result. `work_dir` is relative in real usage; a
   unit test using pytest's (always-absolute) `tmp_path` wouldn't have
   caught this, so the regression test `chdir`s into a tmp directory and
   passes a relative path to actually match production. After the fix: 7/10
   passed on the next live run, with real per-problem pass/fail variation.

## Verification

All on the real RTX 3080 Ti, using `specs/qwen3b-q4km-kvf16.json` (Q4_K_M,
dev split unless noted):

**GSM8K** (20 items, `--lm-eval-limit 20`): score 0.45 (stderr 0.11) on the
`exact_match,flexible-extract` metric. Small sample, wide error bar -- a
real number from a real pipeline, not a precision claim.

**Long-context, 8192 tokens** (2 seeds): needle 6/6 (1.00), multi-hop 0/4
(0.00). A clean instance of exactly what `docs/plan.md` predicted this suite
would catch: a model can look fine on simple retrieval and fail completely
on a test that requires combining two facts, at the same context length.
Actual prompt token counts (via `/tokenize`, not estimated) landed at
7718-8161 against an 8192 target -- comfortably under budget, confirming
the exact-sizing approach doesn't overflow the context window the way a
word-count estimate could have.

**HumanEval** (10 items, custom pass@1 scorer, bypassing lm-eval entirely):
7/10 passed (0.70). Caught a real path bug on the first live run (every item
failed with "can't open file") -- `subprocess.run`'s `cwd=work_dir` combined
with a *relative* `program_path` doubled the directory
(`work_dir/work_dir/<file>.py`). `work_dir` is relative in real usage (it's
built from `run_dir`, itself `results/runs/<id>`), so an absolute `tmp_path`
in a naive unit test wouldn't have reproduced it -- fixed by resolving
`program_path` to an absolute path, with a regression test that `chdir`s
into a tmp directory and passes a relative `work_dir` to actually match the
real failure mode, not just the fix.

**Splits:** `data/search.txt` (15 articles), `data/held_out.txt` (8
articles, locked with 420 paragraph hashes after the boilerplate filter)
built and verified disjoint from `dev.txt` (13-gram overlap: dev-search
0.00015, dev-held_out 0.0, search-held_out 0.00005 -- negligible, consistent
with independent article selection rather than shared quotes).
`--eval-split held_out` without `--allow-held-out` is refused at
argument-parsing time, before any model is built or server started.

All 60 unit tests pass (`pytest tests/`), covering spec/db behavior from
Weeks 1-2 plus this week's splits chokepoints, the resource sampler, the
needle/multi-hop generators, `LlamaServer.build_cmd`'s `-np 1`/`--fit off`
regression coverage, and the GSM8K/HumanEval helper functions -- all without
needing a GPU or network access.

## A first real comparison: does KV cache precision cost anything?

With the quality suite actually working, it's worth pointing it at the
question Week 2 only had KL and speed/VRAM data for. Same Q4_K_M weights
(one cached artifact, per Week 2's two-hash scheme), f16 vs. q8_0 KV cache,
largest-sample result per metric from the runs above:

| Metric | KV f16 | KV q8_0 | Difference |
| --- | --- | --- | --- |
| KL mean vs. Q8_0 weights | 5.996 | 6.014 | ~0 |
| GSM8K (n=30) | 0.667 ± 0.088 | 0.633 ± 0.089 | within 1 stderr |
| HumanEval (n=30) | 0.633 ± 0.088 | 0.633 ± 0.088 | **identical** |
| Needle @ 8k/16k/32k (n=6 each) | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | none |
| Multi-hop @ 8k/16k/32k (n=4 each) | 0.00 / 0.25 / 0.00 | 0.00 / 0.25 / 0.00 | none |

Every quality metric tracks the other KV config within noise, at every
context length tested, while Week 2 already measured q8_0 KV cache as both
faster (3825 vs. 2426 t/s prompt processing at this size) and lighter on
VRAM (2.68 vs. 2.98 GB). On this model and this range, there's no detected
quality cost to q8_0 KV cache -- which is itself informative for the
project's central question (weights vs. KV cache budget at a fixed memory
limit): spending the saved VRAM on bigger weights, not on keeping KV at
f16, looks like the right call so far.

Caveats, stated plainly rather than buried: sample sizes are small (n=30 for
the task suite, n=4-6 per long-context cell) -- these are "the harness
produces a believable signal" numbers, not publication-grade confidence
intervals. Multi-hop's 8k/16k/32k pattern (0.00/0.25/0.00) is not
monotonic, which with n=4 is well within a couple of items flipping by
chance, not a claim that 16k context is special. And this is one model
family at one weight quant -- the real experiment (per `docs/plan.md`
section 7) sweeps weight quant **and** KV precision together across model
families, which is Week 5-6 territory, not this week's.

## What's next

Week 4 per `docs/plan.md` section 10: red-team the verifier just built.
Plant known-bad configs and confirm the harness flags them, try to break the
held-out lock on purpose (the paragraph-hash mechanism exists specifically
for Week 4 to attack), add the "speed of light" physics check against
measured bandwidth, and add interleaved repeats with bootstrap intervals to
every comparison that doesn't have them yet.
