# Efficiency R&D Loop

A measurement harness for LLM inference efficiency on consumer GPUs (starting on one RTX 3080 Ti, 12 GB).

**First question:** at a fixed ~11 GB budget and 8k/16k/32k context, what split between weight
quantization and KV cache precision gives the best long-context quality in llama.cpp? And does
short-context KL divergence predict the answer?

Full plan: [`docs/plan.md`](docs/plan.md).

## Layout

| Path | What |
| --- | --- |
| `scripts/setup_llamacpp.sh` | Clone, pin and build llama.cpp with CUDA; the pinned commit goes in `LLAMACPP_COMMIT` |
| `scripts/lock_gpu.sh` | Lock/unlock core clock and power limit; show status |
| `harness/noise.py` | Run llama-bench N times with cool-down; median, CV, bootstrap CI |
| `harness/kld.py` | KL divergence + top-1 agreement vs a reference (Q8_0) via `llama-perplexity` |
| `harness/memory.py` | Analytical VRAM estimate (weights + KV cache) at each context length |
| `harness/stats.py` | Summaries and bootstrap confidence intervals |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
scripts/setup_llamacpp.sh           # builds into third_party/llama.cpp/build/bin
python -m pytest -q
```

## Week 1: reproduce, don't build

1. **Pin and build llama.cpp.** Run `scripts/setup_llamacpp.sh`, then commit `LLAMACPP_COMMIT`.
2. **Get a 3B–4B model** at F16 or BF16 GGUF (or convert it with `convert_hf_to_gguf.py`), then quantize:
   ```bash
   B=third_party/llama.cpp/build/bin
   for q in Q8_0 Q6_K Q4_K_M Q3_K_M; do $B/llama-quantize models/m-f16.gguf models/m-$q.gguf $q; done
   ```
3. **Lock the card and measure noise** (10 repeats of one config):
   ```bash
   sudo scripts/lock_gpu.sh lock 1500 300
   python -m harness.noise --model models/m-Q4_K_M.gguf --repeats 10 --max-temp 45
   ```
   The printed CV is your noise floor. Any later speed difference smaller than this doesn't count.
   Run it once unlocked too, to see how much the lock helps.
4. **KL vs Q8_0** at three quant levels. Use text that is not a public benchmark
   (for example a slice of recent Wikipedia or your own notes), and aim for at least 100k tokens:
   ```bash
   python -m harness.kld --ref models/m-Q8_0.gguf --text data/dev.txt \
       models/m-Q6_K.gguf models/m-Q4_K_M.gguf models/m-Q3_K_M.gguf
   ```
   You should see KLD and top-1 agreement get worse from Q6_K to Q3_K_M. If they don't, something is wrong.
5. **Check the memory model**:
   ```bash
   python -m harness.memory --gguf models/m-Q4_K_M.gguf --ctx 8192 16384 32768 --k q8_0 --v q8_0
   ```
   Compare against peak VRAM from `scripts/lock_gpu.sh status` while `llama-server -c 32768` runs,
   then set `--overhead-gb` from that measurement.
6. **Read** the KV Pareto paper and one mixed-precision tool writeup (links in `docs/plan.md` §6).

Week 1 is done when you have: a pinned commit, a noise-floor number, a KL table for three quants,
and a memory estimate checked against a real measurement.
