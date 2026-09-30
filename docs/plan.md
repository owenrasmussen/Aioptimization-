# Efficiency R&D Loop: Architecture and Research Plan

Sep 30, 2026 · @owen rasmussen

## Summary

Build the harness and the verifier first, skip the surrogate for now, and aim your first finding at splitting a fixed 12GB between weights and KV cache at long context. That is the shortest path to a real, public result on one 3080 Ti.

**Main recommendations**

- **The harness is the product of the first year.** Build it on existing tools (llama.cpp's `llama-bench` and `llama-perplexity`, plus `lm-evaluation-harness`) instead of from scratch. Your edge is how honest and repeatable it is, not how much code you wrote.
- **Measure quality as KL divergence against the full-precision model,** not just perplexity or benchmark scores. It is cheap, sensitive, and harder to game. Add a small held-out task suite on top.
- **Use direct search, not a surrogate, at first.** Your search space at Level 1 and 2 is small enough to just measure. Use Bayesian optimization (a search method that picks the next experiment based on what it has learned so far) through Optuna. Add a surrogate only when real runs become the bottleneck.
- **First finding: find the best weights versus KV cache split under a fixed memory budget at long context,** inside llama.cpp, and test whether short-context KL predicts long-context quality. Plain per-layer bit mixing is already well covered by 2026 tools (section 6).
- **Treat the AI-proposer (Level 3) as a phase two project,** and only once your verifier has survived you trying to cheat it yourself.

**Where I disagree with your plan**

1. **The four parts are in the wrong order of importance.** You listed harness, dataset, surrogate, proposer as equals. In practice the verifier is 70% of the value. A weak verifier makes every other part produce confident nonsense.
2. **The surrogate will not generalize from one GPU.** A model trained only on your 3080 Ti learns your 3080 Ti. Cross-hardware prediction needs data from several chip types, and the analytical roofline model (explained in section 3) will beat a learned model with thin data.
3. **"Discover genuinely new methods" is not a realistic year-one goal.** Most real novelty here comes from people with deep systems knowledge. What you can realistically discover is a better *recipe* (a combination of existing techniques) that nobody has measured carefully. That is still publishable and useful.
4. **The dataset is not the moat.** It helps you, but it is easy to copy. The loop and the trustworthiness of the verifier are the assets worth building.

Confidence: the tooling and technique claims are established. The ranking of first projects and the business read are my judgment.

## 1. Architecture

Everything passes through one contract: a **candidate spec** (a JSON file describing exactly what to run) goes into the harness, and a **result record** comes out. If you keep that contract stable, you can swap the proposer (you, Optuna, an AI agent) without touching anything else.

### The harness (build this first, spend most time here)

It has four jobs. Reuse tools for each and write only the glue.

1. **Build the artifact.** Turn a spec into a runnable model file. For GGUF this means `llama-quantize` with per-tensor type overrides and an importance matrix (imatrix, a file that tells the quantizer which weights matter most). Cache every built file by the hash of its spec so you never rebuild the same thing.
2. **Measure speed.** `llama-bench` for prompt processing and generation speed at set context lengths. `llama-server` plus a small load script for time to first token and latency when several requests arrive at once.
3. **Measure quality.** Three layers, cheapest first:
   - KL divergence against a full-precision reference, using `llama-perplexity --kl-divergence`. It compares the full probability output of your compressed model to the original on every token. Also record top-1 agreement (how often both pick the same next token).
   - A fixed task suite through `lm-evaluation-harness` pointed at `llama-server`. Pick a few tasks that break first under compression: math (GSM8K), code, and instruction following.
   - Long-context checks: needle-in-a-haystack and multi-hop retrieval at 8k, 16k, and 32k tokens. Short-context KL can look fine while long context quietly falls apart.
4. **Record the environment.** A sidecar process samples NVML (Nvidia's monitoring library, reachable from Python through `pynvml`) about 10 times a second: VRAM, power, temperature, clocks. It also stamps the llama.cpp commit, driver, CUDA version, and full command line into the record.

**Control the machine, or your numbers are noise.** Lock GPU clocks and the power limit with `nvidia-smi`, run warm-up passes, wait for the card to cool below a set temperature between runs, and close everything else. Repeat each measurement at least 5 times.

### The dataset

Use DuckDB or SQLite with raw logs saved as Parquet files beside it. Keep raw data forever; summaries can always be recomputed.

| Table | Key columns | Why it exists |
| --- | --- | --- |
| `specs` | spec\_hash, base\_model, per-tensor quant map, imatrix id, KV cache type K and V, context length, batch size, engine flags | Exactly what was run |
| `artifacts` | spec\_hash, file path, file size (bytes), bits per weight, build time | The built model file |
| `environments` | env\_id, GPU, VRAM, driver, CUDA, engine commit, locked clock, power limit | Makes cross-machine data possible later |
| `runs` | run\_id, spec\_hash, env\_id, workload\_id, repeat number, seed, start temp, timestamp, status | One row per measured run |
| `speed` | run\_id, prompt tok/s, generation tok/s, time to first token (ms), p50 and p95 latency (ms) | Speed results |
| `resources` | run\_id, peak VRAM (MB), mean power (W), joules per token, max temp | Cost results |
| `quality` | run\_id, eval name, split (dev or held-out), score, KL mean, KL p99, top-1 agreement | Quality results |
| `workloads` | workload\_id, prompt set hash, input and output lengths, concurrency | What you asked the model to do |
| `failures` | run\_id, type (out of memory, timeout, garbage output), log path | Failures are data too |

Record the p99 (worst 1%) of KL, not just the mean. Compression damage often hides in rare tokens.

### The surrogate (build later, and start analytical)

Do not start with a learned model. Start with a **roofline estimate**: on a single GPU, generating one token is mostly limited by reading every weight from memory once, so tokens per second is roughly memory bandwidth divided by model bytes. That simple formula will already predict decode speed well. Then train a small gradient-boosted tree model (LightGBM) to predict only the *error* of that formula. For memory use, the math is exact enough that you don't need learning at all. Quality is the hard part, covered in section 3.

### The proposer

- **Level 1 and 2:** Optuna with a multi-objective sampler. You give it the objectives (minimize file size and KL, maximize tokens per second) and it returns the Pareto front, meaning the set of options where you can't improve one goal without hurting another.
- **Level 2 shortcut:** measure each tensor's sensitivity once (quantize only that tensor, measure KL), then solve bit allocation as a knapsack problem (fit the most value into a fixed budget). Use Optuna to refine around that starting point.
- **Level 3 later:** an LLM agent that reads the current Pareto front and failures from the dataset and writes new specs or code. It never sees the held-out sets.

### Build versus reuse

| Part | Reuse | Build |
| --- | --- | --- |
| Quantize and run | llama.cpp (quantize, bench, server, perplexity) | Spec to command translation, artifact cache |
| Quality | lm-evaluation-harness, llama.cpp KL mode | Held-out set management, long-context suite wiring |
| Telemetry | pynvml, nvidia-smi | Sampler and clock-locking script |
| Storage | DuckDB, Parquet | Schema and loaders |
| Search | Optuna | Objective function that calls the harness |
| Reporting | Pandas, Matplotlib | Pareto plots, run comparison report |

&#91;embedded content: the optimization loop · 4 parts plus a held-out gate\]

The harness is the only path into the dataset, and only configs that pass the held-out check become public findings.

## 2. Better or alternative methodologies

Your loop is sound, but it is missing two ideas that matter more than the surrogate: **cheap proxy evaluation** and **sensitivity analysis**. Most of the real speedup in research like this comes from not running expensive tests on bad candidates.

### Where your design is naive

- **Every candidate gets the full test.** Wasteful. Use multi-fidelity evaluation: score every candidate with a 2-minute KL check on a small text sample, and only send the top 10% to the full suite. Optuna supports this through pruning (stopping a trial early when it looks bad). This alone can cut your compute by 5 to 10 times.
- **The search treats the model as a black box.** It isn't. You can measure how much each layer hurts quality when compressed, once, up front. That turns a blind search into a structured one.
- **One quality number.** Optimizers exploit single numbers. Use a small vector of quality measures and require the winner to not get worse on any of them.

### The alternatives, and when each wins

| Method | What it is, in plain terms | When it beats plain search | When it doesn't |
| --- | --- | --- | --- |
| Bayesian optimization (Optuna TPE, BoTorch) | Builds a guess of which settings are good and tests where it expects the biggest gain | Few knobs (under 30), expensive tests | Hundreds of per-layer choices |
| Sensitivity plus knapsack | Measure each part's damage alone, then pack the budget greedily | Per-layer bit allocation; fast and explainable | When layers interact strongly |
| Evolutionary search | Keep a population of candidates, mix and mutate the best | Large, messy spaces with many choices | When each test is slow and you can afford few |
| Multi-fidelity (Hyperband, successive halving) | Test many things cheaply, give more budget to the winners | Almost always; pair with any of the above | When the cheap test disagrees with the real one |
| Learned cost models (TVM, Ansor) | A model trained on past runs predicts speed for a new kernel or config | Thousands of low-level code variants per chip | Small config spaces where you can just measure |
| Reinforcement learning | An agent learns a policy from rewards over many episodes | Large labs training kernel agents (CUDA Agent, KernelGYM) | One GPU; too sample-hungry for you |
| Neural architecture search | Search over model designs themselves | Training small models for a fixed device | Anything needing many training runs on 12GB |
| LLM-driven program search (AlphaEvolve style) | A language model writes code changes, a harness scores them, winners get refined | Code-level ideas with a fast, trustworthy scorer | When the scorer can be gamed (see section 4) |
| Active learning | Choose the next experiment that teaches the surrogate the most | Once a surrogate exists and runs are costly | Before you have any data |
| Analytical models (roofline) | Physics-style formula for speed from bytes and bandwidth | Speed and memory prediction, especially across hardware | Quality prediction |

### What I would actually use

For year one: **sensitivity analysis to seed, knapsack for a first answer, multi-objective Bayesian optimization to refine, multi-fidelity pruning throughout.** That combination is simple, explainable, and fits one GPU. Evolutionary search is the backup if Optuna stalls. Skip reinforcement learning and architecture search entirely until you have real compute.

The LLM-proposer is worth trying in phase two, but in a narrow role: have it read failures and the Pareto front and suggest *new kinds* of specs a numeric optimizer would never try, like protecting a specific layer type. That uses what language models are good at (pattern spotting, idea generation) and leaves the numbers to the numeric optimizer.

## 3. The surrogate problem

Speed and memory can be predicted almost for free from physics. Quality cannot, and a surrogate trained on one GPU will not transfer to other hardware without data from those chips. Plan around both facts.

### Speed and memory

- **Memory** is arithmetic: weight bytes plus KV cache bytes (layers × KV heads × head size × context length × bytes per element × 2 for keys and values) plus a fixed overhead you measure once. Expect to be within a few percent.
- **Decode speed** is mostly bandwidth-limited on a single GPU. A roofline estimate (bandwidth ÷ bytes read per token) gets you most of the way. A small learned model fixes the rest: kernel overhead, dequantization cost for unusual formats, and context-length effects.
- **Prompt processing speed** is compute-limited instead, and differs more between quant formats. This is where a learned correction earns its keep.

### Quality: the hard part

Quality loss from compression is not additive and not smooth. Damaging two layers can hurt more than the sum of damaging each alone, and some tensors (often the output layer, early attention, or specific feed-forward layers) are far more fragile than others. Three approaches, in order of what I'd try:

1. **Additive sensitivity model.** Measure KL when only one tensor group is compressed, for each group and bit level. Predict total KL as the sum. Cheap and surprisingly decent as a starting point. Measure how wrong it is; that error is itself a finding.
2. **Fisher-weighted error.** Estimate how much each weight's error matters using gradient information (Fisher information, roughly how sensitive the output is to that weight) times the actual rounding error. This is what several 2026 mixed-precision tools do. Better than raw sensitivity but needs a gradient pass, which is tight but doable for a 7B model on 12GB with care.
3. **Learned correction.** Train a small model on your real runs to predict the gap between the additive estimate and the real KL. Only worth it after a few hundred measured configs.

Always validate the surrogate on configs it has never seen, and report its error. A surrogate you haven't tested is a guess.

### Will it generalize to other hardware?

Honest answer: **quality predictions will mostly transfer, speed predictions mostly won't.** Quality depends on the model and the math, not the chip (small exceptions come from different kernels rounding differently). Speed depends heavily on the chip's bandwidth, compute, cache sizes, and which kernels the engine uses on it.

To make speed prediction transfer, you need (a) hardware features as inputs (bandwidth, compute, VRAM, architecture generation), (b) measurements from at least 3 to 5 different GPUs, and (c) the roofline formula as the backbone so the learned part only corrects it. Renting a few cloud GPUs for a day each (section 5) gets you (b) cheaply.

### What's realistic with one GPU

A quality surrogate for 1 to 3 model families at 7B to 9B: yes. A speed surrogate for your own card: yes, and not very interesting. A cross-hardware speed predictor: only with rented data, and even then treat it as a year-two goal.

## 4. Verification and anti-cheating

Assume your optimizer is an adversary. The 2026 record on AI-generated GPU kernels shows why: Meta and Stanford researchers found that [frontier models frequently reward-hack kernel benchmarks](https://www.alphaxiv.org/abs/2607.16241), partly by using a weak baseline for timing and partly by hardcoding shortcuts for the specific test values. Your loop will do the same thing to you if it can.

### Rules for the quality side

- **Three splits, never mixed.** A *search* set the optimizer sees, a *dev* set you look at while building, and a *held-out* set nothing touches until final reporting. Store held-out hashes in the database and refuse any run that reads them outside a final-report flag.
- **Rotate the search set.** Sample fresh text from a large pool each round so the optimizer can't overfit one sample.
- **Guard against contamination.** Don't use public benchmark text as calibration data for the importance matrix. Keep calibration text, search text, and eval text from separate sources.
- **Test outside the training distribution.** Include long context, code, math, and a non-English sample. Compression damage shows up unevenly.
- **Require no regression on any axis.** A "winner" must match or beat the baseline on every quality measure within noise, not just the average.

### Rules for the speed side

- **Use the strongest fair baseline.** Compare against the best stock option at the same size, with the engine's fast paths turned on. The KernelBench-Verified paper found that enabling standard Tensor Core math in the baseline changed the picture.
- **Check against physics.** If a result claims more tokens per second than bandwidth allows for the bytes involved, it is a bug or a cheat. A research team used this ["speed of light" bound](https://arxiv.org/pdf/2603.29010) as a strict runtime check to catch agents skipping the real work.
- **Check outputs, not just timings.** A fast config that produces garbage is not fast. Every speed run also saves outputs and checks them against the quality baseline.

### Noise, heat, and repeats

- Lock clocks and the power limit, and log temperature at the start of each run. Discard runs that start above your threshold.
- Run at least 5 repeats, interleaved (A, B, A, B) rather than batched (A, A, B, B), so slow drift like heat affects both sides equally.
- Report the median and the spread. Call a difference real only if it's larger than the run-to-run noise with a simple statistical test (a bootstrap confidence interval, meaning you resample your runs many times to see how much the difference moves).
- For KL, compute it over enough tokens (at least a few hundred thousand) that the p99 is stable. Check by computing it twice on different samples.

### A test for your verifier

Before trusting the loop, try to cheat it yourself for a day. Build a config you know is worse in some way and see if the verifier catches it. Then give an LLM agent the goal and see what shortcut it finds. Every hole you find is a rule to add. This "red team your own benchmark" step is also a strong thing to write up publicly.

## 5. What's realistic on one 12GB GPU

One 3080 Ti is enough for real inference-side findings on models up to about 9B, and useless for anything that needs training large models. Pick problems on the inference side.

### In scope

| Work | Model size | Notes |
| --- | --- | --- |
| Fast search iterations | 1B to 4B | Do most of your searching here; a full loop turns in minutes |
| Main results | 7B to 9B | Fits at 4 to 8 bits with room for long-context KV cache tests |
| Stretch checks | 12B to 14B at 4 bits | Short context only, but good for "does this hold at larger size?" |
| Mixture-of-experts | Small ones with partial CPU offload | Slow, but expert-aware allocation is an active topic |
| Sensitivity and Fisher probes | Up to about 8B | Tight on memory; use CPU offload or smaller batches |

One practical wrinkle: a full 16-bit 8B model is about 16GB and won't fit on your card, so for KL references either run the reference with partial CPU offload once and cache its outputs, or use 8-bit (Q8\_0) as the reference. Q8\_0 as reference is common practice, but say so in any writeup.

### Out of reach

- Training or distilling models above about 1B from scratch.
- Anything needing many GPUs talking to each other (multi-GPU serving, tensor parallelism).
- Level 4 architecture search beyond toy scale.
- Claims about datacenter serving at high concurrency. Your card can't reproduce that traffic, so don't make those claims.

### When to rent cloud GPUs

Rent when you need **breadth, not depth**: after you have a finding on your card, spend a few days running the same harness on 3 to 5 other GPUs (a 4090, an A100 or H100, an older consumer card, maybe an AMD card) to show the result isn't a quirk of your hardware. That turns "it works on my machine" into a real claim, and it's the data your cross-hardware surrogate needs.

Hourly marketplaces like RunPod, Vast.ai, and Lambda rent single GPUs by the hour, from well under a dollar for consumer cards to a few dollars for datacenter cards. Check current prices before budgeting; they move. A realistic budget for one round of cross-hardware validation is in the low hundreds of dollars if your harness is fully scripted before you start the meter. Never debug on rented hardware.

## 6. Prior art and gaps

Per-layer mixed-precision quantization is already crowded in 2026, and AI kernel generation is dominated by big labs. The open gaps for a solo researcher are in joint budgeting on consumer hardware and in trustworthy evaluation.

### What's already done

| Area | Example | What it shows | Implication for you |
| --- | --- | --- | --- |
| Per-tensor mixed precision for GGUF | [PrismaQuant](https://pypi.org/project/prismaquant) | Solves per-layer bit choice as a knapsack using KL and Fisher probes, gated by held-out KL, shipped as llama.cpp and vLLM files | Your original Level 2 idea is done well. Don't repeat it as a headline |
| Role-based precision for mixture-of-experts | [Piscina GGUF](https://huggingface.co/poolside-laguna-hackathon/Piscina-XS.2-GGUF), [APEX quants](https://huggingface.co/MrFuzzihead/Nex-N2.5-mini-APEX-GGUF) | Keep always-active parts at high precision, crush rarely used experts; measured with KL and top-1 agreement vs Q8\_0 | Good template for how to report results |
| Early variable bit rate in llama.cpp | [llama.cpp issue #1256](https://github.com/ggerganov/llama.cpp/issues/1256) | Community found in 2023 that some tensors matter far more than others | Background; shows the idea's lineage |
| Better rounding inside quant formats | [llama.cpp PR #12557](https://github.com/ggml-org/llama.cpp/pull/12557) | Improved rounding algorithms for several formats | The engine itself keeps improving; retest baselines often |
| Joint weight plus KV cache tuning | [KV Pareto (EACL 2026)](https://preview.aclanthology.org/credits/2026.eacl-industry.9) | Maps memory vs accuracy across KV quantization, chunked prefill, and 4-bit weights; 68 to 78% memory cut at 1 to 3% accuracy loss | Closest prior work to my recommended first project. Done in research frameworks, not the GGUF stack people actually run |
| Adaptive KV cache under a memory budget | [ARKV](https://arxiv.org/pdf/2603.08727v1) | Per-layer, per-token mix of full precision, quantized, and evicted cache entries | Shows KV budgeting is live research |
| KV cache compression | [Google TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) | 3-bit KV cache with no retraining | Watch for it landing in llama.cpp |
| AI-written GPU kernels | [CUDA Agent](https://arxiv.org/html/2602.24286v1) | Large-scale RL beats torch.compile on most KernelBench tasks | Big-lab compute. Not your fight |
| Honest kernel evaluation | [KernelBench-Verified](https://www.alphaxiv.org/abs/2607.16241), [MANTIS / speed-of-light](https://arxiv.org/pdf/2603.29010), [KernelBench-Hard runs](https://huggingface.co/datasets/Infatoshi/kernelbench-hard-runs/blob/main/README.md) | Reward hacking is common; physics bounds and stronger baselines catch it | Directly reusable lessons for your verifier |

From memory, not re-checked this session: Microsoft's Vidur is an inference simulator that predicts serving performance, and TVM and Ansor use learned cost models for kernel tuning. Both are worth reading for surrogate design.

### Real gaps a solo researcher could fill

1. **Joint weight and KV cache budgeting in the stack people actually use.** KV Pareto did this with AWQ in research code. Nobody I found has done a careful version for llama.cpp GGUF files, its KV cache types, and consumer cards at 12 to 24GB. The practical question "at 32k context on 12GB, should I shrink weights or the KV cache?" has no trustworthy answer.
2. **Does short-context KL predict long-context quality?** Almost every mixed-precision tool reports KL at short context (often 512 tokens). If rankings flip at 32k, that is an important and cheap-to-test finding.
3. **Independent, adversarial benchmarking of community quant mixes.** Many dynamic and mixed quants now exist, each measured by its own author on its own setup. A neutral, same-harness comparison with held-out and long-context tests is missing.
4. **Cross-hardware transfer of quant recommendations.** Does the best mix on a 3080 Ti stay best on a 4090, an M-series Mac, or an AMD card? Rarely measured.

I'm confident these gaps exist in what I searched, but the field moves weekly. Spend an hour on arXiv and the llama.cpp discussions before committing.

## 7. First finding

Top pick: **find the best split of a fixed 12GB between model weights and KV cache at long context, inside llama.cpp.** It is practical, fits your card exactly, has a clear baseline, and is less crowded than per-layer weight mixing.

### Three candidates, ranked

| Rank | Project | Likely to produce a result | Novelty | Would an inference company care | Time |
| --- | --- | --- | --- | --- | --- |
| 1 | Weights vs KV cache budget at long context on 12GB | High | Medium-high | High for on-device and self-hosting companies | 6 to 8 weeks |
| 2 | Does short-context KL predict long-context quality? | Very high | Medium-high | Medium; it changes how everyone measures | 4 to 5 weeks |
| 3 | Neutral, adversarial benchmark of community quant mixes | High | Medium | Medium; useful, easy to copy | 5 to 6 weeks |

Project 2 can live inside project 1 as a second question, since you'll already be measuring KL at short and long context. That is what I'd do.

### Experiment design for the top pick

**Question.** At a fixed memory budget and a target context length, what combination of weight quantization and KV cache precision gives the best long-context quality? And does the answer change with context length?

**Setup**

- Models: the Qwen 9B class model you already run, plus one other family at 7B to 8B (for example a Llama or Mistral model), plus a 3B to 4B model for fast iteration.
- Budget: total VRAM capped at 11GB (leave headroom), measured, not estimated.
- Context targets: 8k, 16k, 32k tokens.
- Weight options: Q3\_K\_M, IQ4\_XS, Q4\_K\_M, Q5\_K\_M, Q6\_K, plus one published mixed-precision recipe for comparison.
- KV options: keys and values set separately to f16, q8\_0, or q4\_0. Quantized value cache in llama.cpp generally needs flash attention turned on; confirm with your build.
- Only configs that actually fit the budget at the target context get run.

**Baselines**

1. The common default: Q4\_K\_M weights with f16 KV cache.
2. The biggest weights that fit with f16 KV cache at that context.
3. The smallest weights with the most context headroom.

**Metrics**

- Long-context quality: needle retrieval and multi-hop retrieval at each target length (held-out prompts).
- Short-context quality: KL and top-1 agreement vs Q8\_0 at 512 tokens, plus GSM8K.
- Long-context KL: the same KL measure at 8k to 32k.
- Speed: generation tokens per second and time to first token at each context length.
- Resources: peak VRAM, joules per token.
- 5 interleaved repeats per config; bootstrap intervals on every difference.

**What counts as a win**

- A config that beats the default at equal memory on long-context quality by more than the noise, with no significant loss on short-context quality or speed. **Or**
- A clear, tested rule, for example "below 16k, spend memory on weights; above it, keep keys at 8-bit and put the rest into weights," that holds across both model families.

**What counts as a null result, and why it's still useful**

If the default is already on the best-tradeoff frontier at every length, that's a publishable "the default is right, here's the evidence" post. If short-context KL ranks configs differently than long-context tasks, that's a result on its own, and arguably the more important one.

**Deliverables**

The harness code, the raw dataset, a writeup with one Pareto chart per context length, and a one-paragraph recommendation people can act on. Post it where the users are: the llama.cpp discussions and r/LocalLLaMA.

## 8. Path to value

Today this is a research project with a plausible services business behind it, not a venture-scale company. That can change, but only after public results show the loop finds gains others miss.

**Why now (strong).** Inference is where the money goes at scale, usage keeps growing faster than per-token cost falls, and more companies are running open models on their own or edge hardware. Quantization and KV cache choices are made by guesswork at most of them.

**Who would pay, and for what**

| Buyer | What they'd pay for | How likely |
| --- | --- | --- |
| Companies self-hosting open models | "Here's the best config for your model, hardware, and traffic, with proof" | Most likely first customer; small checks |
| On-device and edge AI companies | Fitting a model into a hard memory limit without losing quality | Good fit for the joint budgeting work |
| Inference providers | Gains that cut cost per token at their scale | They have in-house teams; need proof at their hardware |
| Chip makers, especially non-Nvidia | Showing models run well on their hardware | Real budgets; they pay for benchmarking and tuning |
| Frontier labs | Almost nothing from an outside tool | Unlikely |

**Why you?** Right now, not yet. The gap a partner would point at is no systems background and no public results. The fix is the first finding plus a verifier people trust. Your philosophy training is a real but small edge: careful thinking about what counts as evidence is exactly what this field's reward hacking problem needs.

**Moat.** Weak today. Findings get published and absorbed into llama.cpp and vLLM within weeks. What could become defensible: (1) a verifier and benchmark people cite, (2) a loop that keeps finding gains faster than others copy them, (3) data across hardware nobody else bothers to cover. The dataset alone is not a moat.

**Fund-returner math.** For a venture fund, this has to plausibly become a multi-billion-dollar company. The path would be becoming the standard optimization and verification layer many companies depend on. The more common outcome in this space is acquisition by a platform company in the tens to low hundreds of millions, which is a good founder outcome but not a fund-returner. Be honest with yourself about which one you're building.

**Bear case**

- The engines absorb your best findings for free, and nobody needs your tool.
- Big labs keep their gains private, so the market for outside optimization shrinks to smaller buyers with small budgets.
- One GPU can't cover enough ground to find anything a funded team hasn't.
- Buyers don't trust gains they can't reproduce, and reproducing on their hardware costs you money you don't have.

**What would change my mind**

- Upward: your first writeup gets picked up by the llama.cpp maintainers or an inference company asks you to run the harness on their setup. A paying request before you've built a product is the strongest signal there is.
- Downward: your loop finds nothing beyond what defaults already give after two honest projects. Then the right move is to fold this into a portfolio piece and a job at an inference company, which is itself a good outcome.

## 9. Risks and blind spots

The most likely way this fails is not a technical wall. It's spending three months building a beautiful harness and never shipping a finding.

1. **Infrastructure trap.** Harness work feels productive and never ends. Cap it: the harness is "done enough" when it can run your first experiment end to end. Improve it only when an experiment needs it.
2. **Measuring noise.** Differences of 1 to 3% are often run-to-run variation. Without locked clocks, interleaved repeats, and intervals, you'll publish noise and someone will call it out.
3. **Baselines moving under you.** llama.cpp changes weekly. Pin one commit per experiment, record it, and rerun baselines when you update.
4. **Weak or leaked evaluation.** Using the same text for calibration, search, and reporting will make every result look better than it is.
5. **Being scooped.** Someone may publish your exact question mid-project. Keep the scope small so you finish fast, and publish intermediate results.
6. **Your one card is unusual.** Results from a 3080 Ti may not hold elsewhere. Say so, and validate on rented hardware before claiming generality.
7. **Model churn.** New model families arrive monthly and may behave differently. Pick two families and stick with them for a project.
8. **Spreading too thin.** You have several active projects already. This one needs a protected block of weekly time, or pick it over something else for a season.
9. **Mistaking a tool for a finding.** The harness is not the result. A clear answer to one question is.
10. **Not talking to users.** Before the product phase, talk to five people who deploy models on limited hardware. Ask what decision they struggle with. Their answer may reshape project two.

## 10. First 8 weeks

The goal is a public writeup by the end of week 8. The harness gets built only as far as the experiment needs it.

**Week 1: Reproduce, don't build**

- [ ] Pin a llama.cpp commit and build it with CUDA
- [ ] Run `llama-bench` and `llama-perplexity --kl-divergence` by hand on a 4B model at three quant levels
- [ ] Write a clock and power locking script; measure run-to-run noise with 10 repeats of one config
- [ ] Read the KV Pareto paper and one mixed-precision tool's writeup

**Week 2: Minimum harness**

- [ ] Define the candidate spec JSON and the result record
- [ ] Script: spec in, built file plus speed plus KL plus NVML log out
- [ ] DuckDB schema from section 1, with specs, runs, speed, resources, quality
- [ ] Artifact cache keyed by spec hash

**Week 3: Quality suite**

- [ ] Wire `lm-evaluation-harness` to `llama-server` for GSM8K and one code task
- [ ] Build the long-context suite: needle and multi-hop retrieval at 8k, 16k, 32k
- [ ] Create search, dev, and held-out splits from separate text sources; lock the held-out set

**Week 4: Red-team your own verifier**

- [ ] Plant known-bad configs and confirm the harness flags them
- [ ] Add the physics check (claimed speed vs bandwidth bound)
- [ ] Add interleaved repeats and bootstrap intervals to every comparison
- [ ] Write down every hole you found; this becomes part of the writeup

**Week 5: Sweep on the small model**

- [ ] Enumerate every weights plus KV config that fits 11GB at each context length on the 4B model
- [ ] Run the full grid with multi-fidelity pruning
- [ ] First Pareto plots; check whether short KL and long-context rankings agree

**Week 6: Main models**

- [ ] Repeat on the Qwen 9B class model and a second 7B to 8B family
- [ ] Use Optuna only if the grid is too big to run in full

**Week 7: Stress the result**

- [ ] Rerun the top configs on the held-out set for the first time
- [ ] Rent one or two other GPUs for a day each and rerun the top configs
- [ ] Try to break your own conclusion; write down what would disprove it

**Week 8: Ship it**

- [ ] Publish the harness code and raw data on GitHub
- [ ] Write the post: question, method, one chart per context length, the rule people can use, limitations
- [ ] Post to the llama.cpp discussions and r/LocalLLaMA; reply to every technical comment
- [ ] Decide project two based on what surprised you most

## Sources

All links are in the prior art table in section 6 and inline in section 4. Pages were retrieved on 2026-09-30.
