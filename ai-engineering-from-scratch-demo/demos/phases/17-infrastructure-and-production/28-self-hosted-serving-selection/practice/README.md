<!-- generated:start -->
# 17-infrastructure-and-production / 28-self-hosted-serving-selection

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/17-infrastructure-and-production/28-self-hosted-serving-selection/) · upstream spec
`phases/17-infrastructure-and-production/28-self-hosted-serving-selection/docs/en.md`

```bash
uv run demo practice run 28-self-hosted-serving-selection --ex 1
uv run demo explain 28-self-hosted-serving-selection --ex 1
uv run pytest demos/phases/17-infrastructure-and-production/28-self-hosted-serving-selection
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py` with your hardware / scale / workload. Does the output match your intuition? | code | T0 | `ex01_scale_moves_the_pick_on_three_hardware_classes_and_workload_on_two.py` |
| 2 | Your infra is 12 H100s and 8 MI300X AMD. What engine? Why is TRT-LLM off the table? | code | T0 | `ex02_both_pools_get_vllm_and_the_walker_never_offers_trt_llm_on_hopper_either.py` |
| 3 | A team wants to use TGI in 2026 because "it's what we know." Argue the migration case. | explain | T0 | prose, below |
| 4 | Ollama dev to vLLM prod: what changes in quantization, configuration, and observability? | code | T0 | `ex04_the_walker_answers_the_move_with_one_engine_name_and_none_of_the_three_changes.py` |
| 5 | RAG product with P99 prefix length 8K and high reuse across tenants. Pick an engine and stack… | code | T0 | `ex05_the_walker_picks_sglang_but_the_stack_is_vllm_and_its_lmcache_is_the_slowest_config.py` |
<!-- generated:end -->

## Answers

### 1 — scale moves the pick on three hardware classes and workload on two

**On the lesson's 7 scenarios it matches; off them it does not.** The seven
shipped scenarios read as intended. I also ran the whole grid the lesson's
vocabulary spans: 5 hardware classes, 4 scales and the 5 workloads "Workload-third
decision" names, 100 calls in all.

| varying | changes the engine on |
|---|---|
| scale | CPU, Apple Silicon, NVIDIA Hopper |
| workload | AMD, NVIDIA Hopper |

Only "agentic multi-turn" and "RAG with heavy prefix reuse" ever change the
pick. "Code generation" and "long-context 128K" never do. My own box, "NVIDIA RTX
4090", gets `engine: None`, and so do "H100", "cpu" and "MI300X". Only
exact-cased class names match, and an unmatched name fails silently: the TGI line
is the whole explanation.

**The tree does not run hardware, then scale, then workload.**

- On Hopper, a single user doing agentic work gets SGLang, so workload overrides
  scale.
- On AMD a single user gets vLLM, and on Blackwell TRT-LLM. Neither branch reads
  scale, although the lesson sends one user to Ollama.
- Blackwell returns TRT-LLM on all 20 of its cells, agentic included, even though
  the lesson says RadixAttention "dominates" agentic work.

**The enterprise line stacks vLLM's production-stack onto engines that are not
vLLM.** It is appended to all 25 enterprise cells, and 19 of those picked
llama.cpp, SGLang or TRT-LLM. TGI's maintenance line appears on 100 of 100
outputs, and TGI is the engine on 0.

### 2 — both pools get vLLM, and the walker never offers TRT-LLM on Hopper either

**One engine for the whole fleet, with one decision per pool.** A mixed fleet is
two pools, since tensor parallelism does not cross vendors. H100 maps to "NVIDIA
Hopper" and MI300X to "AMD", and the two pools get the same engine for all 5
workloads at production scale:

- vLLM for chat, code and long context
- SGLang for agentic and prefix-heavy work

The walker accepts neither card name: "H100", "MI300X" and "12 H100s and 8
MI300X" all return `None`.

**Why TRT-LLM is out.** It is NVIDIA-only, so it could serve the 12 H100s: 60%
of the GPUs, but only 960 of 2496 GB of HBM (80 GB per H100 SXM, 192 GB per
MI300X, vendor specs). The 8 AMD cards hold 62% of the fleet's memory. Choosing
TRT-LLM means running two engines with two configs and two dashboards. The edge
it would buy is one the lesson places on Blackwell, not Hopper.

**The walker never returns TRT-LLM on Hopper.** The lesson says Hopper → "vLLM
or SGLang or TRT-LLM. All three top-tier." In the walker, TRT-LLM comes back on
20 of 100 cells, all of them Blackwell, and on 0 of 20 Hopper cells. The
sentence "TRT-LLM is NVIDIA-only" appears only on the AMD branch.

**The Blackwell fallback cites the wrong Blackwell.** On B200/GB200 the walker
names vLLM as "close second" because of "Blackwell SM120". The lesson itself
calls SM120 RTX Blackwell, and B200 is SM100.

### 3 — TGI's own docs recommend vLLM or SGLang, and the weights carry over unchanged

*Draws on "The TGI maintenance trap".*

**The date is right, and the vendor has already made the case.** Hugging Face's
Inference Endpoints page for TGI says it is "in maintenance mode as of
12/11/2025", and that only "minor bug fixes, documentation improvements, and
lightweight maintenance tasks" will be accepted. The TGI README and docs index
carry the same notice. They recommend "vllm, SGLang, as well as local engines …
such as llama.cpp or MLX" (all checked 2026-09-26). The lesson's own walker
agrees: across 100 inputs it returns TGI 0 times.

**"It's what we know" is an argument for migrating now.** TGI does not stop
working, but it stops moving. New model architectures, new kernels and new
accelerator support will land in vLLM and SGLang and not in TGI. A project
starting in 2026 would therefore take on a migration it already knows it must
make, at a later date and with more traffic behind it. What the team knows
mostly carries over:

- **Weights**: TGI loads HF safetensors, as vLLM and SGLang do, so this is not
  the GGUF-to-safetensors conversion of the pipeline pattern.
- **Hardware**: TGI's README lists NVIDIA and AMD ROCm among its backends. vLLM
  covers both, and exercise 2 finds it the pick on both.
- **Observability**: TGI advertises Prometheus metrics and OpenTelemetry
  tracing. vLLM exposes both, so dashboards need their metric names remapped,
  not rebuilt.

**What actually has to be redone** is the capacity configuration. TGI's max
batch total tokens and max input tokens become vLLM's `--max-model-len`,
`--max-num-seqs` and `--gpu-memory-utilization`, and they need a load test at
the new settings. HF's own migration guide is four steps: create a new vLLM
endpoint with the same model ("you can typically use the same hardware and
configuration"), test it, switch traffic, and retire the old endpoint.

**Posture.** A new project should not start on TGI. An existing TGI service can
keep running but should be migrated behind a canary; the lesson's
`skill-engine-picker.md` says to start within six months. I did not verify the
lesson's "~10% slower raw throughput than vLLM" figure, and this argument does
not depend on it.

### 4 — the walker answers the move with one engine name and none of the three changes

The worked model is Llama-3.1-8B: 8.03B parameters, 32 layers, 8 KV heads of
dim 128. Prod is one 80 GB H100 at vLLM's default memory utilization of 0.9.

| | Ollama (dev) | vLLM (prod) |
|---|---|---|
| weights | GGUF Q4_K_M ≈ 4.85 bpw → **4.87 GB** | safetensors FP16 **16.06 GB** / FP8 **8.03 GB** |
| memory model | one context: 8K × 128 KiB = **1.07 GB** KV | shared KV pool: 72 GB − weights = **427K tokens → 52** concurrent 8K seqs (FP8: **59**) |
| knobs | context size, parallel slots | `--max-model-len`, `--max-num-seqs`, `--gpu-memory-utilization`, `--quantization` |
| metrics | per-response timing fields; no server `/metrics` | Prometheus `/metrics`, OTLP traces |

**Quantization changes the file, not just a flag.** vLLM's docs call GGUF
support "highly experimental and under-optimized", so prod serves a different
weight file with different numerics, and the dev evals have to be rerun on it.

**Observability.** As of an open Ollama PR dated 2026-09-17 that proposes
`/metrics` behind `OLLAMA_METRICS`, Ollama has no server metrics endpoint;
per-request timings are all it reports. vLLM's metrics are what lesson 18's
production-stack scrapes.

**The walker says nothing about any of the three.** Of its 100 outputs, 0
mention quantization, GGUF, safetensors, metrics or tracing. Ollama and vLLM
are both reachable by changing scale only on Hopper. On CPU and Apple Silicon
no scale reaches vLLM, and on AMD and Blackwell no scale reaches Ollama. The
laptop-to-server move is therefore also a hardware change, and the walker takes
only one hardware input.

**The staging tier cannot "mirror production quantization".** The lesson says
that about llama.cpp staging, and also says llama.cpp takes GGUF while the GPU
engines take safetensors. Staging never tests the file that ships.

### 5 — the walker picks SGLang, but the stack is vLLM's, and its LMCache is the slowest config

**Pick vLLM with prefix caching, lesson 11's cache-aware router and lesson 18's
LMCache.** Given the exercise's own words, the walker says SGLang on Hopper and
AMD, only because the string contains "prefix". Drop that word and it says
vLLM. On Blackwell it says TRT-LLM either way. The stack the exercise names,
though, is vLLM's: lesson 18 is the vLLM production-stack and its Connector
API, and lesson 11 calls RadixAttention "the intra-replica equivalent", with
cross-replica routing "strictly upstream". So the stack chooses the engine.

**In lesson 18's own simulator, LMCache is the slowest config.** Lesson 18's
eviction is hash-seeded, so the native row was run under PYTHONHASHSEED 0–4.

| config | total ms (shipped mix) | total ms (all 8K) |
|---|---:|---:|
| CPU offload | 369,402 | 372,633 |
| native | 375,033–376,133 | 382,433–383,833 |
| LMCache | **380,833** | **394,233** |

LMCache avoids the most re-prefills, 194, and still spends the most prefill
time, 14,800 ms. An 8K hit costs 500 blocks × 3 ms = 1500 ms, while
re-prefilling the same 8K costs 200 ms. It would break even at 0.4 ms per block,
7.5x cheaper than the code sets it. `hbm_capacity_blocks_per_engine` is assigned
and never read, so the "KV exceeds HBM" case that lesson 18 says LMCache exists
for cannot happen in the simulator. Decode is 97.3–97.6% of every run, so no
prefill cache in this model can move the total by more than 2.7%.

**Cache-aware routing lifts the hit rate but not the P99.** In lesson 11,
across hash seeds, REGIONAL routing hits 0.301–0.312 of requests and GLOBAL
0.863–0.892. GLOBAL gets there by sending 565–603 of 1000 requests to another
region, which multi-tenant residency rules often forbid. Misses stay above 1%
under every strategy, so P99 TTFT is simply the miss: 800 ms, or 3200 ms with
the miss scaled linearly to 8K. For a P99 target, what helps is keeping
tenants' shared prefixes resident: vLLM's prefix cache and LMCache with a
realistic fetch cost. Better routing does not.
