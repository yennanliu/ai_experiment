<!-- generated:start -->
# 17-infrastructure-and-production / 08-inference-metrics-goodput

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/17-infrastructure-and-production/08-inference-metrics-goodput/) · upstream spec
`phases/17-infrastructure-and-production/08-inference-metrics-goodput/docs/en.md`

```bash
uv run demo practice run 08-inference-metrics-goodput --ex 1
uv run demo explain 08-inference-metrics-goodput --ex 1
uv run pytest demos/phases/17-infrastructure-and-production/08-inference-metrics-goodput
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Generate a distribution with 1% tail spike. How does goodput change when… | code | T0 | `ex01_a_one_percent_token_spike_never_reaches_a_15ms_tpot_slo_so_goodput_does_not_move.py` |
| 2 | A vendor quotes "15,000 tok/s on Llama 3.3 70B H100". Name three questions to ask before trus… | code | T0 | `ex02_one_fp8_h100_reaches_15000_tok_s_only_if_every_sequence_holds_160_tokens_or_less.py` |
| 3 | Why does chunked prefill protect P99 TPOT but not mean TPOT? | code | T0 | `ex03_chunked_prefill_moves_the_stall_rather_than_removing_it.py` |
| 4 | Construct a consumer SLO for a voice assistant (first token is heard, not read). Which metric… | code | T0 | `ex04_for_voice_tpot_and_e2e_drop_out_and_time_to_first_phrase_is_the_slo.py` |
| 5 | Read the LLMPerf README and the GenAI-Perf docs. Identify three other metrics where the tools… | code | T0 | `ex05_the_tools_also_disagree_on_per_request_throughput_token_count_and_the_percentile_itself.py` |
<!-- generated:end -->

## Answers

All five are code: every answer is measured on the lesson's own
`synth_workload()` traces and `goodput()` / `percentiles()`. Sources for
exercises 2, 4 and 5 were read on 2026-09-26: LLMPerf's README and source
(`token_benchmark_ray.py`, the OpenAI client), the GenAI-Perf README, NVIDIA's
NIM metrics page, the H100 spec page, a public copy of Llama 3.3 70B's
`config.json`, and search-result summaries of the Llama 3 paper and of
Stivers et al. 2009.

### 1 — a 1% token spike never reaches a 15 ms TPOT SLO, so goodput does not move

**It does not change: 80.10% at 30 ms and 80.10% at 15 ms.** TTFT and E2E are
held at the target profile, 500 ms and 2000 ms. `tail_spike_rate` spikes
single *tokens*, but `goodput()` tests each request's *mean* TPOT, which
averages about 175 tokens:

| TPOT measured | value |
|---|---:|
| worst request | 9.18 ms |
| P99 request | 8.30 ms |
| P99 token | 18.17 ms |

The constraint first bites at 15 ms when the spike rate reaches 20%, where
TPOT-only goodput is 92.35%. At 5% it is still 100%.

**What sets goodput is E2E and TTFT.** On the shipped run, the target profile
fails 22.75% of requests on E2E, 2.30% on TTFT and 0% on TPOT. The spikes do
cost goodput, 84.45% at rate 0 against 80.10% at 1%, but through E2E.

**Spiking whole requests instead** models a long-prefill neighbour. 1% of
requests decode 3–8x slower, which hits 26 requests at TPOT 21.9–55.5 ms.
Tightening then costs 0.5 points alone (99.20% → 98.70%) and 0.1 points
jointly (83.30% → 83.20%), because E2E already fails 24 of the 26.

The module's printed KEY FINDING says "P99 TPOT ~25-40 ms" and "collapses
from 99%". Its own run prints 8.99 ms and 100.00% for the loose profile.

### 2 — one FP8 H100 reaches 15,000 tok/s only if every sequence holds 160 tokens or less

Three questions, each checked against arithmetic that could change the
answer.

1. **How many GPUs, at what precision?** In BF16 the weights are 141 GB and
   do not fit an 80 GB H100. In FP8 they leave 9.4 GB of KV cache, which is
   57,373 tokens at 160 KiB each. With the cache full, a memory-bound step
   reads 80 GB at 3.35 TB/s, 23.88 ms. 15,000 tok/s then needs 358
   sequences of **160 tokens or less** each. At the lesson workload's
   2,394-token mean context, one GPU gives 963 tok/s and an 8-GPU node
   60,761. So "H100" means a node, which is 1,875 tok/s per GPU.
2. **Input plus output, or output only?** The workload's mean prompt is
   2,218 tokens against 176 output tokens. Counting input tokens inflates the
   figure 13.6x.
3. **At what goodput, under which SLO?** The same 2000 requests score 100%,
   75.95% and 42.15% under the lesson's three profiles, and the throughput is
   the same in all three.

The lesson cannot compute its own throughput formula. `main.py` has no
throughput function, and `RequestTrace` carries no timestamps, so elapsed
time is undefined.

### 3 — chunked prefill moves the stall rather than removing it

A 7 ms decode step, with an 800 ms 32k-token prefill (the lesson's figure)
arriving every *period* steps. Chunking splits it into 16 × 50 ms pieces at
2048 tokens. Each figure is per-token ITL:

| period | prefill | mean | P90 | P99 |
|---:|---|---:|---:|---:|
| 50 | whole | 23.0 | 7 | 807 |
| 50 | chunked | 23.0 | 57 | 57 |
| 200 | whole | 11.0 | 7 | 7 |
| 200 | chunked | 11.0 | 7 | 57 |

**The mean cannot move, because the 800 ms of work is the same either way.**
Chunking only changes where that work lands. It cuts one huge stall into
many small ones: P99 falls, but P90 rises as 32% of steps carry a chunk.
The protection also only holds when the stall already reaches the P99. At
one long prompt per 200 steps the stall is 0.5% of steps, and chunking
*raises* P99 from 7 to 57 ms. The lesson's per-request TPOT cannot see any
of this. Its P99 is 25.29 ms both ways at period 50.

The lesson's own example claims a P99 of 65 ms for 500 × 7 ms + 20 × 60 ms
tokens. Its mean is 9.04 ms, as stated, but no token exceeds 60 ms, so the
P99 is 60. The code's 0.05 ms per prompt token would prefill 32k tokens in
1638 ms, not 800.

### 4 — for voice, TPOT and E2E drop out and time to first phrase is the SLO

**The SLO:**

- Time until the first 8-token phrase exists is at most 500 ms at P99.
- The playback buffer never underruns at 300 ms per token (150 wpm at 0.75
  words per token).

**The most user-visible metric is TTFT, at 91% of first-phrase time at P99.**
Human turn gaps are about 200 ms (Stivers et al., PNAS 2009), and ASR and
TTS spend part of that before the LLM's part starts.

On the lesson's workload, first-phrase time is 139/493/583 ms at
P50/P90/P99, so the deployment misses its P99 target. Voice goodput is
91.35%, against 75.95% for the text target profile on the same requests.
The text failures were E2E, which is not heard.

**TPOT and E2E drop out.** The slowest single token is 86.3 ms against
300 ms of speech, so no request underruns, and TPOT has 33x headroom. The
mean answer takes 52.8 s to speak.

**Where the SLO fails is long context.** Prompts up to 2048 tokens pass at
≥ 99.7%. 8192-token prompts pass 57.3%, with a first-phrase P99 of 641 ms.
A voice session grows its history every turn, so it drifts toward the
failing case.

### 5 — the tools also disagree on per-request throughput, token count and the percentile itself

The lesson's claim checks out in the source. LLMPerf appends TTFT as the
first gap, and NIM's ITL is `(e2e − TTFT) / (tokens − 1)`. Three more
disagreements, each measured on the lesson's traces:

| metric | LLMPerf | GenAI-Perf | measured |
|---|---|---|---|
| per-request throughput | `tokens / e2e` | (tokens − 1) / generation time | mean 117.3 vs 130.4 tok/s, up to 2.50x |
| output token count | `LlamaTokenizer` (`hf-internal-testing/llama-tokenizer`) for every model | the model's tokenizer | 3.94 / 3.17 chars per token → 1.243x on Llama 3 |
| percentiles | pandas linear, p25/50/75/90/95/99 | p99/p90/p75 in its table | P99 TTFT on 50 requests: 564.7 ms by the lesson's rule, 558.3 linear |

The lesson's `percentiles()` is a third rule, index `int(p·n)`. On 50
requests that makes its P99 the sample maximum.

Two more inconsistencies turned up:

- **LLMPerf's ITL and throughput use different token counts.** ITL is divided
  by the streamed-chunk count. `NUM_OUTPUT_TOKENS` is then overwritten with
  the tokenizer count before throughput is computed, so ITL × throughput is
  1.243 on Llama 3, not 1.
- **The lesson's trace has one decode gap too many.** It has N gaps for N
  tokens, so `tpot_genaiperf()` overstates the per-token mean by 0.055 ms.
