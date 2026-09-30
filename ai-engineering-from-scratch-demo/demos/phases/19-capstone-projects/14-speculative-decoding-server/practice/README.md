<!-- generated:start -->
# 19-capstone-projects / 14-speculative-decoding-server

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/14-speculative-decoding-server/) · upstream spec
`phases/19-capstone-projects/14-speculative-decoding-server/docs/en.md`

```bash
uv run demo practice run 14-speculative-decoding-server --ex 1
uv run demo explain 14-speculative-decoding-server --ex 1
uv run pytest demos/phases/19-capstone-projects/14-speculative-decoding-server
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Measure acceptance-rate degradation when the draft is one version behind the target (e.g., Ll… | code | T0 | `ex01_a_drift_that_changes_a_third_of_greedy_tokens_moves_the_lessons_acceptance_0_003.py` |
| 2 | Implement ngram-fallback: if EAGLE-3 acceptance drops below a threshold, switch to ngram draf… | code | T0 | `ex02_on_repetitive_traffic_the_fallback_lifts_reliable_blocks_from_34_to_92pct_and_on_the_lessons_traffic_it_hurts.py` |
| 3 | Run a controlled MoE experiment: same Qwen3-Coder-30B with routing noise injected vs without.… | code | T0 | `ex03_routing_noise_of_0_1_swaps_experts_on_64pct_of_tokens_and_costs_0_15_greedy_acceptance.py` |
| 4 | Extend to H200 (141 GB). Report the model-size-per-replica headroom gained and whether you ca… | code | T0 | `ex04_bf16_llama_3_3_70b_fits_the_h200_card_but_not_vllms_default_budget_and_the_h200_adds_60_gb.py` |
| 5 | Benchmark TensorRT-LLM speculative decoding on the same H100 hardware. Report where it wins v… | code | T0 | `ex05_tensorrt_llms_3_1x_lead_at_temperature_1_is_greedy_text_and_from_batch_128_it_loses_to_vllm.py` |
<!-- generated:end -->

## Answers

The lesson's `code/main.py` is a toy draft/verify scheduler. The vocabulary
has 10 tokens, each position has its own random distribution, and the draft
emits the target's argmax with probability `alignment`. The verify step
accepts any token whose probability is at least half the maximum. Exercises
1-3 and 5 run this scheduler with a changed target, draft or verify rule.
Exercise 4 is memory arithmetic sized from the lesson's claims. External
facts were read on 2026-09-29: the Llama 3.3 70B and Qwen3-Coder-30B-A3B
configs, nvidia-smi capacities, vLLM and TensorRT-LLM docs, NVIDIA's H100
page, the vLLM v0.7.0 release and arXiv:2503.01840. Each solution cites the
URLs it used.

### 1 — a drift that changes a third of greedy tokens moves the lesson's acceptance by 0.003

**Acceptance degrades as shown below, and the alert has to watch greedy
agreement.** "v3.4" is v3.3 with each position's weights multiplied by
exp(sigma x N(0, 1)). The draft stays aligned to v3.3, with alignment 0.9
and k = 4.

| sigma | argmax kept | greedy acceptance | lesson acceptance | tok/call |
|---:|---:|---:|---:|---:|
| 0 | 1.000 | 0.919 | 0.941 | 4.76 |
| 0.1 | 0.674 | 0.625 | 0.938 | 4.75 |
| 0.25 | 0.464 | 0.435 | 0.876 | 4.50 |
| 0.5 | 0.328 | 0.311 | 0.474 | 2.90 |
| 1.0 | 0.229 | 0.221 | 0.175 | 1.70 |

The alert calibrates on 100 target calls and fires when a 50-call rolling
mean falls below 90% of baseline. On greedy agreement it fires 16 calls
after the upgrade at sigma = 0.1 and 8 calls after at larger gaps, with 0
false alarms in 20 quiet streams.

**The lesson's metric cannot see a small version gap.** At sigma = 0.1 the
argmax changes on a third of positions, but acceptance only moves from 0.941
to 0.938. The same alert on that metric never fires at 0.1, needs 41 calls
at 0.25, and raises 3 false alarms in 20 quiet streams. Its rule passes 55%
of all tokens. The accounting also cannot show the skill file's "Drifted
drafts cost more than no speculation". A draft that always proposes the
least likely token still gets 1.001 tokens per call, because draft cost is
zero.

### 2 — on repetitive traffic the fallback lifts reliable blocks from 34% to 92%, and on the lesson's traffic it hurts

**On code-like traffic the fallback improves reliability from 0.343 to
0.923.** Reliability here is the share of 20-call blocks after the upgrade
that average at least 3.0 tokens per call. The traffic is a sharpened target
that repeats every 12 positions. The target is upgraded (sigma 1.0) at token
2,000. The fallback switches for good when the draft's 20-call acceptance
falls below 0.5, and here it switches 90 tokens after the upgrade.

| after the upgrade | EAGLE only | ngram only | fallback |
|---|---:|---:|---:|
| code: tok/call | 2.85 | 3.34 | 3.75 |
| code: reliable blocks | 0.343 | 0.724 | 0.923 |
| chat (the lesson's target): tok/call | 1.72 | 1.28 | 1.28 |

On healthy streams it never trips, so it matches EAGLE exactly.

**On the lesson's own traffic the fallback makes things worse.** Each
position there is an independent random distribution, so prompt lookup has
nothing to copy. Healthy ngram gets 2.13 tokens per call, below a draft that
proposes uniform random tokens (2.16). A threshold on EAGLE acceptance alone
cannot tell whether ngram will help. The fallback needs ngram's own
acceptance on the same traffic.

### 3 — routing noise of 0.1 swaps experts on 64% of tokens and costs 0.15 greedy acceptance

**Greedy acceptance falls 0.152 by sigma = 0.1, a slope of -1.52 per unit
of router-logit noise.** The target is one routed layer shaped like
Qwen3-Coder-30B-A3B: 128 experts, top 8, and renormalised gates. It gets
fresh N(0, sigma) noise on every call. The draft is aligned to the
noise-free router.

| sigma | expert set changed | argmax kept | greedy acc | lesson acc | tok/call |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 1.000 | 0.920 | 0.924 | 4.69 |
| 0.05 | 0.378 | 0.914 | 0.843 | 0.907 | 4.62 |
| 0.1 | 0.636 | 0.832 | 0.768 | 0.899 | 4.60 |
| 0.2 | 0.879 | 0.729 | 0.675 | 0.899 | 4.60 |
| 0.5 | 0.993 | 0.509 | 0.476 | 0.776 | 4.10 |
| 1.0 | 1.000 | 0.306 | 0.291 | 0.495 | 2.98 |

The sensitivity comes from the top-8 boundary. The gap between the 8th and
9th router logit has a median of 0.044, so even tiny noise swaps experts.
**The lesson's metric reads the same noise as 6x weaker** (slope -0.25). It
shows 0.899 at both 0.1 and 0.2.

### 4 — BF16 Llama 3.3 70B fits the H200 card but not vLLM's default budget, and the H200 adds 60 GB

**The H200 adds 60.01 GB per replica. Unquantized Llama 3.3 70B fits the
card but not vLLM's default budget.** The config gives 70,553,706,496
parameters: 141.11 GB in BF16 and 72.66 GB in FP8. At vLLM's default
`--gpu-memory-utilization` of 0.92:

| | H100 (81,559 MiB) | H200 (143,771 MiB) |
|---|---:|---:|
| vLLM budget | 78.68 GB | 138.69 GB |
| FP8 KV pool | 6.02 GB / 18,380 tokens | 66.04 GB / 201,532 tokens |
| BF16 weights | do not fit | 2.41 GB over budget |

BF16 needs a utilisation of at least 0.936 before any KV cache. At 0.95 the
pool holds 6,438 tokens, and at 0.98 it holds 20,240. So one H200 can run
BF16 only as a small-batch replica. For batch serving, BF16 still means two
GPUs.

**"141 GB" is closer to GiB.** The card is 150.75 GB, or 140.4 GiB. Read
literally, 141 GB would reject weights that actually fit with 9.64 GB to
spare. **The lesson's 1xH100 FP8 plan cannot hold its own batch-32 report.**
Its pool gives 574 tokens per sequence at batch 32, below the 620-token
response in its Use It block.

### 5 — TensorRT-LLM's 3.1x lead at temperature 1 is greedy text, and from batch 128 it loses to vLLM

**The algorithm shows nowhere that TensorRT-LLM wins, so its wins have to
be kernel time, which needs the H100 run.** The commands are in `COMMANDS`.
TensorRT-LLM's documented rule is greedy only, with no way to switch
speculation off. vLLM's is lossless rejection sampling, with k chosen per
batch size. At temperature 0 the two rules coincide. At temperature 1
TensorRT-LLM emits 4.28 tokens per call at k = 4 against vLLM's 1.38, but it
emits greedy text: 0.813 in total variation from the requested distribution,
against 0.007 for vLLM. On an H100 FP8 roofline:

| batch | 1-32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|
| TensorRT-LLM, fixed k = 4 | 4.28x | 4.07x | 2.04x | 1.02x | 0.86x |
| vLLM, best k per batch | 4.28x | 4.07x | 2.21x | 1.14x | 1.00x |

These are upper bounds. Real engine efficiency moves the crossover to a
smaller batch.

**The lesson's own verify rule is not lossless either.** It is 0.736 in
total variation from the target and scores 4.756 tokens per call, where the
lossless rule gets 1.378. Its printed speedups come from not sampling the
target. **"EAGLE-3 in vLLM 0.7" predates EAGLE-3.** vLLM v0.7.0 shipped on
2025-01-27, and the EAGLE-3 paper was submitted on 2025-03-03.
