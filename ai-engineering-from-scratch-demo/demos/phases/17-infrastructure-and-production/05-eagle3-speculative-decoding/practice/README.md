<!-- generated:start -->
# 17-infrastructure-and-production / 05-eagle3-speculative-decoding

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/17-infrastructure-and-production/05-eagle3-speculative-decoding/) · upstream spec
`phases/17-infrastructure-and-production/05-eagle3-speculative-decoding/docs/en.md`

```bash
uv run demo practice run 05-eagle3-speculative-decoding --ex 1
uv run demo explain 05-eagle3-speculative-decoding --ex 1
uv run pytest demos/phases/17-infrastructure-and-production/05-eagle3-speculative-decoding
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. At K=5, what alpha do you need for a 2x speedup? For a 3x speedup? How se… | code | T0 | `ex01_the_formula_asks_alpha_0_26_for_2x_and_its_own_simulator_asks_0_78.py` |
| 2 | Imagine production traffic splits 70% general chat, 30% code. General chat hits alpha 0.7 wit… | code | T0 | `ex02_blended_alpha_0_61_passes_the_gate_while_the_code_slice_runs_slower_than_plain.py` |
| 3 | Read the vLLM `speculative_config` documentation. Name the three modes (draft model, EAGLE, N… | explain | T0 | prose, below |
| 4 | You see mean ITL drop 25% after enabling EAGLE-3 but P99 ITL went up 15%. Diagnose and propos… | code | T0 | `ex04_a_p99_up_15_percent_is_one_verify_overhead_on_steps_that_accept_nothing.py` |
| 5 | Compute the memory cost of the EAGLE-3 draft head for Llama 3.3 70B. How does it compare to r… | code | T0 | `ex05_the_eagle3_head_for_70b_is_bigger_than_llama_3_2_1b_but_reads_less_per_step.py` |
<!-- generated:end -->

## Answers

The lesson's `code/main.py` is a toy analyzer with two parts: a closed-form
`expected_speedup` and a seeded latency simulator, `simulate_tail`. The
exercises show the two disagree. Exercises 1, 2 and 4 use the simulator's
exact expectation, which matches `simulate_tail` to within 0.2% at 200,000
tokens. The vLLM docs and Hugging Face configs were read on 2026-09-26.

### 1 — the formula asks alpha 0.26 for 2x, and its own simulator asks 0.78

**At K=5 and verify_overhead 0.15, the lesson's formula needs alpha 0.26 for
2x and 0.49 for 3x.** `(1 + K*alpha) / (1 + eps)` is linear, so the
sensitivity is exact: every 0.1 of extra overhead costs 0.04 alpha for 2x and
0.06 for 3x.

| verify_overhead | 2x, lesson formula | 3x, lesson formula | 2x, stop at first rejection | 3x, stop at first rejection |
|---:|---:|---:|---:|---:|
| 0.0 | 0.20 | 0.40 | 0.509 | 0.709 |
| 0.15 | 0.26 | 0.49 | 0.582 | 0.771 |
| 0.3 | 0.32 | 0.58 | 0.642 | 0.824 |
| 0.5 | 0.40 | 0.70 | 0.709 | 0.883 |

The formula counts every draft token as independently accepted.
`simulate_tail` stops at the first rejection, and so does a real verifier.
That gives the right-hand columns. The simulator also charges a second pass
on every rejection. With that charge, at 32 concurrent it needs alpha 0.511
to break even, 0.781 for 2x and 0.889 for 3x. At alpha 0.3 the printed
column says 2.14x while the simulator runs at 0.70x.

The printed break-even alphas are 0.034, 0.045 and 0.060. The KEY FINDING's
"~0.4 at 256" comes from none of them; the formula reaches 0.4 only at
concurrency 3157. The simulator's own break-even *falls* with concurrency,
from 0.511 at 32 to 0.348 at 256, because plain decode's cost grows faster
than a spec step's.

### 2 — blended alpha 0.61 passes the gate while the code slice runs slower than plain

**Blended alpha is 0.7 × 0.7 + 0.3 × 0.4 = 0.61, and by the lesson's rules
spec decode is net-positive.** 0.61 clears the 0.55 gate, and the formula
gives 3.47x at 32 concurrent. It rates even code alone at 2.57x.

The simulator's cost model disagrees, blending by time rather than by alpha:

| 32 concurrent | speedup |
|---|---:|
| chat slice (alpha 0.7) | 1.56x |
| code slice (alpha 0.4) | **0.82x** |
| whole mix, spec on | 1.23x |
| one blended alpha 0.61 | 1.24x |
| code routed to plain decode | **1.34x** |

So the mix is net-positive, but code loses and routing it off wins. At 256
concurrent code reaches 1.09x, and spec on everything wins, 1.63x to 1.57x.

The alpha you would log is a different quantity from the alpha in the
formula. vLLM's `draft_acceptance_rate` is accepted over proposed drafts,
and the lesson's recipe (accepted tokens divided by K) is the same quantity.
Per-position alpha 0.7 logs as 0.388, 0.4 as 0.132, and the mix as 0.278.
Gated at 0.55, the logged number would switch off a config the formula
rates at 3.47x.

### 3 — the docs name many methods and mark speculative decoding as a whole compatible with chunked prefill

*Draws on "Where EAGLE-3 is already deployed".*

`docs.vllm.ai/en/latest/features/spec_decode/` now redirects, so I read the
docs source in the vLLM repository instead: `docs/features/speculative_decoding/`
on `main` and at tag `v0.18.0`.

**The three modes, as `speculative_config` spells them:**

- **Draft model:** `"method": "draft_model"`, with `model` set to the smaller
  model and `num_speculative_tokens`.
- **EAGLE:** `"method": "eagle"` or `"eagle3"`, with `model` set to the head.
  The docs' example is `RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3`.
- **N-gram:** `"method": "ngram"`, with `prompt_lookup_min` / `prompt_lookup_max`
  (default 5 when both are omitted) and no model. The v0.18.0
  `SpeculativeMethod` literal also has `ngram_gpu`.

The docs list more than three. The index names EAGLE, MTP, draft model,
PARD, MLP, n-gram and suffix decoding. `main` adds LiLiCorr, hidden-state
extraction, a custom proposer, dynamic speculative decoding and adaptive
verification. The literal also carries `medusa`, `mlp_speculator` and
`extract_hidden_states`.

**Chunked prefill: the docs single out no method.** No speculative-decoding
page, on `main` or at v0.18.0, mentions chunked prefill. The feature
compatibility matrix (`docs/features/README.md`) marks SD × chunked prefill
✅ at v0.8.0, v0.10.0, v0.18.0 and `main`. It rates speculative decoding as a
whole, not per method. The documented incompatibilities are these two:

- pipeline parallelism (`vllm<=0.15.0`);
- draft models, unsupported in `vllm<=0.10.0`.

I also grepped v0.18.0's `engine/arg_utils.py`, `config/vllm.py`,
`config/scheduler.py`, `config/speculative.py` and
`v1/spec_decode/draft_model.py` and found no check that combines chunked
prefill with a spec method.

So I could not confirm two lesson claims: that N-gram GPU is "the variant
compatible with chunked prefill", and that v0.18.0 draft-model spec decode
with `--enable-chunked-prefill` "does not compile". I did not run vLLM on a
GPU to test either one.

Two smaller mismatches:

- **The metric name.** The lesson's `spec_decode_metrics.accepted_tokens_per_request`
  does not appear in these docs. vLLM exposes
  `vllm:spec_decode_num_accepted_tokens_total` and
  `vllm:spec_decode_num_draft_tokens_total`, plus an experimental per-request
  `metrics.speculative_decoding.draft_acceptance_rate`.
- **The positioning.** The docs pitch spec decode for "medium-to-low QPS ...
  memory-bound workloads". They also document per-batch-size K, which can be
  0, for high concurrency.

### 4 — a P99 up 15% is one verify overhead on the steps that accept nothing

**Diagnosis:** a verify always emits one token of its own; vLLM's mean
acceptance length "ranges from 1.0 (nothing accepted)". So a step that
accepts nothing costs 1 + eps = 1.15 plain steps for one token. That is
exactly the +15%.

A mean ITL down 25% means 1.15 / E = 0.75. So E = 1.533 tokens per step and
alpha is 0.349. At that alpha, 42% of tokens come from zero-acceptance
steps, far more than the 1% it takes to own the P99. The draft is simply
wrong at the first position most of the time. Low alpha on this traffic is
the cause, not a scheduler bug.

| mitigation | mean ITL | P99 ITL |
|---|---:|---:|
| as observed (K=5, alpha 0.349) | -25% | +15% |
| lower K to 1 | -14.8% | +15% |
| raise alpha to 0.947 (domain-trained head) | -78.2% | -42.5% |
| K=0 for this traffic | 0% | 0% |

Lowering K does not help in this cost model; it helps only where verify
overhead grows with K. **Mitigation:** turn spec off where alpha is low
until a domain head brings it past about 0.95. That means K=0 for the
affected batch-size range through vLLM's `num_speculative_tokens_per_batch_size`,
or plain decode for the low-alpha slice (exercise 2).

The lesson's simulator cannot show this symptom at its own concurrency
points. It charges a second full pass (`reroll_ms`) on every rejected step,
so a zero-acceptance step costs 2.05 plain steps per token. Its P99 is
+35%..+116% over alpha 0.3..0.9 at concurrency 1, 32 and 256, and all 15 rows
`main()` prints are tagged TAIL. The tail also shrinks as concurrency grows:
alpha 0.7 goes from +98.5% at 1 to +13.3% at 512 and -15.2% at 1024. That is
the opposite of the lesson's warning.

### 5 — the EAGLE-3 head for 70B is bigger than Llama 3.2 1B, but reads less per step

The head is `yuhuili/EAGLE3-LLaMA3.3-Instruct-70B`. Its config and the
per-tensor byte counts of its checkpoint were read from Hugging Face.

| | EAGLE-3 head | Llama 3.2 1B | Llama 3.3 70B |
|---|---:|---:|---:|
| parameters | 1.576B | 1.236B | 70.6B |
| weights (fp16/bf16) | 3.15 GB | 2.47 GB | 141 GB |
| streamed per draft step | 788M params | 1.236B params | — |
| KV per token | 4 KiB | 32 KiB | 320 KiB |
| KV at 256 × 8192 tokens | 8 GiB | 64 GiB | 640 GiB |

**By weight the head is 1.28x the 1B, and each is about 2% of the bf16
target.** The head has five parts:

- `fc`, which fuses three 8192-wide target layers into 6144: 151M;
- one decoder layer whose q/k/v read a 12288-wide input: 440M;
- a 32000-token draft lm_head: 197M;
- its own full-vocabulary 6144-wide embedding: 788M.

The config-derived sum matches the checkpoint's tensor bytes exactly.

Half the head is a lookup table, so per draft step it streams 0.64x the 1B's
weights, through one layer instead of 16. The largest difference is KV
cache: 8x less than the 1B, 1.25% of the target's against 10%. The lesson's
analyzer models none of this. `SpecPoint` has only alpha, k,
verify_overhead and concurrency.
