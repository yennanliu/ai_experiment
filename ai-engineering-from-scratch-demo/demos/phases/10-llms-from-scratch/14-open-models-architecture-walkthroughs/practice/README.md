<!-- generated:start -->
# 10-llms-from-scratch / 14-open-models-architecture-walkthroughs

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/10-llms-from-scratch/14-open-models-architecture-walkthroughs/) · upstream spec
`phases/10-llms-from-scratch/14-open-models-architecture-walkthroughs/docs/en.md`

```bash
uv run demo practice run 14-open-models-architecture-walkthroughs --ex 1
uv run demo explain 14-open-models-architecture-walkthroughs --ex 1
uv run pytest demos/phases/10-llms-from-scratch/14-open-models-architecture-walkthroughs
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Read the Qwen 2.5 72B config from HuggingFace. Compute total parameters from scratch. Compare… | code | T0 | `ex01_the_delta_is_the_untied_head_to_three_decimals.py` |
| 2 | DeepSeek V3 uses 256 experts with top-8 routing. Compute the ratio of activated experts to to… | code | T0 | `ex02_the_expert_share_is_not_the_parameter_share.py` |
| 3 | Compute the KV cache for Llama 3 405B at 128k context in FP8 and BF16. At FP8 it is half the… | code | T0 | `ex03_the_weights_do_not_fit_before_the_cache_is_counted.py` |
| 4 | Gemma 2 alternates full-attention and sliding-window-attention layers. Write the math for the… | code | T0 | `ex04_the_window_can_never_save_more_than_half.py` |
| 5 | Find a recent frontier open model that was released after this lesson was written. Identify w… | code | T0 | `ex05_the_seventh_knob_is_already_in_the_table.py` |
<!-- generated:end -->

## Answers

`code/main.py` is a parameter counter: seven model configs, a from-scratch
breakdown of attention, MLP, norm and embedding params, a KV-cache size, and a
six-knob "verdict" string. Four of the five exercises turn out to be arithmetic
the counter already does; the fifth asks whether the table has a column missing,
and it does.

All five are **T0** on **no** dependency group (stdlib only).

### 1 — the delta is the untied head, to three decimals

```text
analyze                       71.4597 B
+ untied LM head (V × h)     + 1.2457 B
+ q/k/v attention biases     + 0.0008 B
                             ──────────
                              72.7062 B   vs Qwen's reported 72.706 B   (0.0003%)
```

**ANSWER: the whole 1.246B gap is one term.** Qwen 2.5 72B does not tie its
embedding to its output head, and `analyze` counts `vocab_size × hidden_size`
once.

**FINDING: neither candidate the exercise names contributes anything.** Head dim
is `8192/64 = 128` exactly — zero rounding. The KV sharing factor is already in
`attention_params_per_layer`; turning GQA *off* would add **9.4B**, not 1.2B.

**MECHANISM: an untied head is a whole embedding table.** 1.246B parameters —
1.7% of the model, more than any single layer's 0.88B.

**FINDING: the same omission is worth 0.93B on DeepSeek V3**, closing 18% of its
665.9B-vs-671B gap; the rest is the MTP module the config does not describe.

### 2 — the expert share is not the parameter share

| | experts | activated | total | active | active share | capacity/FLOP |
|---|---:|---:|---:|---:|---:|---:|
| Mixtral 8x7B | 8 | top-2 = **25.0%** | 46.6B | 12.7B | **27.4%** | 3.7× |
| DeepSeek V3 | 256 | top-8 = **3.1%** | 665.9B | 32.4B | **4.9%** | **20.6×** |

**ANSWER: an 8× shift in experts is a 5.6× shift in parameters.** The expert
ratio overstates the sparsity, because attention, embeddings, norms and the
shared expert are active on every token whatever the router does.

**FINDING: the expert got smaller, which is how the ratio moved.**
`moe_intermediate_size` is **2048** against Mixtral's 14336 — so 256 DeepSeek
experts is 8.0× Mixtral's 8 in parameters, not 32×.

**FINDING: the floor is 11.9B — 37% of DeepSeek's active parameters** — before
one routed expert is consulted. No routing decision can reduce it.

### 3 — the weights do not fit before the cache is counted

```text
405B at BF16   752.0 GB of a 640 GB node  →  −112.0 GB left
405B at FP8    376.0 GB of a 640 GB node  →   264.0 GB left
                                              ÷ 63.0 GB/seq (BF16 KV)  =  4 sequences
                                              ÷ 31.5 GB/seq (FP8 KV)   =  8 sequences
```

**ANSWER: at BF16 weights the subtraction the exercise asks for goes negative.**
At FP8 weights the node serves 8 sequences at 128k.

**ANSWER: FP8 KV is exactly half of BF16** — `2 × layers × kv_heads × head_dim ×
seq × bytes` is linear in `bytes`.

**MECHANISM: 504 KB per token is a GQA number.** With 128 KV heads instead of 8,
a single 128k sequence would need **1,008 GB**.

### 4 — the window can never save more than half

| context | all-full | alternating | saving |
|---:|---:|---:|---:|
| 2k / 4k | — | — | **0.0%** |
| 8k | 2.88 GB | 2.16 GB | **25.0%** |
| 32k | 11.50 GB | 6.47 GB | 43.8% |
| 128k | 46.00 GB | 23.72 GB | 48.4% |

**ANSWER: 25.0% at 8k, exactly.** `(8192 + 4096) / (2 × 8192) = 0.75`.

**FINDING: 50% is the ceiling, approached and never reached.** Half the layers
have no window, so their cache grows with context forever.

**MECHANISM: the saving is `(1 − window/context) / 2`** — zero whenever the
context fits inside the window. 8k is the smallest context at which the answer
is not zero.

**FINDING: the window is the second halving of the same tensor.** Gemma 2 27B is
already GQA (32/16), so 25% of its cache is 0.72 GB where 25% of an MHA model's
would be 1.44 GB.

### 5 — the seventh knob is already in the table

```text
verdict for deepseek-v3:  RMSNORM · SWIGLU · ROPE · MLA · MoE 256e/top-8
                          ^ five flags, and shared_experts is in none of them
```

**ANSWER: the missing knob is worth 2.554B active parameters — 7.9%.**
`shared_experts: 1` is in the config, is read by `analyze`, changes both
`total_params` and `active_params`, and appears in no flag.

**FINDING: a shared expert is a different *kind* of knob.** The six are
substitutions (LayerNorm *or* RMSNorm, dense *or* MoE); a shared expert is an
addition, making dense-vs-sparse a spectrum. DeepSeek V3 is both in every MoE
layer.

**FINDING: norm and activation are settled** across every config that postdates
GPT-2 — all RMSNorm, all SwiGLU. Only attention sharing and dense-vs-MoE still
discriminate.

**MECHANISM: what makes a knob a knob is that `analyze` reads it.** Switching
`position` from `rope` to `learned` leaves the total bit-for-bit unchanged — the
third knob in the table is invisible to the counter that scores the other five.
