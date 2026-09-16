<!-- generated:start -->
# 10-llms-from-scratch / 20-deepseek-v3-walkthrough

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/10-llms-from-scratch/20-deepseek-v3-walkthrough/) · upstream spec
`phases/10-llms-from-scratch/20-deepseek-v3-walkthrough/docs/en.md`

```bash
uv run demo practice run 20-deepseek-v3-walkthrough --ex 1
uv run demo explain 20-deepseek-v3-walkthrough --ex 1
uv run pytest demos/phases/10-llms-from-scratch/20-deepseek-v3-walkthrough
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Compare the calculator's total-parameter estimate to the published 671B a… | code | T0 | `ex01_the_head_dim_is_fifty_six_and_should_be_128.py` |
| 2 | Modify the config to use MLA rank 256 instead of 512. Compute the resulting KV cache size at… | code | T0 | `ex02_half_the_cache_for_five_hundredths_of_a_percent.py` |
| 3 | Compare DeepSeek-V3's (256 experts, top-8) routing to a hypothetical (512 experts, top-8) var… | code | T0 | `ex03_the_extra_experts_are_memory_and_nothing_else.py` |
| 4 | Read Section 2.1 of the DeepSeek-V3 technical report (arXiv:2412.19437) on MLA. Explain in th… | code | T0 | `ex04_absorption_costs_parameters_and_saves_the_cache.py` |
| 5 | DeepSeek-V3 uses FP8 training for most operations. Compute the memory savings of FP8 vs BF16… | code | T0 | `ex05_the_weights_were_never_the_constraint.py` |
<!-- generated:end -->

## Answers

`code/main.py` is a parameter and KV-cache calculator for one model: MLA, 256
experts with top-8 routing, 61 layers, an MTP module, and a totals report. Four
of the five exercises are arithmetic on that calculator, and the first one finds
that a single line of it — `head_dim = hidden_size // num_attention_heads` —
accounts for almost the whole gap to the published figures.

All five are **T0** on **no** dependency group (stdlib only).

> Exercise 4 is scaffolded as a prose item ("read Section 2.1, explain in three
> sentences"). It is built here as **arithmetic on `mla_attention_params`**,
> because "inference-time efficiency" is a quantity and the function that counts
> MLA's matrices is where to count it.

### 1 — the head dim is fifty-six and should be 128

```text
active as computed             30.36 B
+ attention at real head dims  + 6.26 B     (187.1M/layer vs the counted 84.4M)
+ the untied LM head           + 0.93 B
                               ─────────
                                37.55 B     against a published 37 B
```

**ANSWER: 664.5B against 671B total, 30.36B against 37B active** — and
`head_dim = 7168 // 128 = 56` explains the second gap almost exactly. DeepSeek's
published widths are **128** for V and **128+64** for QK.

**FINDING: the total's delta is a different pair of terms.** The MTP block is
counted with a *dense* MLP (0.705B); with the MoE structure DeepSeek uses it is
**11.63B**. Adding that and the untied head gives 676.39B — overshooting, because
the 14B the docstring cites exceeds what the MoE arithmetic gives.

**MECHANISM: `hidden // n_heads` is an MHA identity and MLA is not MHA.** In MLA
the heads read from a latent and their width is a free hyperparameter. The line
assumes the constraint MLA exists to remove.

**FINDING: the embedding is counted once** and `DEEPSEEK_V3` has no
`tie_word_embeddings` key.

### 2 — half the cache for five hundredths of a percent

| rank | KV at 128k | total params | attention/layer |
|---:|---:|---:|---:|
| 512 | 7.62 GB | 664.54B | 84.4M |
| 256 | **3.81 GB** | 664.20B | 78.9M |
| 128 | 1.91 GB | 664.04B | 76.2M |

**ANSWER: exactly 50.0% of the cache for 0.05% of the parameters.**
`kv_cache_bytes` is linear in the rank and in nothing else.

**MECHANISM: the cost is a rank bound.** All 128 heads' keys live in a subspace
of dimension ≤ the rank — 14× compression at 512, 28× at 256.

**FINDING: the ratio of the two percentages is 989 to 1** — cache is per-token,
parameters are one-time.

**FINDING: the GQA reference the lesson prints is not the same model.**
`kv_heads_hypothetical = 8` is hard-coded; `DEEPSEEK_V3`'s
`num_key_value_heads` is **128**.

### 3 — the extra experts are memory and nothing else

| experts | total | active | active ratio | router/layer |
|---:|---:|---:|---:|---:|
| 256 | 664.5B | 30.36B | 4.57% | 1.84M |
| 512 | **1318.6B** | 30.47B | 2.31% | 3.67M |
| 1024 | 2626.6B | 30.68B | 1.17% | 7.34M |

**ANSWER: the total grows 1.98× and the active count moves 0.4%** — all of it
router.

**MECHANISM: storage is what the capacity buys and what it costs.** +654B
parameters, 0.6 TB at FP8, at unchanged FLOPs per token.

**FINDING: the router scales with the expert count and sits on the active
path** — 425.7M at 1024 experts, 1.4% of the active total, to choose 8 of 1024.

**FINDING: the active ratio measures how many experts were bought**, not how
selectively they are used.

### 4 — absorption costs parameters and saves the cache

```text
materialised:  W_q_up 11.01M + W_k_up 3.67M           =  14.68M per layer
absorbed:      q_lora × kv_lora × n_heads              = 100.66M   ← 6.9× MORE
cache:         kv_lora 512 vs n_heads×head_dim 7168    =     14×   ← the actual saving
```

**ANSWER: the absorbed matrix is 6.9× bigger, and that is not the point.**

**MECHANISM: the per-head keys are never materialised**, so the cache holds 512
numbers per token per layer instead of 7168 — 7.62 GB against 213.50 at 128k.

**FINDING: the trade turns on `q_lora·kv_lora < head_dim·(q_lora+kv_lora)`** —
786,432 against 114,688 here, so the parameters lose. Parameters are paid once
per model; the cache once per token per sequence in flight.

**FINDING: the absorbed form is never the one the counter counts.**
`mla_attention_params` sums the *training* form.

### 5 — the weights were never the constraint

| precision | weights | per GPU (2,048) |
|---|---:|---:|
| BF16 | 1,238 GB | 0.60 GB |
| FP8 | **619 GB** | **0.30 GB** |

**ANSWER: 619 GB saved is 0.30 GB per card.** What FP8 buys on a 2,048-GPU
cluster is halved link bandwidth and halved activation and gradient traffic —
none of which the calculator models.

**MECHANISM: the token budget meets the precision through compute.** 6ND at
30.36B active over 14.8T tokens is 2.70e24 FLOPs, implying **27.2% MFU** against
the published 2.788M H800-hours. FP8 moves the denominator, not the numerator.

**FINDING: the calculator has no dtype anywhere** — `kv_cache_bytes` hard-codes
`× 2`.

**FINDING: each parameter sees 22.3 tokens, or 487 if it is active**, against a
Chinchilla-optimal 20. That is the sparse model's actual bargain, and the
exercise's framing does not reach it.
