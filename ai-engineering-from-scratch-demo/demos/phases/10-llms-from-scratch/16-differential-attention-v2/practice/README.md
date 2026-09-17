<!-- generated:start -->
# 10-llms-from-scratch / 16-differential-attention-v2

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/10-llms-from-scratch/16-differential-attention-v2/) · upstream spec
`phases/10-llms-from-scratch/16-differential-attention-v2/docs/en.md`

```bash
uv run demo practice run 16-differential-attention-v2 --ex 1
uv run demo explain 16-differential-attention-v2 --ex 1
uv run pytest demos/phases/10-llms-from-scratch/16-differential-attention-v2
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Verify the signal-to-noise ratio reported for differential attention is h… | code | T0 | `ex01_the_crossover_runs_the_other_way.py` |
| 2 | Compute the parameter-count delta from baseline to DIFF V1 and from baseline to DIFF V2 for a… | code | T0 | `ex02_the_baseline_is_not_a_2025_model.py` |
| 3 | Read Section 3 of the DIFF V1 paper (arXiv:2410.05258) and Section 2 of the DIFF V2 Hugging F… | code | T0 | `ex03_the_output_scale_varies_five_fold_across_heads.py` |
| 4 | Implement an ablation: compute differential attention with `lambda = 0` (pure first softmax)… | code | T0 | `ex04_the_best_lambda_is_zero_where_it_matters.py` |
| 5 | Extend the toy to GQA + DIFF V2. Pick 8 KV heads and 32 Q heads. Show that the KV cache size… | code | T0 | `ex05_the_cache_matches_because_v2_never_touches_k_or_v.py` |
<!-- generated:end -->

## Answers

`code/main.py` is a stdlib toy for differential attention: two softmax branches,
a subtraction, a signal-to-noise metric, and a parameter counter for V1 and V2.
The five exercises find that the toy's headline result reverses one step outside
the range the lesson prints, and that its parameter comparison uses a baseline
no 2025 model has.

All five are **T0** on **no** dependency group (stdlib only).

> Exercise 3 is scaffolded as a prose item ("read the papers, explain in two
> sentences"). It is built here as a **measurement**, because the quantity both
> papers argue about — the magnitude a differential head hands to the residual
> stream — is computable from `diff_attention` directly.

### 1 — the crossover runs the other way

| noise | standard SNR | diff SNR (λ=0.8) | ratio |
|---:|---:|---:|---:|
| 0.25 | 53.00 | **182.45** | 3.44 |
| 0.50 | 48.37 | **91.27** | 1.89 |
| 1.00 | 33.41 | 34.49 | 1.03 |
| 1.50 | **18.24** | 13.75 | 0.75 |
| 2.00 | **7.78** | 4.75 | 0.61 |
| 3.00 | 0.68 | 0.38 | 0.57 |

**ANSWER: differential attention wins below noise 1.0 and loses above it.** The
exercise's first sentence holds at the noise level `main` prints and fails at
twice it.

**FINDING: standard attention is never unusable while differential attention is
usable.** Both collapse together; the ratio stays near 0.5.

**MECHANISM: the subtraction removes a floor softmax has already removed.** At
low noise `A1`'s noise weights are a flat ~`1/n` floor that `A2` also is. At
high noise they are exponentials of a wide Gaussian, and subtracting an
*independent* draw of the same kind adds variance.

**FINDING: the lesson's own sweep contains the crossing** (0.25 → 2.00) at one
seed per level, and never names it.

### 2 — the baseline is not a 2025 model

| component | baseline | DIFF V1 | DIFF V2 | |
|---|---:|---:|---:|---|
| Q | 16,777,216 | 16,777,216 | **33,554,432** | ×2 |
| K | 16,777,216 | 16,777,216 | **4,194,304** | ÷4 |
| V | 16,777,216 | 16,777,216 | **4,194,304** | ÷4 |
| O | 16,777,216 | 16,777,216 | **33,554,432** | ×2 |
| λ | 0 | 8,192 | 16,384 | |
| **total** | 67,108,864 | 67,117,056 | 75,513,856 | |
| **delta** | | **+8,192** (0.01%) | **+8,404,992** (12.5%) | |

**ANSWER: V1 is free.** Halving `d_head` and running two branches leaves Q and K
exactly where they were; the only new parameters are the λ vectors.

**FINDING: the baseline is multi-head, and V2 is not.** V2's K and V are already
GQA-sized, so the delta compares a GQA model against an MHA baseline. Against a
like-for-like GQA baseline (41.9M), V2 costs **+80%**, not +12.5% — the
comparison understates it **6.4×**.

**MECHANISM: V1 spends nothing and V2 spends 33.6M for the same two branches**
— the difference is the decision to stop halving the head.

### 3 — the output scale varies five-fold across heads

| branch correlation | ‖out‖ | vs standard |
|---:|---:|---:|
| 0.00 | 0.655 | 0.98× |
| 0.60 | 0.518 | 0.77× |
| 0.90 | 0.273 | 0.41× |
| 1.00 | 0.134 | **0.20×** |

**ANSWER: at a fixed λ = 0.8 the output norm ranges 4.9×**, depending on how
correlated a head's two branches are — through the same λ, into the same
residual stream. That table is the argument for V1's per-head RMSNorm.

**MECHANISM: the weights no longer sum to one.** `A1 − λA2` has row sum exactly
`1 − λ`; at λ=1 it is **0.000** and the head emits a pure contrast.

**FINDING: V2's removal is a width change.** V1 concatenates two half-width
branches back to exactly 4096 — the baseline's hidden size, so the output
projection cannot absorb anything. V2 hands it **8192** inputs.

**FINDING: λ cannot do the RMSNorm's job.** The only λ at which every head emits
the same magnitude is 0, where there is no differential attention left.

### 4 — the best lambda is zero where it matters

| noise | λ=0.0 | λ=0.5 | λ=0.8 | λ=1.0 | λ=1.2 | argmax |
|---:|---:|---:|---:|---:|---:|---|
| 0.5 | 48.37 | 83.84 | 91.27 | 84.42 | 72.87 | **0.75** (91.68) |
| 2.0 | **7.78** | 5.78 | 4.75 | 4.27 | 3.87 | **0.00** (7.78) |

**ANSWER: 0.75 at the lesson's noise, 0.00 at four times it.** The hard-coded
0.8 is within **0.4%** of optimal in the quiet case and **39%** off in the loud
one — where the optimising λ is the ablation's own "pure first softmax"
endpoint.

**FINDING: the curve is single-peaked and the peak walks left as noise rises.**

**MECHANISM: λ trades a shrinking signal against a shrinking noise floor.** When
the second branch matches badly, the signal falls faster.

**FINDING: past λ=1 the weights go negative and `snr` keeps reporting**, because
it takes `abs(weights[pos])`. At λ=1.2 it returns 72.87 for a row summing to
**−0.20**.

### 5 — the cache matches because V2 never touches K or V

**ANSWER: 4,096 bytes per token per layer, both ways, exactly.** `2 × 8 KV heads
× 128 dims × 2 bytes` — and `attention_params_diff_v2` already sizes K and V at
`kv_heads × head_dim`.

**FINDING: the exercise asks to confirm the one thing V2 left alone.** Q and O
are **doubled** (33.5M against 16.8M each); K and V are untouched. The 1,074M
parameters that changed are not mentioned.

**MECHANISM: the two branches share K and V, so there is one cache.** 64 query
heads read 8 shared KV heads, 32 per branch — twice the compute per cached byte,
same bytes.

**FINDING: the equality matters above 16k context and the omission below it.**
At 128k the identical cache is 16.00 GB per sequence against 2.00 GB of extra
weights; they are equal at exactly **16,384 tokens**.
