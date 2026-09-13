<!-- generated:start -->
# 07-transformers-deep-dive / 16-speculative-decoding

Solutions to all 4 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/07-transformers-deep-dive/16-speculative-decoding/) · upstream spec
`phases/07-transformers-deep-dive/16-speculative-decoding/docs/en.md`

```bash
uv run demo practice run 16-speculative-decoding --ex 1
uv run demo explain 16-speculative-decoding --ex 1
uv run pytest demos/phases/07-transformers-deep-dive/16-speculative-decoding
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py`. Confirm the speculative token distribution matches the verifier's d… | code | T0 | `ex01_the_test_it_specifies_rejects_a_true_theorem.py` |
| 2 | Medium. Plot speedup (tokens per big-model forward) as a function of `N` for `α = 0.5, 0.7, 0… | code | T0 | `ex02_there_is_no_optimal_n_for_the_metric_named.py` |
| 3 | Hard. Implement a tiny Medusa: take the capstone GPT from Lesson 14, add 3 extra LM heads tha… | code | T1 | `ex03_truncating_layers_is_not_a_smaller_model.py` |
| 4 | Hard. Implement rollback: start with a 10-token prefix KV cache, feed 5 draft tokens, simulat… | code | T0 | `ex04_prefix_plus_two_is_what_pins_the_indexing.py` |
<!-- generated:end -->

## Answers

The lesson is 159 lines implementing Leviathan's algorithm correctly and then
checking it with a statistic that rejects it. Exercise 1 specifies a test that
fails 43% of the time on a true theorem; Exercise 2 asks for the optimum of a
monotone function; Exercise 4's instruction is ambiguous until its own
verification target disambiguates it. Exercises 1, 2 and 4 are **T0**; Exercise 3
is **T1** (numpy).

### 1 — the test it specifies rejects a theorem that is exactly true

**ANSWER: the theorem holds to 1.4e-17, with no sampling at all.**
`p_i·min(1, q_i/p_i) + P(reject)·res_i = q_i` for every `i`, and
`P(reject) = 0.040306 = 1 − Σ min(p, q)` — the total variation distance, exactly.

| statistic, 40 repeats of 50,000 samples | mean | exceeds 14.067 |
|---|---:|---:|
| `chi_square(spec, direct)` — what `main()` prints | **12.79** | **17 / 40** |
| `chi_square(spec, 50000·q)` — against the truth | 6.31 | 1 / 40 |
| (a χ² on 7 df) | 7 | ~2 / 40 |

**FINDING: the printed statistic runs twice as hot as it should**, because its
"expected" counts are a second random sample and it carries both variances.

**FINDING: so the exercise's own criterion fails 43% of the time on a true
theorem.** A reader following `p > 0.05` would conclude Leviathan's theorem fails
on almost half their runs.

**FINDING: and the threshold the lesson ships can never fire.** `main()` passes
below **30** (≈ p = 1e-4) — exceeded **0** times in 40. The test as written
cannot fail; the test as specified fails half the time.

### 2 — the metric the exercise names has no optimal N

| α | N=1 | N=3 | N=5 | N=12 | ceiling |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 1.500 | 1.875 | 1.969 | 2.000 | **2.000** |
| 0.70 | 1.700 | 2.533 | 2.941 | 3.301 | **3.333** |
| 0.85 | 1.850 | 3.187 | 4.152 | 5.861 | **6.667** |

**ANSWER: `(1 − α^(N+1))/(1 − α)` is strictly increasing in N for every α.** Each
extra position adds exactly `α^N` — a gain that decays to zero and never turns
negative. "Identify the optimal N" has the answer "as large as you like".

**FINDING: the optimum appears as soon as the draft is charged for.**

| draft cost c | α=0.5 | α=0.7 | α=0.85 |
|---|---|---|---|
| 1/3 | N*=1 (1.12×) | N*=2 (1.31×) | N*=3 (1.59×) |
| 1/10 | N*=2 (1.46×) | N*=4 (1.98×) | N*=7 (2.85×) |
| 1/20 | N*=3 (1.63×) | N*=6 (2.35×) | N*=10 (**3.70×**) |

The missing variable is the draft model's cost — missing from the hint, from
`expected_tokens_per_verify`, and from the exercise.

**FINDING: the formula assumes α does not decay with depth, and so does
`spec_step_n`** ("Simplified: q and p are fixed per call"). With a 10%
per-position decay, `E(5)` at α₀ = 0.85 falls from **4.152 to 3.380**; at 20%, to
**2.927**.

### 3 — truncating layers gives a broken model, not a smaller one

| draft | acceptance |
|---|---:|
| head 1 (t+1) | 0.202 |
| head 2 (t+2) | **0.214** |
| head 3 (t+3) | 0.114 |
| head 4 (t+4) | 0.125 |
| 1-layer truncation of the same model | **0.079** |
| always predict the commonest character | **0.192** |

**ANSWER: every Medusa head beats the truncated draft, the weakest by 1.4×.**

**FINDING: only two of the four clear the unigram baseline**, and heads 3 and 4
land below it. On 899 training characters both arms have learned almost nothing,
so the ranking is the only part that survives. That head 2 outscores head 1 is
the tell: predicting two ahead cannot genuinely be easier than predicting one.

**FINDING: truncating layers is not the same as having a smaller model.** The
1-layer prefix agrees with the full network **0.079** of the time — less than
half the unigram baseline. The residual stream it hands the head was built for
three layers of refinement. Real draft models are trained, not sliced.

### 4 — "prefix + first 2 accepted" is the only thing that pins the index

**ANSWER: 10 → 15 → 12, and all 12 entries are bit-identical to a cache built
from scratch.** Length alone would also pass a cache holding the right *number*
of stale rows.

**FINDING: the instruction is ambiguous without its verification target.**
"Rejection at position 3" leaves 3 accepted drafts and a cache of 13 if
0-indexed, 2 and 12 if 1-indexed. Only "first 2 accepted drafts" settles it.

**FINDING: the verifier's own corrected token then makes 13.** A 5-token draft
with 2 accepted nets **3 tokens for one verifier forward**; a rollback that stops
at 12 drops the token the rejection produced.

**FINDING: truncation and rollback agree, which is what makes the cheap
implementation legal.** A causal KV entry depends only on its own position and
those before it — so the rejected entries can be discarded instead of recomputed.
A bidirectional cache would not have that property.

**CONTROL: without the rollback the next step attends 15 positions**, 3 of them
rejected. No shape is wrong and nothing raises, so a missing rollback fails
silently and shows up as quality.
