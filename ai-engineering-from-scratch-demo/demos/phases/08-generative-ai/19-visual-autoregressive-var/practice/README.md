<!-- generated:start -->
# 08-generative-ai / 19-visual-autoregressive-var

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/08-generative-ai/19-visual-autoregressive-var/) · upstream spec
`phases/08-generative-ai/19-visual-autoregressive-var/docs/en.md`

```bash
uv run demo practice run 19-visual-autoregressive-var --ex 1
uv run demo explain 19-visual-autoregressive-var --ex 1
uv run pytest demos/phases/08-generative-ai/19-visual-autoregressive-var
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Scale count ablation. Train VAR with 4, 6, 8, 10 scales. Measure reconstruction quality vs nu… | code | T0 | `ex01_coarse_passes_cost_quality_and_four_full_res_passes_win.py` |
| 2 | Codebook size. Train tokenizers with codebook sizes 512, 4096, 16384. Larger codebooks give b… | code | T0 | `ex02_past_256_codes_the_codebook_collapses_and_both_axes_get_worse.py` |
| 3 | Parallel-within-scale check. For a trained VAR, measure the attention pattern explicitly. Wit… | code | T0 | `ex03_no_mask_exists_and_the_last_scale_is_uniform_noise_98_percent_of_the_time.py` |
| 4 | VAR vs DiT scaling. For the same ImageNet class-conditional task, train VAR and DiT at matche… | code | T0 | `ex04_the_toy_var_is_no_closer_to_the_data_than_uniform_noise_at_any_size.py` |
| 5 | Text conditioning. Extend VAR to take a text embedding (CLIP pooled) as an extra conditioning… | code | T0 | `ex05_conditioning_cannot_fix_a_predictor_that_does_not_know_where_pixels_go.py` |
<!-- generated:end -->

## Answers

The lesson is 236 lines of numpy: a scalar residual-VQ tokenizer over a
library of five 8x8 patterns, and a "transformer" that is one count table per
scale keyed on `context_key` — `int(mean * 1000)` of each earlier scale. Every
exercise runs that code as shipped, at **T0** with numpy (`deps_group: math`).
Exercises 4 and 5 name torch, ImageNet, CLIP and DiT, none of which are here, so
they ship the scaled-down runnable `DESIGN D11` requires, with a pixel-space
Fréchet distance standing in for FID.

Two facts about the predictor decide exercises 3–5: every position of a scale is
drawn from **one shared distribution**, and at generation the 8x8 scale's
context is one the table never saw **98%** of the time, so `generate` samples it
uniformly.

### 1 — coarse passes cost quality; four full-resolution passes win

The 8x8 grid admits only the sizes 1, 2, 4, 8, so 6, 8 and 10 scales must repeat
one. Val MSE, lesson's own `train_codebooks` and `reconstruction_mse`:

| passes | 4 | 6 | 8 | 10 |
|---|---:|---:|---:|---:|
| repeats at 8x8 | 1.5e-04 | 5.9e-07 | 2.0e-09 | **2.3e-12** |
| repeats at 1/2/4 | 1.5e-04 | 1.2e-04 | 4.0e-04 | **4.2e-04** |

**ANSWER:** "more scales = better" holds only when the extra passes are
full-resolution; coarse repeats make 10 passes worse than 4.

**FINDING: the pyramid is the costly part.** `(8,8,8,8)` reaches **3.8e-10**,
~400,000x below the lesson's `(1,2,4,8)` at the same 4 passes; one 8x8 pass
alone gets 5.2e-04.

**FINDING:** the paper's `(1,2,3,4,5,6,8,10,13,16)` cannot run —
`downsample(img, 3)` raises `ValueError`.

**CONTROL:** what the pyramid saves is tokens (85 vs 256), not passes.

### 2 — past 256 codes the codebook collapses and both axes get worse

512, 4096 and 16384 each raise `ValueError` on `main()`'s 64 images:
`fit_codebook` needs one sample per code, and scale 1 has one sample per image.
The sweep runs 16..512 on 512 images.

| codes | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|
| val MSE | 2.5e-04 | 4.5e-06 | 4.3e-06 | 3.4e-08 | **2.5e-08** | 2.1e-06 |
| NLL nats/token | 1.46 | 1.63 | 1.96 | 2.21 | 2.36 | 2.56 |

**ANSWER: the knee is 256, and it is a cliff** — beyond it reconstruction gets
worse while prediction keeps getting harder (3/3 data seeds).

**FINDING:** at 512 codes the scales hold only **111, 19, 17, 36** distinct
values. The tokenizer is scalar: once scale 1 can memorise every pooled mean,
the residuals below it hold a handful of values, and k-means seeded from them
starts duplicate centres that never separate.

**CONTROL:** `fit_predictor`'s add-one prior is 3% of the scale-1 table at 16
codes and **50%** at 512, so part of "harder prediction" is the smoothing.

### 3 — no mask exists, and the last scale is uniform noise 98% of the time

There are no attention weights, so dependence is measured instead.

**ANSWER: nothing attends within a scale** — instrumenting `sample_categorical`
inside `generate` shows one distribution object per scale per sample. The model
is also blind to position: real 8x8 per-position histograms differ from the
pooled one by mean TV **0.55**; the model's by 0.

**FINDING: cross-scale, values but never positions** — 19,200/19,200 token
edits move `context_key`, **0/3,681** within-scale swaps do.

**FINDING:** unseen-context fallback to uniform runs **0%, 10%, 86%, 98%** by
scale.

**CONTROL:** `main()`'s "attention check" prints `sum(s*s)` over scales 1..k−1
(1,428 pairs); docs/en.md says 1..k (5,797 pairs). The code consults neither.

### 4 — the toy VAR is no closer to the data than uniform noise, at any size

**ANSWER:** across 304 → 6,080 table parameters (16 → 4,096 images) the
Fréchet distance to held-out images is 14.8, 10.5, 9.5, 12.1, 11.1 against
**11.4** for uniform random pixels; log-log slope **−0.05**, no power law.
There is no VAR advantage for DiT to be compared against.

**FINDING:** compute is fixed — 4 passes, 85 tokens at every budget.

**CONTROL:** replaying training token streams through the same decoder scores
0.06–0.21, real vs real 0.03. The tokenizer and the metric work; the predictor
is what fails.

### 5 — conditioning cannot fix a predictor that does not know where pixels go

The "text" is the pattern class; adaLN's per-condition parameters become
per-class tables from the lesson's own `fit_predictor`.

**ANSWER: about 10%** — class-matched distance 17.7 → 15.9; alignment at most 5%.

**FINDING:** each class's *true* tokens, shuffled within each scale, score
**6.4–31.9** against **0.002–0.040** unshuffled. Position blindness is the
ceiling, not missing conditioning.

**FINDING:** a class that is one fixed image, seen 11–15 times, gets its correct
1x1 token with probability **0.44–0.52** — exactly (n+1)/(n+16), the add-one
prior.

**CONTROL:** the prompt does reach the model: samples with the true 1x1 token
rise from 16–22% to 45–54%.
