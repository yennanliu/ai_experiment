<!-- generated:start -->
# 02-ml-fundamentals / 08-feature-engineering

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/02-ml-fundamentals/08-feature-engineering/) · upstream spec
`phases/02-ml-fundamentals/08-feature-engineering/docs/en.md`

```bash
uv run demo practice run 08-feature-engineering --ex 1
uv run demo explain 08-feature-engineering --ex 1
uv run pytest demos/phases/02-ml-fundamentals/08-feature-engineering
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add robust scaling (using median and interquartile range instead of mean and standard deviati… | code | T0 | `ex01_robust_scaling.py` |
| 2 | Implement leave-one-out target encoding: for each row, compute the target mean excluding that… | code | T0 | `ex02_leave_one_out_encoding.py` |
| 3 | Build an automated feature selection pipeline that combines variance threshold, correlation f… | code | T0 | `ex03_feature_selection_pipeline.py` |
<!-- generated:end -->

## Answers

**1 — robust scaling keeps the inliers; standardisation destroys them.**

95 draws from N(10, 1) plus five outliers at ±400–700:

| | inlier IQR | inlier span | inliers within ±1 | max \|outlier\| |
|---|---:|---:|---:|---:|
| standardised | **0.0100** | 0.0406 | **100%** | 5.50 |
| robust | **0.9577** | 3.8885 | 87% | 541.32 |

A factor of **96** in inlier resolution. Five values in a hundred set the standard
deviation, and then every ordinary point is divided by it.

The second column is the framing that matters. "Robust scaling handles outliers
better" is easy to say and hard to mean: both transforms are affine, so **neither
hides an outlier** — they come out at 5.50 and 541.32 respectively, flagged either
way. The difference is entirely about what happens to the other 95 points, and
after standardisation they are mutually indistinguishable. The z-score's usual
reading — ±1 is typical, ±3 is far — no longer holds, because here ±1 covers
everything.

**And robust scaling costs nothing when there is nothing to be robust to.** With
the outliers removed the two transforms correlate at **1.000000** — the same
affine map — differing only by a scale factor of 1.4059, against the Gaussian
constant IQR/σ = 1.3490. (The 4% discrepancy is the sample IQR's own error at
n=95, not a difference between the methods.)

**2 — the naive encoding does not merely flatter the training score; it is worse.**

80 categories over 300 training rows, nine of them holding a single row. The
control column is assigned *independently* of the target, so any signal found in
it is leakage by construction.

| | train corr | fitted slope | train RMSE | test RMSE | gap |
|---|---:|---:|---:|---:|---:|
| naive | **+0.4607** | **+1.0000** | 0.8616 | 1.1409 | **+0.2793** |
| leave-one-out | −0.1323 | — | 0.9622 | 0.9967 | +0.0345 |
| mean-only baseline | — | — | — | 0.9940 | — |

The **slope of exactly +1.0000** is the cleanest diagnostic in this lesson: least
squares has discovered that the feature is a copy of the target. For a
single-row category the naive encoding *is* that row's own target value, and
nine categories are in that position.

Note the last row. Naive encoding's held-out error (1.1409) is **worse than
ignoring the feature entirely** (0.9940). Leakage is not a free lunch that only
inflates the training score — a model that has learned to read the target off a
column it will not have at prediction time is actively misfitted.

**Leave-one-out keeps genuine signal.** On a second column where the category
really does shift the mean, held-out RMSE is 1.0423 leave-one-out against 1.4886
for the mean-only baseline — and 1.0593 for naive, so removing the leak *wins*
out of sample despite the worse training score.

**FINDING: leave-one-out over-corrects.** Its residual training correlation is
**−0.1323**, not zero. Removing your own target from your category's mean pushes
that mean the other way, so a row above its group average gets a below-average
encoding. With small categories the effect is large enough for a model to learn
to invert it — leave-one-out reduces leakage, it does not eliminate it, and the
sign of what remains is the opposite of the obvious one.

**FINDING: the lesson's smoothing does not address leakage.** `target_encode`'s
`smoothing` parameter shrinks each category mean toward the global mean, and the
training correlation on the independent column barely moves: **+0.4607, +0.4320,
+0.4196** at smoothing 0, 10 and 50. Shrinkage is nearly monotone, so it rescales
the leaked values without unranking them, and leakage is about which row a value
came from rather than how large it is. Smoothing is regularisation, not a fix for
train-time contamination.

**3 — two stages work exactly as advertised; the third ranks noise second.**

Ten columns: the seven real ones (imputed sqft and age, bedrooms, three one-hot
neighbourhoods, has_pool) plus three planted with a known right fate.

| stage | k | drops | held-out RMSE | design condition |
|---|---:|---|---:|---:|
| all columns | 10 | — | 23,725 | **3.79e+19** |
| variance ≥ 0.01 | 9 | `near_constant` ✓ | 23,600 | 2.94e+19 |
| \|corr\| < 0.9 | 8 | `sqft_dup` ✓ | 23,600 | 1.84e+19 |
| MI top-5 | 5 | keeps `noise` ✗ | 24,411 | **1.24e+04** |

(Mean-only baseline: 83,642.)

**FINDING: mutual information ranks the pure-noise column second of eight.**

| feature | MI (5-bin target) | MI (raw target) |
|---|---:|---:|
| sqft | 0.5377 | 2.297 |
| **noise** | **0.1213** | 1.887 |
| age | 0.1190 | 2.269 |
| bedrooms | 0.1090 | 1.597 |
| hood_downtown | 0.0580 | 0.634 |
| has_pool | 0.0078 | 0.693 |

Two separate problems are visible here.

The **binned** column is a null-model problem. `mutual_information` bins the
feature into 10 and estimates a joint distribution over 150 training rows. A
continuous random column spreads over all 10 bins, so plenty of (bin, class)
cells hold a couple of rows and the plug-in estimator reads that as information.
MI has no null model to compare against — it never asks how much a random column
would score — so the noise column outranks age, bedrooms, and every one-hot.

The **raw** column is worse and is a spec problem. `mutual_information` computes
`p_target` with `target.count(t)`, so it treats the target as *classes*. Against
raw prices each of 200 values is its own class, and the score degenerates into a
measure of how many bins the *feature* occupies — which is why the two continuous
columns (sqft, age) top that ranking and the binary `has_pool` sits near the
bottom despite contributing $15,000 to every price. The exercise says to use
mutual information ranking on the housing dataset and never says to bin the
target; done literally, the stage cannot work.

**ANSWER: selection does not improve held-out accuracy here.** 23,725 → 23,600 →
24,411 as columns come off. The two mechanical stages remove genuinely useless
columns and win 0.5%; the MI stage *loses* 3.4% because it drops three real
features to keep the noise one. Least squares already assigns a redundant column
a near-zero weight, so there was little for a filter to recover.

**What selection actually buys is conditioning.** The condition number of the
training design falls from **3.79e+19** to **1.24e+04** — fifteen orders of
magnitude. The three one-hot columns sum to the intercept and `sqft_dup` is
collinear with `sqft`, so the full design is numerically singular. `lstsq` returns
its minimum-norm solution and reports nothing, which is why the RMSE column looks
untroubled — but the coefficients are not interpretable, no other solver need be
so forgiving, and the fit is one dataset away from being unstable. That is the
case for the pipeline, and it is not the case the exercise asks you to look for.

## A note on the reference code

`make_housing_data` calls the module-level `random.seed(seed)` rather than using
a local `random.Random(seed)`, so importing and calling it reseeds the caller's
global RNG. Solutions here use their own `random.Random` instances, which is
unaffected — but any code that seeded the global RNG before calling it will find
its stream replaced.
