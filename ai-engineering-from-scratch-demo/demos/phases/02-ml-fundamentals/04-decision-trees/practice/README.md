<!-- generated:start -->
# 02-ml-fundamentals / 04-decision-trees

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/02-ml-fundamentals/04-decision-trees/) · upstream spec
`phases/02-ml-fundamentals/04-decision-trees/docs/en.md`

```bash
uv run demo practice run 04-decision-trees --ex 1
uv run demo explain 04-decision-trees --ex 1
uv run pytest demos/phases/02-ml-fundamentals/04-decision-trees
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Train a single decision tree on a 2D dataset with 3 classes. Manually trace the splits and dr… | code | T0 | `ex01_tree_depth_boundaries.py` |
| 2 | Implement variance reduction splitting for regression trees. Generate y = sin(x) + noise for… | code | T0 | `ex02_regression_tree.py` |
| 3 | Build a random forest with 1, 5, 10, 50, and 200 trees. Plot training accuracy and test accur… | code | T0 | `ex03_forest_size.py` |
| 4 | Compare Gini impurity vs entropy as split criteria on 5 different datasets. Measure accuracy… | code | T0 | `ex04_gini_vs_entropy.py` |
| 5 | Implement permutation importance. Compare it with MDI importance on a dataset where one featu… | code | T0 | `ex05_permutation_importance.py` |
<!-- generated:end -->

## Answers

**1 — depth 2 vs depth 10, and the exercise's framing is backwards here.**

| | leaves (= regions) | train | test |
|---|---:|---:|---:|
| depth 2 | 4 | 71.2% | 66.7% |
| depth 10 | 29 | 100.0% | **98.3%** |

The region count *is* the leaf count — each leaf is exactly one region, and
`leaves = splits + 1` holds in both. Depth 2 has 4 regions and uses only 2 of the
3 class labels: a depth-d binary tree has at most 2^d regions, which is the hard
limit the exercise wants you to see.

All 31 internal nodes carry a single `(feature, threshold)` pair, so no boundary
is oblique — that is what makes the regions axis-aligned boxes rather than
arbitrary polygons.

Worth noting the comparison does **not** come out as simple-versus-overfit: depth
10 wins on *test* by 31 points. The shallow tree is too small to represent three
classes; the deep one is not overfitting at all.

**2 — the piecewise-constant fit, and two depth effects the exercise omits.**

The prediction is a staircase: distinct predicted values equal contiguous runs
over sorted x, exactly (4 / 49 / 189 at depths 2 / 6 / 12).

RMSE against the true sine is **U-shaped**, not decreasing: 0.253 → 0.111 → 0.161.
At depth 12 there are 189 distinct values for 200 points, so the leaves are fitting
noise.

And "a constant cannot follow a slope" is only visible at the *right* depth. The
steep/flat error ratio (|cos x| > 0.8 against < 0.2) is **2.35 at depth 6** but
~1.0 at both extremes — a too-coarse tree is uniformly bad, a too-fine one
uniformly noise-fitted, and both wash the effect out.

**3 — test accuracy plateaus and does not decrease, averaged over 5 seeds.**

| trees | test | seed spread |
|---:|---:|---:|
| 1 | 88.7% | 13.3% |
| 5 | 95.0% | 3.3% |
| 200 | 94.3% | **1.7%** |

The largest drop below the running maximum is 0.67% — within seed noise. Five
points from a single seed could not have supported the claim either way, which is
why each size is averaged.

The spread column is the mechanism, visible directly: each tree is fit on a
bootstrap sample so their errors are partly independent, and averaging n of them
divides the variance of that average by up to n while leaving its expectation
alone. Adding trees therefore *cannot* overfit — the resistance belongs to bagging,
not to the trees.

**4 — why Gini and entropy agree.**

Across 5 datasets they differ by at most 2.0% in test accuracy and agree on depth
5 times out of 5. The reason has two parts, both measurable.

They are **not** the same function: entropy peaks at exactly 1.0 bit where Gini
peaks at 0.5, so H/2 is the fair comparison, and even then the largest gap over
every binary class balance is **0.0545** — small, but not zero.

What matters is that a split is chosen by *argmax* of impurity reduction, not by
the impurity value. The two curves correlate at **0.99578** over p ∈ (0,1), so
they order candidate splits identically almost everywhere. The trees coincide
because the ranking does.

**5 — MDI ranks the noise feature highly; permutation does not. With two caveats.**

| feature | values | MDI | permutation |
|---|---:|---:|---:|
| signal_1 | 200 | 0.360 | +0.060 |
| signal_2 | 200 | 0.389 | +0.032 |
| **noise_hi_card** | 200 | **0.252** | +0.020 |
| noise_binary | 2 | 0.000 | +0.000 |

MDI hands **25% of total importance** to a column that is pure noise. Permutation
importance, measured on held-out rows, gives it +0.02 and ranks both signal
features above both noise ones.

**Label noise is required for the effect to exist at all.** With a label the
signal determines exactly, the tree reaches 100% training accuracy on the signal
features alone and MDI gives the noise columns 0.006 and 0.000 — the exercise's
premise never arises. The label here carries 25% flips, leaving residual impurity
for a spurious split to reduce.

**And cardinality is the mechanism, not noise.** The binary noise column is
*equally* uninformative and gets MDI **0.0**. ~200 candidate thresholds always
include one that splits the training labels a little, and MDI pays for it; one
threshold does not. That control is what identifies the cause.
