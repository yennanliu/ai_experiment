<!-- generated:start -->
# 02-ml-fundamentals / 10-bias-variance

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/02-ml-fundamentals/10-bias-variance/) · upstream spec
`phases/02-ml-fundamentals/10-bias-variance/docs/en.md`

```bash
uv run demo practice run 10-bias-variance --ex 1
uv run demo explain 10-bias-variance --ex 1
uv run pytest demos/phases/02-ml-fundamentals/10-bias-variance
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run the decomposition with `noise_std=0` (no noise). What happens to the irreducible error te… | code | T0 | `ex01_without_noise_the_optimum_runs_off_the_sweep.py` |
| 2 | Increase the training set size from 30 to 300. How does this affect the variance component? D… | code | T0 | `ex02_degree_6_wins_only_by_monte_carlo_luck.py` |
| 3 | Add L2 regularization (Ridge regression) to the experiment. For a fixed high-degree polynomia… | code | T0 | `ex03_ridge_on_raw_monomials_penalises_the_wrong_terms.py` |
| 4 | Modify the true function from a polynomial to `sin(x)`. How does the bias-variance decomposit… | code | T0 | `ex04_above_degree_8_the_bias_column_is_monte_carlo_noise.py` |
| 5 | Implement a simple bootstrap aggregating (bagging) wrapper: train 10 models on bootstrap samp… | code | T0 | `ex05_bagging_a_polynomial_raises_its_variance.py` |
<!-- generated:end -->

## Answers

The lesson is one numpy file: `true_function = sin(1.5x) + 0.5x`, a
least-squares (or ridge) polynomial fit, and `bias_variance_decomposition`,
which refits on 200 fresh training sets of 30 and scores a 100-point grid. All
five exercises are **T0** against that code, `deps_group: math`.

### 1 — without noise the optimum runs off the sweep

| noise_std | noise term | `find_optimal` | total at degree 6 | total at degree 15 |
|---|---:|---:|---:|---:|
| 0.5 | 0.25 | 6 | 0.3095 | 26504 |
| 0 | 0 | **15** | 1.30e-3 | **6.22e-13** |

**ANSWER: the irreducible term drops to exactly 0, and the optimum moves from
degree 6 to 15**, the largest degree offered, so it is the edge of the grid,
not a minimum.

**FINDING: with no noise there is nothing to overfit.** The error falls 2.1e9x
from degree 6 to 15; `sin(1.5x) + 0.5x` is analytic, so extra terms only remove
bias.

**FINDING: variance survives without noise**: 0.0298 at degree 1, 69% of its
noisy value. A misspecified fit depends on where the 30 x values land.

**CONTROL:** the lesson's `total_error` is the noiseless error plus
`noise_std**2`, so "Total = B+V+N" holds to 4.7e-10 by algebra, not by
measurement. Measured against fresh noisy labels, 200 degree-6 fits err 0.3413
against 0.0917 to the truth: the noise adds 0.2497, i.e. sigma^2.

### 2 — degree 6 wins only by Monte Carlo luck

| | n=30 | n=300 | n=1000 |
|---|---:|---:|---:|
| lesson's `find_optimal` | 6 | 6 | — |
| degree-6 variance (lesson) | 0.0583 | 0.0040 | — |
| optimum on common datasets | **5** | 5 ≈ 7 (0.4%) | **7** |
| degree 5 vs 6 (common) | 0.0556 / 0.0660 | 0.00532 / 0.00594 | 5 < 6 |

**ANSWER: 10x the data cuts variance 14.4x at degree 6** (2.5e6x at degree 15,
which at n=30 nearly interpolates), and the lesson's optimum stays at 6.

**FINDING: degree 6 is never the real optimum.** The target is odd, so x^6 buys
no bias (0.00122 for both 5 and 6) and adds only variance. Fit every degree to
the same datasets and 5 beats 6 at every size; the lesson draws new data per
degree, and that noise is bigger than the gap.

**FINDING: the optimum does shift**, to 7, but only past n=300: a 0.4% tie at
300, a clear win at 1000 (0.00154 vs 0.00231).

**CONTROL:** bias^2 does not move with data, 0.00117 -> 0.00117 at degree 5.

### 3 — ridge on raw monomials penalises the wrong terms

| lambda | 0 | 1e-3 | 0.01 | 0.1 | 1 | 10 | 100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bias^2 | 102 | 2.06 | 0.391 | 0.0602 | 0.334 | 0.330 | 0.588 |
| variance | 12400 | 315 | 82.5 | 72.4 | **124** | 31.9 | **47.4** |
| bias^2, x in [-1, 1] | 102 | 0.0016 | 0.0158 | 0.0845 | 0.300 | 0.785 | 1.25 |
| variance, x in [-1, 1] | 12400 | 0.073 | 0.048 | 0.042 | 0.041 | 0.039 | 0.043 |

**ANSWER: the lesson's sweep bottoms out at lambda=10 with bias^2 + variance =
32.2**, 500x the best unregularised degree. Variance never gets below 31.9.

**FINDING: variance is not monotone in lambda, and high lambda never yields
the doc's "near-constant function".** `fit_polynomial` penalises raw
monomials: x^15 reaches ~1e7 on [-3, 3], so its weight is tiny and its penalty
negligible, and lambda lands on the low-order terms. On one dataset the x
coefficient falls 2.07 -> 0.08 while the fit still spans 2.74 units.

**FINDING: rescale x to [-1, 1] and it is the textbook U** — bias^2 rises
monotonically, variance falls, best lambda=0.01 at 0.064.

**CONTROL:** the replay loop reproduces the lesson's decomposition exactly at
scale 1.

### 4 — sin(x) moves the optimum to 3; the high-degree bias column is noise

**ANSWER: with `sin(x)` the optimum moves from degree 6 to 3**, a clear but
modest one: 0.282 against 0.292 at degree 4 (+3.4%) and 0.308 at 6. Bias^2 at
degree 3 falls 0.0529 -> 0.0020.

**FINDING: from degree 8 up the truth does not matter.** Bias^2 and total agree
between `sin(x)` and the lesson's function within 0.5% at every degree 8-15:
same seeds, same noise, and the error is all noise.

**FINDING: the lesson's high-degree bias^2 is variance leaking through 200
draws.** It reads 0.0008, 0.0017, **0.545**, 0.0061 at degrees 7-10; the
200-draw mean prediction carries variance/199 of its own, which at degree 9 is
0.552, 99% of the reported value.

**CONTROL:** the exercise's "from a polynomial" does not match the lesson, whose
truth is `sin(1.5x) + 0.5x`. With a real cubic `0.5x^3 - x`, bias^2 at degree 3
is 6.7e-5 (Monte Carlo zero).

### 5 — bagging a least-squares polynomial raises its variance

| degree | 1 | 3 | 5 | 8 | 15 |
|---|---:|---:|---:|---:|---:|
| single fit | 0.0434 | 0.0365 | 0.0449 | 0.281 | 1.24e4 |
| bagged x10 | 0.0495 | 0.0423 | 0.0798 | 36.7 | 1.13e15 |
| 10 independent sets | — | — | 0.0052 | — | — |

**ANSWER: it does not reduce variance; bagging raises it at every degree**,
against `docs/en.md`'s rule "high variance (deep trees, high-degree
polynomials), use bagging".

**FINDING: a bootstrap sample of 30 holds 19.1 distinct points**, so each
member is a degree-d fit to ~19 points; at degree 15 the median per-set error
goes 0.52 -> 5.2e6.

**FINDING: even at degree 1 there is nothing to gain**: least squares is
linear, so the bootstrap average is to first order the same fit. Bagging pays
off for unstable non-linear learners such as trees.

**CONTROL:** averaging 10 *independent* training sets cuts degree-5 variance
8.6x, so the averaging is sound and the resampling is what fails.
