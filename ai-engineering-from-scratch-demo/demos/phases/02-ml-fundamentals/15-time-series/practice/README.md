<!-- generated:start -->
# 02-ml-fundamentals / 15-time-series

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/02-ml-fundamentals/15-time-series/) · upstream spec
`phases/02-ml-fundamentals/15-time-series/docs/en.md`

```bash
uv run demo practice run 15-time-series --ex 1
uv run demo explain 15-time-series --ex 1
uv run pytest demos/phases/02-ml-fundamentals/15-time-series
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Stationarity experiment. Generate a series with a linear trend. Check stationarity with rolli… | code | T0 | `ex01_one_difference_passes_a_noisy_quadratic.py` |
| 2 | Lag selection. Compute ACF on a seasonal series (period=7). Which lags have the highest autoc… | code | T0 | `ex02_acf_picks_28_not_7_and_the_gain_is_averaging.py` |
| 3 | Walk-forward vs random split. Train a Ridge regression on lag features. Evaluate with random… | code | T0 | `ex03_random_split_flatters_knn_2x_but_ridge_only_1_3x.py` |
| 4 | Feature engineering. Add rolling mean (window=7), rolling std (window=7), and day-of-week fea… | code | T0 | `ex04_rolling_mean_is_a_lag_combination_and_adds_nothing.py` |
| 5 | Multi-step forecasting. Modify the AR model to predict 5 steps ahead instead of 1. Compare tw… | code | T0 | `ex05_direct_beats_recursive_by_29pct_at_five_steps.py` |
<!-- generated:end -->

## Answers

The lesson is 352 lines of numpy: lag features, an expanding walk-forward
splitter, a least-squares `SimpleAR` with a recursive `forecast`, and a
half-split stationarity check. Every exercise runs at **T0** against that code.
Exercise 3 adds scikit-learn's `Ridge` and `KNeighborsRegressor`.

### 1 — a quadratic needs two differences, but the check passes it after one

300 points, noise sd 2, verdicts from the lesson's own `check_stationarity`:

| trend | passes at d= |
|---|---:|
| `0.1 t` | 1 |
| `0.01 t²` | 2 |

**ANSWER: one round for a linear trend, two for a quadratic.** Without noise,
the second difference of `b t²` is the constant `2b`.

**FINDING: a gentle quadratic passes after one round.** The half-split test
only fires once the remaining drift beats the noise, and differencing doubles
the noise variance. After one difference the pass rate is **100/100** at
`b = 0.001`, 40/100 at 0.005 and 0/100 at 0.01, with the boundary predicted at
`b = sqrt(12/11)·sqrt(2)·σ / 2n = 0.0049`.

**FINDING: the check never says "too many".** White noise passes at d = 0, 1, 2
and 3, while its variance grows **1.96x, then 5.83x** and its lag-1 ACF becomes
−0.49, then −0.66. The doc says "differencing never hurts". Over-differencing is
exactly the harm, and this check cannot see it.

**CONTROL:** white noise passes on 100/100 seeds. A random walk passes on 13/100.

### 2 — the ACF's top lag is 28, and the gain is averaging

**ANSWER:** on the differenced `make_seasonal_series`, the top four lags are
**28, 7, 14, 21** (0.684, 0.676, 0.672, 0.664). They beat lags 1–7:

| lags | walk-forward MSE |
|---|---:|
| 1–7 | 5.17 |
| 28, 7, 14, 21 | **3.81** |
| 7 | 10.70 |
| 7, 14 | 10.72 |
| 1–30 | 3.10 |
| seasonal naive `y[t-7]` | 12.23 |

The noise floor is 2.25.

**FINDING: the strongest lag is not the period.** On the raw series lag 28
(0.855) beats lag 7 (0.738), because the generator also adds a period-30 sine.
Ranked by |ACF|, the next four lags (4, 10, 17, 25, about −0.64) are the
half-period anti-phase.

**FINDING: selection is not what helps.** One or two seasonal lags lose to
lags 1–7 by a factor of 2. Four copies of the same phase win because the fit
averages their independent noise.

### 3 — the random split flatters Ridge 1.35x and kNN 2x

| model | random 80/20 (20 shuffles) | walk-forward | ratio |
|---|---:|---:|---:|
| Ridge(α=1) | 6.99 | 9.42 | **1.35x** |
| kNN(5) | 9.12 | 18.71 | **2.05x** |
| Ridge, trend removed | 4.82 | 5.49 | 1.14x |
| kNN, trend removed | 4.94 | 5.26 | 1.07x |

**ANSWER: by about a third for Ridge.** The random split makes Ridge's error
look 26% smaller than walk-forward.

**FINDING: the leak is the trend.** A shuffled test point sits between training
points in time, so the random split grades interpolation. kNN cannot
extrapolate, so it is flattered most. Subtract the generator's `0.05 t` and both
gaps nearly close.

**FINDING: the lesson's printed ratio comes from a lucky shuffle.** Its demo
prints 5.92 against 9.41 (ratio 0.63) from `RandomState(42)`, a shuffle in the
luckiest **5.5%** of 200. The mean is 7.38, a ratio of 0.78.

**CONTROL:** Ridge matches `SimpleAR` within 0.01 MSE.

### 4 — day-of-week does all the work

| lags | lags only | + rolling mean | + rolling std | + day-of-week | + all |
|---:|---:|---:|---:|---:|---:|
| 3 | 10.13 | 5.96 | 10.16 | 3.77 | 3.80 |
| 7 | 5.32 | 5.32 | 5.32 | **3.81** | 3.82 |
| 14 | 4.00 | 4.00 | 4.01 | 4.00 | 4.01 |

**ANSWER: the extras cut MSE from 5.32 to 3.82, and day-of-week accounts for
all of it.**

**FINDING: with 7 lags the 7-day rolling mean is not a new feature.** It equals
the average of the seven lag columns exactly, so the design matrix has rank 8 of
9. It helps only when the lags stop short of the window. With 14 lags, no extra
helps.

**CONTROL:** if the window includes `y[t]`, walk-forward MSE drops to
**2.9e-25**, because `y[t] = 7·mean − (y[t−1] + … + y[t−6])`. This reproduces
the doc's target-alignment trap.

### 5 — direct beats recursive by 29% at five steps

There are 246 origins, and every model is refitted on all history at each one:

| h | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| recursive (`SimpleAR.forecast`) | 8.12 | 11.92 | 18.18 | 27.74 | 39.79 |
| direct (one model per h) | 8.12 | 11.18 | 15.71 | 22.08 | **28.08** |

**ANSWER: direct.** It is 29% better at h=5 and 19% better averaged over the five
horizons.

**FINDING: the doc recommends the loser.** It says "start with recursive for
short horizons (1-5 steps)". On the seasonal series the two tie (6.79 against
6.57).

**FINDING: the error growth comes from the model, not the horizon.** The series
is a fixed curve plus i.i.d. noise, so an oracle scores **3.97** at every
horizon. Recursive grows 4.9x because AR(10) is misspecified for sine plus
trend, and feeding its own outputs back compounds the error.

**CONTROL:** on a true AR(1) (φ = 0.8), recursive wins, 2.54 against 2.59 at
h=5. Theory gives 2.48.
