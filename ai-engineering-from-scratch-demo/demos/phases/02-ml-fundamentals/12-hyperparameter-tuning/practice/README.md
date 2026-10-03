<!-- generated:start -->
# 02-ml-fundamentals / 12-hyperparameter-tuning

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/02-ml-fundamentals/12-hyperparameter-tuning/) · upstream spec
`phases/02-ml-fundamentals/12-hyperparameter-tuning/docs/en.md`

```bash
uv run demo practice run 12-hyperparameter-tuning --ex 1
uv run demo explain 12-hyperparameter-tuning --ex 1
uv run pytest demos/phases/02-ml-fundamentals/12-hyperparameter-tuning
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run grid search and random search with the same total budget (e.g., 50 evaluations). Compare… | code | T0 | `ex01_random_wins_4_of_10_optimum_on_grid_edge.py` |
| 2 | Implement Hyperband from scratch. Start with 81 configurations, each trained for 1 epoch. Kee… | code | T0 | `ex02_halving_saves_22x_but_cuts_the_winner.py` |
| 3 | Add a learning rate scheduler (cosine annealing) to the gradient boosting implementation from… | code | T0 | `ex03_cosine_is_a_halved_rate_and_loses_to_one.py` |
| 4 | Use Optuna to tune a RandomForestClassifier on a real dataset (e.g., sklearn's breast cancer… | code | T0 | `ex04_importance_ranking_flips_with_the_metric.py` |
| 5 | Implement a simple acquisition function (Expected Improvement) and demonstrate exploration vs… | code | T0 | `ex05_ei_exploits_first_explores_last.py` |
<!-- generated:end -->

## Answers

The lesson's `tuning.py` is grid search, random search and a GP + Expected
Improvement optimiser, all wrapped around a pure-Python `GBMForTuning`. That
model costs 3–25 ms per tree, so where an exercise needs hundreds of fits
(1, 2) the lesson's own search code runs unchanged on scikit-learn's
`GradientBoostingRegressor` with the same signature, and a control checks the
conclusion against `GBMForTuning`. Optuna (4) is not a dependency; its default
importance evaluator is rebuilt in a few lines. Everything is **T0**.

### 1 — random search wins 4 of 10, because the optimum is on the grid's edge

The doc's own 3×3 grid (lr 0.01/0.1/1.0 × depth 3/5/7) against 9 random draws
over the same box, data and search reseeded together:

| second axis | random wins | mean best MSE, grid | mean best MSE, random |
|---|---:|---:|---:|
| max_depth 3–7 | **4 / 10** | **0.6117** | 0.6435 |
| min_samples_split 2–20 (control) | **10 / 10** | 0.5889 | **0.4684** |

**ANSWER: 4 of 10.** **FINDING:** the grid's best cell is depth 3 — its
smallest value — on all 10 seeds, and random search draws depth 3 on only 21%
of trials. The doc's Bergstra–Bengio argument assumes the second axis barely
matters; when it does and its optimum is a corner, the grid always visits the
corner. Swap in an axis that does not matter and random search wins every time.
The lesson's own `GBMForTuning` agrees depth 3 beats 5 (0.684 vs 0.962).

### 2 — successive halving saves 22x and keeps the winner once in three

One epoch = one boosting tree; 81 configs from the lesson's own search space,
rungs 81×1, 27×3, 9×9, 3×27, 1×81:

| | epochs | vs 81 × 81 |
|---|---:|---:|
| halving, resuming each rung | **297** | 22.1x less |
| halving, retraining each rung | 405 | 16.2x less |
| Li et al.'s full Hyperband (5 brackets, 143 configs) | 1,902 | **3.45x less** |

**ANSWER: 22x** — inside the doc's "10–50x faster". **FINDING:** that figure
is one bracket; the hedged algorithm called Hyperband saves 3.45x.

**FINDING: what it saves on, it often gets wrong.** On seeds 0–2 halving's pick
ranks **25th, 1st and 26th** of 81; both missed winners were cut at the 3-tree
rung. The expected best of 3 random configs run to the full 81 trees — 243
epochs, *less* than halving's 297 — beats it on 2 of 3 seeds (0.689 / 0.594 /
0.720 against 0.693 / 0.367 / 0.750). Early rungs rank fast starters, not
finishers.

### 3 — cosine annealing is a halved learning rate, and loses to one

Lesson 11's boosting loop with tree t shrunk by `lr · ½(1 + cos(πt/T))`,
100 trees, 400 fresh test rows:

| | train MSE | test MSE |
|---|---:|---:|
| lr 0.1 fixed (lesson default) | 0.0623 | **0.2693** |
| lr 0.1 cosine | 0.1656 | 0.3941 |
| lr 0.5 fixed | 0.0037 | 0.3168 |
| lr 0.5 cosine | 0.02015 | 0.3049 |
| fixed 0.2525 (same total shrinkage as lr 0.5 cosine) | 0.02011 | **0.2614** |

**ANSWER: no** — +46% test MSE at the lesson's lr, −4% at 0.5.
**FINDING:** the cosine shrinkages sum to lr(T+1)/2, and the schedule's train
MSE matches a fixed rate of that size to 0.2%: in boosting, cosine annealing is
a learning-rate change in disguise — and the plain change does better.

### 4 — the importance ranking flips with the metric

40 random trials of a `RandomForestClassifier` on breast cancer (stratified 30%
holdout), plus a `dummy` hyperparameter the model never sees; importance is an
fANOVA-style main effect under a random-forest surrogate (Optuna's default):

| | min_samples_leaf | max_features | n_estimators | max_depth | dummy |
|---|---:|---:|---:|---:|---:|
| accuracy | **73.8%** | 23.9% | 0.2% | 0.1% | 2.0% |
| log loss | 0.5% | 37.2% | **59.4%** | 2.8% | 0.0% |

**ANSWER: no.** The doc's Random Forest row reads "n_estimators, max_depth,
min_samples_leaf" and it files max_features under "Low importance". Under
accuracy max_features is second and n_estimators and max_depth sit *below the
dummy*. **FINDING:** under log loss n_estimators goes from last to first — a
small forest votes in steps of 1/n, which log loss punishes and accuracy cannot
see. "Which hyperparameters matter" has no answer until the metric is fixed.

### 5 — EI exploits first and explores last, and the lesson's EI goes negative

The lesson's `SimpleBayesianOptimizer` on log learning rate for `GBMForTuning`,
4 random points then 8 guided. Before the first guided step:

| lr | 0.005 | 0.0126 | 0.0315 | 0.0792 | 0.1991 | 0.3155 | 0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GP mean | −1.665 | −1.960 | −2.214 | −1.663 | −0.542 | −0.204 | **−0.163** |
| GP sd | 0.891 | 0.557 | 0.134 | 0.010 | 0.068 | 0.229 | 0.454 |
| EI | +0.114 | **−0.007** | −0.000 | +0.000 | 0.458 | 0.793 | **0.828** |

**ANSWER:** EI picks lr 0.417, where the mean is high *and* uncertain; next to
observed points (sd < 0.03) it is zero.

**FINDING: the order is the reverse of the doc's.** Splitting EI into its
exploit term (μ − best)Φ(z) and explore term σφ(z), the explore term chose the
point on guided steps **5–8 and none of 1–4**. Once the incumbent is good,
μ − best < 0 almost everywhere and only uncertainty is left, largest at the
box's edges (steps 4–7 pick lr 0.497, 0.491, 0.500, 0.0078). The doc says the
optimizer explores early and focuses later.

**FINDING: the lesson's `_norm_cdf` is `0.5(1 + tanh(0.798x))`** — the GELU
approximation missing its `0.044715x³` term, off by up to 0.0177. EI/σ then goes
negative below z = −1.55 (the −0.007 above), which true EI never does: negative
on 107–308 of 500 candidates per step, and the argmax differs from exact EI's
on **3 of 8** steps.
