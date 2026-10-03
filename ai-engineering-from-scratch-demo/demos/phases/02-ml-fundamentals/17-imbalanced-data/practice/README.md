<!-- generated:start -->
# 02-ml-fundamentals / 17-imbalanced-data

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/02-ml-fundamentals/17-imbalanced-data/) · upstream spec
`phases/02-ml-fundamentals/17-imbalanced-data/docs/en.md`

```bash
uv run demo practice run 17-imbalanced-data --ex 1
uv run demo explain 17-imbalanced-data --ex 1
uv run pytest demos/phases/02-ml-fundamentals/17-imbalanced-data
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Borderline-SMOTE: modify the SMOTE implementation to only generate synthetic samples for mino… | code | T0 | `ex01_borderline_rule_is_vacuous_under_overlap.py` |
| 2 | Cost matrix optimization: implement cost-sensitive learning where the cost matrix is a parame… | code | T0 | `ex02_cost_rule_needs_calibration_the_lesson_lacks.py` |
| 3 | Threshold calibration: implement Platt scaling (fit a logistic regression on the model's raw… | code | T0 | `ex03_platt_only_calibrates_when_converged.py` |
| 4 | Ensemble with balanced bagging: train multiple models, each on a balanced bootstrap sample (a… | code | T0 | `ex04_bagging_is_noisier_than_one_smote_model.py` |
| 5 | Imbalance ratio experiment: take a balanced dataset and progressively increase the imbalance… | code | T0 | `ex05_smote_only_helps_an_undertrained_model.py` |
<!-- generated:end -->

## Answers

The lesson is 352 lines of numpy: SMOTE through a pure-Python k-NN, random
over- and undersampling, class weights, and a gradient-descent logistic
regression (`lr=0.1`, 300 epochs) that every method trains. All five exercises
are **T0** against that code. Two things recur. First, in 2-D every resampling
of a linear model leaves the test ranking (AUC) essentially unchanged and moves
only the intercept, so F1 gaps at threshold 0.5 are threshold gaps. Second, the
lesson's 300-epoch fit is far from converged, and several "wins" are really
fixes for that.

### 1 — the borderline rule is vacuous under overlap

Minority centre moved to (1.5, 1.5) so the classes overlap; 5 seeds:

| method | F1 | test AUC |
|---|---:|---:|
| no resampling | 0.179 | 0.9454 |
| SMOTE | **0.335** | 0.9458 |
| Borderline-SMOTE | 0.324 | 0.9452 |

**ANSWER: Borderline-SMOTE does not beat SMOTE.**

**FINDING: all three rank the test set the same.** Oversampling moves the
intercept (−2.92 to −1.95), not the weight direction.

**FINDING: the exercise's rule ("any majority neighbour") selects every point
once overlap is heavy.** At (1, 1) all 40.4 minority points qualify and the
output is bit-identical to the lesson's `smote`; 16.8 of them have *only*
majority neighbours, the noise points Han et al. exclude.

**CONTROL:** on the lesson's own (2.5, 2.5) data the rule keeps 14.0 of 40.4,
and F1 orders Borderline 0.460 < SMOTE 0.552 < none 0.818.

### 2 — the cost rule needs calibration the lesson lacks

`min_cost_predict(probs, cost)` predicts positive where expected cost is lower,
which is threshold `C_FP/(C_FP+C_FN)`. On a fresh 10,000-row test set:

| cost ratio | threshold | converged P / R | lesson-fit flags |
|---|---:|---:|---:|
| 1:10 | 0.0909 | 0.84 / 0.94 | 21.65% |
| 1:100 | 0.0099 | 0.60 / 1.00 | 93.87% |
| 1:1000 | 0.0010 | 0.36 / 1.00 | 99.99% |

**ANSWER: precision collapses, recall saturates.**

**FINDING: on the lesson's 300-epoch fit the rule flags nearly everything.**
Mean predicted probability is 0.080 against a 0.050 base rate; at 1:10 the
rule costs **1665** against **263** for the best threshold on the same scores.

**FINDING: `find_optimal_threshold` sweeps 0.05–0.95**, so it cannot reach the
1:100 or 1:1000 thresholds: its best costs 15,137 at 1:1000 against 410.

**CONTROL:** at 1:1 the rule is the 0.5 default, 107 errors against a best of 104.

### 3 — Platt only calibrates when converged

Platt scaling of the lesson's class-weighted model on its 200-row validation split:

| | AUC | AP | ECE | Brier | mean p |
|---|---:|---:|---:|---:|---:|
| raw | 0.99808 | 0.9660 | 0.127 | 0.0427 | 0.175 |
| Platt, converged (a = 5.70) | 0.99808 | 0.9660 | **0.009** | 0.0101 | 0.059 |
| Platt, lesson defaults (a = 0.79) | 0.99808 | 0.9660 | 0.118 | — | — |

**ANSWER: same ranking, same PR curve (identical row order), honest
probabilities** — base rate 0.050.

**FINDING: with `logistic_regression_weighted`'s default lr 0.01 / 200 epochs,
Platt barely moves.**

**FINDING: the lesson's tuned threshold is set by two points.** The validation
split's 10 positives separate perfectly (top negative 0.667, bottom positive
0.706), F1 = 1.0 across the gap, and the sweep returns its low edge: test F1
0.858 there against 0.896 at the best threshold.

**CONTROL:** a negated slope gives AUC 0.0019 = 1 − 0.99808.

### 4 — balanced bagging is noisier than one SMOTE model

Ten runs, 15 models per bag, F1 at 0.5 on 10,000 fresh rows:

| | data fixed | fresh data each run |
|---|---:|---:|
| one SMOTE model | **0.683 ± 0.002** | 0.648 ± 0.020 |
| 15-model balanced bag | 0.659 ± 0.015 | 0.623 ± 0.022 |
| one undersampled model | 0.628 ± 0.051 | — |

**ANSWER: SMOTE edges the bag and varies far less.**

**FINDING: bagging cuts undersampling's spread 3.4x but is still 7.5x
SMOTE's**; with fresh data each run, the data sets the spread, not the sampler.

**CONTROL:** mean test AUC 0.9981 for bag and SMOTE alike.

### 5 — SMOTE only helps an under-trained model

F1 without / with SMOTE, 5 seeds:

| ratio | lesson fit (300 epochs) | converged |
|---|---:|---:|
| 50/50 | 0.964 / 0.964 | — |
| 70/30 | 0.954 / 0.923 | 0.960 / 0.958 |
| 90/10 | 0.914 / 0.776 | 0.918 / 0.895 |
| 95/5 | 0.806 / 0.632 | 0.886 / 0.822 |
| 99/1 | **0.000 / 0.240** | 0.767 / 0.536 |

**ANSWER: SMOTE makes a meaningful difference from 90/10 — a loss — and its
only gain is at 99/1.**

**FINDING: that gain rescues an unconverged fit.** Converged, the untreated
model scores 0.767 at 99/1 and SMOTE hurts at every imbalanced ratio.

**CONTROL:** the lesson's `random_oversample` lands within 0.02 of SMOTE
everywhere.
