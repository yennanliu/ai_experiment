<!-- generated:start -->
# 02-ml-fundamentals / 11-ensemble-methods

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/02-ml-fundamentals/11-ensemble-methods/) · upstream spec
`phases/02-ml-fundamentals/11-ensemble-methods/docs/en.md`

```bash
uv run demo practice run 11-ensemble-methods --ex 1
uv run demo explain 11-ensemble-methods --ex 1
uv run pytest demos/phases/02-ml-fundamentals/11-ensemble-methods
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Modify the AdaBoost implementation to track training accuracy after each round. Plot accuracy… | code | T0 | `ex01_train_converges_at_142_test_at_20.py` |
| 2 | Implement a random forest from scratch by adding random feature subsampling to the regression… | code | T0 | `ex02_forest_cuts_variance_75pct_and_pays_in_bias.py` |
| 3 | In the gradient boosting implementation, add early stopping: track validation loss after each… | code | T0 | `ex03_needs_188_trees_and_lr_0_5_needs_43.py` |
| 4 | Build a stacking ensemble with three base models (logistic regression, decision tree, k-neare… | code | T0 | `ex04_stack_wins_4_of_5_lesson_stack_is_a_vote.py` |
| 5 | Run XGBoost on the same dataset with default parameters. Compare its accuracy to your from-sc… | code | T1 | `ex05_speed_gap_grows_with_n_accuracy_gap_is_the_loss.py` |
<!-- generated:end -->

## Answers

The lesson is one numpy file: a decision stump, AdaBoost, a variance-split
regression tree, gradient boosting on squared error, bagging and a stacking
classifier. Every exercise runs at **T0** against it. Exercise 4 adds
scikit-learn's base models (the exercise names them), and exercise 5 stands
scikit-learn's histogram booster in for XGBoost, which is not a dependency here.

### 1 — training accuracy converges at round 142, test accuracy by round 20

The lesson's `AdaBoostScratch`, 150 rounds, on its own demo data (320 / 80):

| rounds | 1 | 2 | 3 | 10 | 20 | 50 | 100 | 150 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 0.8281 | 0.8281 | 0.8812 | 0.9219 | 0.9531 | 0.9812 | 0.9938 | **1.0000** |
| test | 0.7500 | 0.7500 | 0.7625 | 0.7875 | **0.8250** | 0.8375 | 0.8250 | 0.8250 |

**ANSWER:** training accuracy first touches 1.0 at round 112, loses it, and
holds it only from **round 142**; test accuracy reaches 0.8250 at **round 20**
and then wanders within three test points. The rounds in between memorise.

**FINDING: the curve is not monotone, but AdaBoost's objective is.** Training
accuracy drops on **40 of 149** rounds while the mean exponential loss rises on
none — it equals ∏ sech(α_t) to 3.3e-16, the exact AdaBoost normaliser.

**FINDING: round 2 changes nothing.** α₁ = 0.786 > α₂ = 0.532, so two weighted
voters are a dictatorship: sign(α₁h₁ + α₂h₂) = sign(h₁) everywhere.

### 2 — a forest cuts variance by 75% and pays for it in bias

The forest subclasses the lesson's `SimpleRegressionTree`: at each node all but
2 of 5 features are hidden as constant columns and the lesson's own `_build`
picks the split. Five 200-row training sets, depth 4, scored against the true
function:

| | variance | tree correlation | bias² | bias² + variance |
|---|---:|---:|---:|---:|
| one tree | 0.444 | — | 0.771 | 1.215 |
| bagging, 100 trees | 0.150 | 0.240 | **0.722** | **0.872** |
| random forest, 100 trees | **0.113** | **0.065** | 1.016 | 1.129 |

**ANSWER:** variance falls **75%** from one tree, more than bagging's 66%.

**FINDING: the decorrelation is bought by making each tree worse** — one
random-feature tree varies 1.530, 3.4x a full-feature tree — **and by bias.**
x0 carries the slope-2 term, is offered at 40% of splits, and is the root of
exactly 40% of trees. At √5 → 2 features on a problem with one dominant input,
the forest loses to plain bagging. The doc's table says only that forests "do
not help with bias".

The lesson's `demo_bagging` prints `bag_acc - single_acc` under the label
"Variance reduction". It never redraws a training set, so it cannot measure one.

### 3 — 188 trees at lr 0.1; lr 0.5 needs 43 and does as well

Tree-at-a-time on 240 fit / 80 validation / 80 test rows, stop after 10 rounds
without a new validation minimum:

| | trees | validation MSE | test MSE |
|---|---:|---:|---:|
| lesson default (lr 0.1, 100 trees) | 100 | 0.4896 | 0.4475 |
| **lr 0.1, patience 10** | **188** | 0.4131 | **0.4210** |
| lr 0.1, patience 20 | 232 | 0.4071 | 0.4205 |
| lr 0.5, patience 10 | 43 | — | 0.4208 |

**ANSWER: 188 trees** — almost twice the lesson's default, which leaves test MSE
6% worse. Patience 10 stops inside a 13-round plateau; patience 20 gets 1.4%
more out of the *validation* set and nothing out of the test set.

**FINDING:** at lr 0.5 the rule keeps 43 trees and reaches the same test MSE
(0.4208 against 0.4210). The lesson prints "Lower learning rates need more trees
but often generalize better"; with early stopping the second half does not show.

### 4 — stacking wins 4 of 5; the lesson's stacker is a majority vote

Logistic regression, a depth-5 tree and 15-NN, a logistic meta-learner on
out-of-fold probabilities; five 400-row training sets, one 4,000-row test set:

| | logreg | tree | k-NN | stack | leaky stack | lesson's stacker |
|---|---:|---:|---:|---:|---:|---:|
| mean accuracy | 0.8600 | 0.8733 | 0.8978 | **0.9056** | 0.8841 | 0.8980 |

**ANSWER:** stacking beats the best base model on **4 of 5** sets, by 0.78
points on average; k-NN alone wins the fifth.

**FINDING: in-sample meta-features double the tree's weight** (1.74–2.04 →
4.25–4.97) and lose on all five sets — the leak the doc warns about, measured.

**FINDING: the lesson's `StackingClassifier` is a vote.** It feeds hard ±1
labels to a tanh meta-learner; with three voters it agrees with the plain
majority on 92.8–100% of rows (exactly on 2 of 5 sets). Probabilities are what
let a meta-learner do more than count.

**CONTROL:** scikit-learn's `StackingClassifier` with the same folds disagrees on
0 of 20,000 predictions.

### 5 — the speed gap grows with n; the accuracy gap is the loss

`HistGradientBoostingClassifier` (histogram, second-order, log-loss — XGBoost's
recipe) at defaults, against `GradientBoostingScratch`. Timings are one run on
the author's machine; the checks assert orderings only.

| | fit, 400 rows | per tree, 400 rows | per tree, 20,000 rows | accuracy, 4,000 rows |
|---|---:|---:|---:|---:|
| scratch GBM | ~0.78 s | ~9 ms | ~64 ms | 0.9042 |
| histogram booster | ~0.13 s | ~1.2 ms | ~3.4 ms | **0.9220** |

**ANSWER: about 6x faster at the lesson's size**, growing to **about 19x per
tree at 20,000 rows**: the scratch tree's cost grows with n, binning makes the
booster's split search nearly independent of it.

**FINDING: the lesson's 100-row test set triples the accuracy gap** — 0.89 vs
0.95 there, 1.8 points on 4,000 rows.

**FINDING: what is left is the loss function.** The booster capped at depth 3
still scores 0.9235; scikit-learn's exact GBM on the scratch version's objective
(squared error on ±1 labels) scores 0.8985 and agrees with the scratch GBM on
94.9% of rows.

The T3 run is `pip install xgboost`, then `XGBClassifier().fit(X_train, y_train)`.
