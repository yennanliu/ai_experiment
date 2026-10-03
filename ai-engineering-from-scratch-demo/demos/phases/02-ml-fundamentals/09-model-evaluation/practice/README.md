<!-- generated:start -->
# 02-ml-fundamentals / 09-model-evaluation

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/02-ml-fundamentals/09-model-evaluation/) · upstream spec
`phases/02-ml-fundamentals/09-model-evaluation/docs/en.md`

```bash
uv run demo practice run 09-model-evaluation --ex 1
uv run demo explain 09-model-evaluation --ex 1
uv run pytest demos/phases/02-ml-fundamentals/09-model-evaluation
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Implement precision-recall curves: plot precision vs recall at different thresholds. Compute… | code | T0 | `ex01_roc_cannot_see_prevalence_ap_can.py` |
| 2 | Build a nested cross-validation loop: the outer loop evaluates model performance, the inner l… | code | T0 | `ex02_stratified_folds_dump_the_minority_in_the_last_fold.py` |
| 3 | Implement a permutation test for model comparison: shuffle the labels, retrain, and measure p… | code | T0 | `ex03_the_lessons_reseeding_makes_every_permutation_identical.py` |
<!-- generated:end -->

## Answers

The lesson is one stdlib file: split and fold helpers, counting metrics, a
trapezoid `auc_roc`, `SimpleLogistic` trained by SGD, and three data
generators. All three exercises are **T0** against that code with no
dependencies.

### 1 — ROC cannot see prevalence; average precision can

The ranking is held fixed by replicating the negatives of one balanced pool, so
each class's score distribution is unchanged and only the ratio moves.

| positives | AUC | AP | precision at 80% recall | FPR at 80% recall |
|---|---:|---:|---:|---:|
| 48% | 0.9682 | 0.967 | 0.91 | 0.048 |
| 9% | 0.9682 | 0.846 | 0.53 | 0.048 |
| 5% | 0.9682 | **0.787** | **0.35** | 0.048 |

**ANSWER: AUC does not move at all, while AP falls 0.967 -> 0.787.** The ROC
curve is literally the same curve, because TPR and FPR each divide by one
class. Use ROC for ranking quality, or when the deployed prevalence is unknown;
use PR when positives are rare and the cost is in how many alerts are false.

**FINDING: the ROC operating point is identical while nearly two alarms in
three turn false**: FPR 0.048 at every ratio, precision 0.91 -> 0.35.

**FINDING: the lesson's own imbalanced data cannot show the difference.**
`make_imbalanced_data` puts the minority at (3, 3) against N(0, 1), and the
lesson's split leaves **3 positives in 60** test rows: AUC 1.0000, AP 1.0000.

**CONTROL:** `auc_roc` equals the Mann-Whitney P(score_pos > score_neg)
(0.968159 both).

### 2 — nested CV shows the shortcut flatters kNN; stratified folds dump the minority last

`SimpleLogistic` (epochs 1-30) against a kNN (k 1-31): outer loop the lesson's
`kfold_split`, inner loop its `cross_validate` on the outer training part.

| | logistic | kNN | gap |
|---|---:|---:|---:|
| nested, 8 real datasets | 0.8865 | 0.8583 | **0.0281** |
| best-of-grid CV, same | 0.8969 | 0.8781 | 0.0187 |
| nested, 24 random-label sets | 0.5052 | 0.5031 | — |
| best-of-grid, random labels | 0.5038 | **0.5372** | — |

**ANSWER: logistic wins by 2.8 points nested and 1.9 by the shortcut.**

**FINDING: the leak is all kNN's.** On random labels kNN's best grid point
claims 53.7%; logistic's grid of near-identical linear models claims 50.4%. The
shortcut's optimism scales with how different the grid's models are, so it is
unfair to the less flexible model and understates logistic's lead by a third.

**FINDING: `stratified_kfold_split` puts each class's remainder in the last
fold.** On `make_imbalanced_data(300, 0.05)`'s 13 positives it gives
**[2, 2, 2, 2, 5]** (round-robin would give 3, 3, 3, 2, 2); with 4 positives
**[0, 0, 0, 0, 4]** — all the minority in one fold, the failure `docs/en.md`
says stratification exists to prevent.

**CONTROL:** the outer folds partition all 120 rows.

### 3 — the lesson's reseeding makes every permutation identical

**ANSWER: observed 5-fold accuracy 0.84 against a null of mean 0.503 (0.36-0.62),
p = 1/101 = 0.0099**, the floor for 100 permutations.

**FINDING: shuffle with `random.shuffle` and all 100 permutations are one
permutation.** `cross_validate` calls `kfold_split`, which calls
`random.seed(42)` on the global RNG, so every shuffle after a retrain starts
from the same state. The null is a single score (0.52), and p can only be 1/101
or 1. Seven functions in the lesson's file reseed the global RNG.

**FINDING: on random labels that test rejects 3 times in 10** (p = 0.0099 or
1.0, nothing between); a private `random.Random` rejects 0 times, with p from
0.158 to 0.802.

**CONTROL:** the private RNG gives 100 distinct label orders and 31 distinct
null scores.
