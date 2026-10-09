<!-- generated:start -->
# 02-ml-fundamentals / 18-feature-selection

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/02-ml-fundamentals/18-feature-selection/) · upstream spec
`phases/02-ml-fundamentals/18-feature-selection/docs/en.md`

```bash
uv run demo practice run 18-feature-selection --ex 1
uv run demo explain 18-feature-selection --ex 1
uv run pytest demos/phases/02-ml-fundamentals/18-feature-selection
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Forward selection: implement the opposite of RFE. Start with zero features. At each step, add… | code | T0 | `ex01_rfe_spends_three_slots_on_one_signal.py` |
| 2 | Stability selection: run L1 feature selection 50 times, each time on a random 80% subsample o… | code | T0 | `ex02_l1_keeps_every_copy_and_stability_agrees.py` |
| 3 | Multicollinearity detection: compute the correlation matrix for all features. Implement a fun… | code | T0 | `ex03_mi_tiebreak_drops_the_true_features.py` |
| 4 | Feature selection pipeline: chain variance threshold, mutual information filter, and RFE into… | code | T0 | `ex04_mi_filter_discards_a_true_signal.py` |
| 5 | Permutation importance from scratch: implement permutation importance. For each feature, shuf… | code | T0 | `ex05_tree_importance_buries_x3_under_noise.py` |
<!-- generated:end -->

## Answers

The lesson is 353 lines of numpy: variance threshold, binned mutual
information, RFE and L1 over a gradient-descent logistic regression, and a
bagged-tree importance, all on a 500-row dataset whose target is
`2 x1 − 1.5 x2 + x3 + noise`. All five exercises are **T0** against that code.
The dataset's labels mislead: `info_3` and `info_4` are near-copies of x1 and
x2 (r ≈ 0.995), so "informative" already contains redundancy, and every
method below is judged by whether it finds **one copy each of x1, x2, x3**.
Accuracies are also measured on 5000 fresh rows from the same generator,
because the lesson's 100 test rows cannot tell 0.92 from 0.93.

### 1 — RFE spends three slots on one signal

| | selected | fits | feature-columns | 100 test rows | 5000 fresh rows |
|---|---|---:|---:|---:|---:|
| forward | info_0, info_4, info_2 | 74 | 180 | 0.94 | **0.929** |
| RFE to 5 | info_0, info_1, info_2, info_3, corr_0 | 15 | 195 | **0.96** | 0.919 |
| all 20 | — | — | — | — | 0.912 |

**ANSWER: forward selection stops at one copy per signal; three of RFE's five
are copies of x1.**

**FINDING: RFE is faster by fits (4.9x) but not by work** (195 columns vs 180).

**FINDING: the lesson's 100-row test set ranks them backwards** — two rows.

**CONTROL:** x1, x2, x3 themselves score 0.930, so forward selection found them.

### 2 — L1 keeps every copy, and stability selection agrees

50 runs on 320-row subsamples, alpha from 0.04–0.06:

**ANSWER: the stable set is the single run's 8 features** (info_0–4,
corr_0–2). Stability adds one thing the single run hides: noise_9 enters
**38%** of runs.

**FINDING: L1 does not "pick one and zero the others".** The doc says it does;
the three x1 copies get weights 0.554, 0.553, 0.534, and still 0.846, 0.712,
0.277 after 50x the epochs.

**FINDING: stability cannot see redundancy** — all three copies are selected
in 100% of runs.

**CONTROL:** nine noise columns and the mixtures corr_3, corr_4 are selected in
0 of 50.

### 3 — the MI tiebreak drops the true features

Pairs with |r| > 0.9, keep the higher `mutual_information`:

| pair | kept (MI) | dropped (MI) |
|---|---|---|
| x2 copies | info_4 (0.104) | **info_1 = x2** (0.099) |
| x3 copies | corr_2 (0.022) | **info_2 = x3** (0.021) |

**ANSWER: five removed (info_1, info_3, corr_0, corr_1, info_2), one survivor
per signal** — but corr_2, corr_3 and corr_4 stay.

**FINDING: two of the three generating features lose to noisier copies**, by
margins smaller than the 0.011 that pure noise averages.

**FINDING: pairwise correlation misses mixtures.** corr_3 has max |r| 0.672 to
any survivor but R² **0.975** on them (VIF 39).

**CONTROL:** the noise columns survive with R² ≤ 0.040.

### 4 — the pipeline's MI filter discards a true signal

| | RFE fits / columns | 5000 fresh rows | lesson test |
|---|---:|---:|---:|
| variance → MI top half → RFE | 5 / 40 | 0.854 | 0.90 |
| RFE alone | 15 / 195 | **0.919** | 0.96 |

**ANSWER: faster (4.9x less work), not equally accurate.**

**FINDING: binned MI ranks x3 11th of 20, below noise_9 and noise_5**, so the
pipeline ends with three x1 copies, two x2 copies and no x3.

**FINDING: the variance stage removes nothing** — smallest variance 0.217.

**CONTROL:** without the MI stage the pipeline selects exactly RFE's five.

### 5 — tree importance buries x3 under noise

| feature | permutation rank | tree rank |
|---|---:|---:|
| info_2 (x3) | **1** | 14 |
| corr_3 (mixture) | 18 | 6 |

**ANSWER: they disagree on x3 and on the mixture**; rank correlation 0.54.

**FINDING: the cause is in-sample credit, not cardinality.** Every column has
400 distinct values, yet noise takes **22%** of tree importance against a summed
permutation drop of −0.005.

**FINDING: single-column shuffles understate correlated groups.** Shuffling the
three x1 copies together drops F1 by **0.293**; their single drops sum to 0.155.

**CONTROL:** noise columns average a permutation drop of −0.0005.
