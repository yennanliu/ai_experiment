<!-- generated:start -->
# 02-ml-fundamentals / 16-anomaly-detection

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/02-ml-fundamentals/16-anomaly-detection/) · upstream spec
`phases/02-ml-fundamentals/16-anomaly-detection/docs/en.md`

```bash
uv run demo practice run 16-anomaly-detection --ex 1
uv run demo explain 16-anomaly-detection --ex 1
uv run pytest demos/phases/02-ml-fundamentals/16-anomaly-detection
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Threshold tuning. Run the Z-score detector with thresholds from 1.0 to 5.0 in steps of 0.5. P… | code | T0 | `ex01_threshold_3_works_only_on_seed_42.py` |
| 2 | Multivariate anomalies. Create 2D data where each feature individually looks normal, but the… | code | T0 | `ex02_isolation_forest_catches_only_half_of_off_diagonal_points.py` |
| 3 | LOF from scratch. Implement Local Outlier Factor using k-nearest neighbors. Compare against s… | code | T0 | `ex03_lof_matches_sklearn_and_k_barely_matters_below_cluster_size.py` |
| 4 | Streaming anomaly detection. Modify the Z-score detector to work in a streaming setting: upda… | code | T0 | `ex04_streaming_matches_batch_until_it_refuses_to_learn_from_anomalies.py` |
| 5 | Real-world evaluation. Take a dataset with known anomalies (credit card fraud from Kaggle, fo… | code | T0 | `ex05_the_winner_flips_on_one_irrelevant_amount_column.py` |
<!-- generated:end -->

## Answers

The lesson is 340 lines of numpy: per-feature `zscore_detect` and `iqr_detect`
(each row is scored by its worst feature), a from-scratch `IsolationForest`,
and two generators. Every exercise runs at **T0**. Exercise 3 compares against
scikit-learn's `LocalOutlierFactor`, and exercise 5 ships the scaled-down
runnable `DESIGN D11` requires in place of the Kaggle download.

### 1 — the sweet spot at 3.0 belongs to seed 42

| threshold | 1.0 | 2.0 | 2.5 | **3.0** | 3.5 | 4.0 | 5.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| precision | 0.16 | 0.89 | 0.96 | **1.00** | 1.00 | 1.00 | 1.00 |
| recall | 1.00 | 1.00 | 1.00 | **1.00** | 0.76 | 0.52 | 0.04 |

**ANSWER: 3.0.** The classes separate perfectly: any threshold in
**(2.58, 3.04)** gives F1 = 1.

**FINDING: on seeds 0–19 the window always ends below 3.0**, with upper ends
from 2.16 to 2.97. Threshold 3.0 loses recall on 20 of 20, even though the
classes separate on 15.

**FINDING: 3.0 works because the anomalies inflate the std.** Twenty-five
anomalies (4.8% of the points) raise the per-feature std **1.52x and 1.46x**,
so "3 sigma" here means about 4.5 sigma of the normal data. With clean
statistics the window moves to (3.85, 4.38), and 3.0 has precision 0.89.

### 2 — Isolation Forest finds only about half

Over 5 seeds: correlation 0.95, with 25 anomalies 5.4–8.9 sigma off the
diagonal and every coordinate within |2.77|.

| detector | P@25 |
|---|---:|
| z-score (threshold 3 flags **0 of 25**) | 0.33 |
| Isolation Forest (lesson) | **0.55** |
| Isolation Forest (sklearn) | 0.54 |
| Mahalanobis | 1.00 |

**ANSWER:** the z-score misses all of them. Isolation Forest ranks more of them
first, but only about half.

**FINDING: axis-parallel splits cut off the tips of the diagonal as easily as
the off-diagonal points.** The normals in its top 25 average |x| = 2.71. sklearn
behaves the same way, so this is the algorithm, not the lesson's code. Without
correlation, Isolation Forest reaches 0.90 against Mahalanobis at 0.98.

### 3 — LOF matches sklearn, and k barely matters until it reaches the cluster size

| k | max \|mine − sklearn\| | P@20 | AP | anomalies > 1.5 | normals > 1.5 |
|---:|---:|---:|---:|---:|---:|
| 10 | 1.5e-9 | 0.85 | 0.840 | 17 | 18 |
| 50 | 1.2e-9 | 0.85 | 0.875 | 17 | 26 |
| 200 | 9.4e-11 | — | — | 8 | 0 |

sklearn adds 1e-10 to the reachability mean, which explains the 1e-9 differences.

**ANSWER: k=10 against k=50 hardly changes the ranking.** k=200, the size of one
cluster, mixes clusters, and only 8 anomalies stay above 1.5.

**FINDING: 0.85 is the ceiling.** 3 of the 20 uniform "anomalies" land inside a
cluster, within 2.35 Mahalanobis units of its centre (k=10 ranks 106, 143 and
353). At k=50 the other 17 hold ranks 0–16 exactly.

**FINDING: the score scale moves with k.** A fixed LOF threshold flags 18, 26
and then 0 normals as k grows.

### 4 — streaming matches batch; skipping anomalies costs precision

| detector | flags | precision | recall |
|---|---:|---:|---:|
| batch `zscore_detect` | 25 | 1.00 | 1.00 |
| Welford, streaming | 25 | 0.96 | 0.96 |
| Welford, skip flagged points | 38 | **0.66** | 1.00 |

**ANSWER:** streaming and batch disagree on 2 of 525 points: a normal at
position 2 (two points of history) and an anomaly at position 48, where two
earlier anomalies had pushed the running std to 1.13. The final running stats
equal the batch stats within 2.4e-15.

**FINDING: "don't learn from anomalies" makes it worse.** The running std stays
clean, so 3.0 becomes a tighter cut. Exercise 1 explains why: the batch
detector's precision depends on contamination.

**CONTROL:** at an offset of 1e8, the one-pass `E[x²] − E[x]²` returns a
variance of **−2.0**. Welford returns 0.975268, matching the true value.

### 5 — the winner flips on one irrelevant column

The fixture has 20,000 rows at Kaggle's 0.172% fraud rate (34 frauds), with
frauds shifted −4 in 5 of 28 components. Table entries are AUPRC:

| variant | z-score | IQR | Isolation Forest | LOF (k=20) |
|---|---:|---:|---:|---:|
| Gaussian, no Amount | 0.965 | **0.971** | 0.560 | 0.001 |
| + log-normal Amount (same in both classes) | 0.114 | 0.015 | 0.349 | **0.487** |
| heavy-tailed t(3) + Amount | 0.005 | 0.004 | **0.127** | 0.012 |

**ANSWER: it depends on the nuisance features, not the fraud signal.** On the
Kaggle-like variant Isolation Forest wins, weakly, with P@100 0.09.

**FINDING: one irrelevant column flips the winner.** `zscore_detect` and
`iqr_detect` score a row by its worst feature, so one skewed Amount column
outvotes 5 informative ones.

**FINDING: LOF ranks a fraud ring below chance.** The 34 frauds form a cluster,
and with k=20 < 34 they are each other's neighbours: AUROC **0.336**. AUPRC is
0.456 at k=50 and 0.998 at k=100.

**FINDING: "AUROC is misleading" holds twice.** Isolation Forest scores AUROC
0.967 with AUPRC 0.127. The z-score with Amount scores AUROC 0.994 with AUPRC
0.114.

**CONTROL:** a vectorised walk of the lesson's fitted trees equals its
`anomaly_score` exactly. With 34 frauds, P@100 is capped at 0.34 and P@500 at
0.068. On Kaggle's 492 frauds the caps are 1.0 and 0.984.

## A note on the reference code

`docs/en.md` describes a third demo scenario ("50 features, but anomalies
differ in only 5 of them") and says the code "compares from-scratch
implementations against sklearn". `code/anomaly_detection.py` has neither: it
has two generators and no sklearn import. Exercise 5's fixture is the closest
thing to the missing scenario.
