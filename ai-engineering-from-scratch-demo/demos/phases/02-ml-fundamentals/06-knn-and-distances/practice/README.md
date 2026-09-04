<!-- generated:start -->
# 02-ml-fundamentals / 06-knn-and-distances

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/02-ml-fundamentals/06-knn-and-distances/) · upstream spec
`phases/02-ml-fundamentals/06-knn-and-distances/docs/en.md`

```bash
uv run demo practice run 06-knn-and-distances --ex 1
uv run demo explain 06-knn-and-distances --ex 1
uv run pytest demos/phases/02-ml-fundamentals/06-knn-and-distances
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Implement KNN classification on a 2D dataset with 3 classes. Plot the decision boundary for K… | code | T0 | `ex01_k_sweep.py` |
| 2 | Generate 1000 random points in 2, 5, 10, 50, 100, and 500 dimensions. For each dimensionality… | code | T0 | `ex02_curse_of_dimensionality.py` |
| 3 | Compare L1, L2, and cosine distance for KNN on a text classification problem (use TF-IDF vect… | code | T0 | `ex03_text_distance_metrics.py` |
| 4 | Implement a KD-tree and measure query time vs brute force for datasets of 1k, 10k, and 100k p… | code | T1 | `ex04_kdtree_vs_brute.py` |
| 5 | Build a weighted KNN regressor for y = sin(x) + noise. Compare it with unweighted KNN for K=3… | code | T0 | `ex05_weighted_knn.py` |
<!-- generated:end -->

## Answers

**1 — the two ends fail differently, and both are exactly characterisable.**

| K | train | test |
|---:|---:|---:|
| 1 | **100.0%** | 96.7% |
| 5 | 98.3% | 96.7% |
| **15** | 97.5% | **98.3%** |
| 240 (=N) | 35.4% | **21.7%** |

K=1's perfect training score is not evidence — every training point is its own
nearest neighbour, so the label is returned unchanged. K=N emits a **single**
label across the whole test set and scores **21.7%**, *below* the 33.3% a coin
gets over 3 classes, because the training majority (35.4%) is not the test
majority (50.0%): predicting the wrong constant is worse than guessing.

Note the train-test gap at K=N is +13.8% with *both* terms near zero — a small
gap is no evidence of a good model.

**2 — the ratio collapses 134×, and √d is why.**

| d | max/min | (max−min)/mean | mean/√d |
|---:|---:|---:|---:|
| 2 | 169.78 | 3.463 | 1.218 |
| 10 | 10.98 | 1.736 | 1.375 |
| 500 | **1.27** | 0.235 | 1.416 |

At d=500 the furthest of 7,140 pairs is only 27% further apart than the closest.
The relative spread confirms it using *every* pair rather than the two extremes —
worth having, because max/min is set by two observations and one unusually close
pair moves it a lot.

The mechanism is in the last column: mean distance / √d is nearly constant, so
mean distance ≈ √(2d) for unit Gaussians. Distances all grow together; the
differences between them do not. The ratio is what's left over.

**3 — cosine wins, at 96.7% against L1 90.0% and L2 86.7%.**

The reason is measurable: TF-IDF norms span **11.9×** (3.09 to 36.74) because
documents are 5, 10 or 60 tokens long. L1 and L2 read a long document as far from
a short one on the same topic; cosine divides that magnitude out and compares
direction only. L2 is worst because squaring amplifies exactly the difference
that is the problem.

Both corpus conditions are needed for the exercise to say anything. With disjoint
topic vocabularies and uniform lengths all three metrics score **100%** and the
comparison is vacuous — which is what a first attempt produces. This corpus shares
7 common words and varies lengths deliberately.

**4 — the KD-tree stops winning at about d=10, and is slower by d=50.**

| | n=1k | n=5k | n=20k |
|---|---:|---:|---:|
| d=2 | 25× | 109× | **413×** |
| d=10 | 1.00× | 1.18× | 1.87× |
| d=50 | 0.97× | 0.98× | **0.94×** |

At d=2 the advantage *grows* with n, because the tree is O(log n) per query where
the scan is O(n). At d=50 the tree is doing all of the scan's distance work plus
traversal bookkeeping on top.

A branch is pruned only when the splitting coordinate's gap alone exceeds the
current best radius — and by exercise 2's measurement, in high dimensions every
point is roughly equidistant, so almost nothing prunes and the tree visits nearly
every leaf.

(Sizes stop at 20k rather than 100k: brute force here is a pure-Python O(n) scan
per query and 100k × 50D takes minutes without changing the answer.)

**5 — the exercise's claim is false. Weighting is rougher, not smoother.**

| K | plain RMSE | weighted RMSE | plain roughness | weighted roughness |
|---:|---:|---:|---:|---:|
| 3 | **0.1207** | 0.1494 | **0.1090** | 0.1433 |
| 10 | **0.0865** | 0.1199 | **0.0708** | 0.1245 |
| 30 | 0.1560 | **0.1115** | **0.0475** | 0.1092 |

Measured as mean |Δ prediction| between neighbouring x, weighting is rougher at
**every** K — the nearest neighbour dominates the weighted average, so the
prediction tracks it and jumps when it changes. The exercise says the opposite,
and says it most strongly for large K, where the gap is widest in the other
direction (0.1092 against 0.0475).

What weighting actually buys is **accuracy at large K**: 0.1115 against 0.1560 at
K=30, where unweighted averaging drags in all 30 neighbours at full strength
including ones far enough away that sin(x) has moved. And it *costs* accuracy at
small K (0.1494 against 0.1207), where three nearby neighbours are all equally
informative and the plain mean averages their noise away.

One measurement choice was mandatory. The weight is 1/(d + 1e-10), so a query
that is also a training point gets weight **1e10** and the prediction collapses to
that point's own label — RMSE 9e-08 against the training labels at *every* K,
with K no longer mattering at all. Predicting on the training set makes this
comparison meaningless, so it uses a held-out split.
