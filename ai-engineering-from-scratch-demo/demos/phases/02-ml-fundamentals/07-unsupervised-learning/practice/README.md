<!-- generated:start -->
# 02-ml-fundamentals / 07-unsupervised-learning

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/02-ml-fundamentals/07-unsupervised-learning/) · upstream spec
`phases/02-ml-fundamentals/07-unsupervised-learning/docs/en.md`

```bash
uv run demo practice run 07-unsupervised-learning --ex 1
uv run demo explain 07-unsupervised-learning --ex 1
uv run pytest demos/phases/02-ml-fundamentals/07-unsupervised-learning
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Implement K-Means++ initialization: instead of picking random centroids, pick the first rando… | code | T0 | `ex01_kmeans_plus_plus.py` |
| 2 | Add hierarchical agglomerative clustering to the code. Implement Ward's linkage and produce a… | code | T0 | `ex02_agglomerative_linkage.py` |
| 3 | Build a simple anomaly detection pipeline: run DBSCAN and GMM on the same data, flag points t… | code | T0 | `ex03_outlier_agreement.py` |
<!-- generated:end -->

## Answers

**1 — k-means++ converges faster, and that is the only thing it reliably buys.**

40 seeds per arm, plain Lloyd iteration in both, so the initialisation is the only
difference.

| layout | arm | mean iterations | mean inertia | best inertia |
|---|---|---:|---:|---:|
| 9 tight clusters | random | 15.12 | 414.59 | 379.93 |
| 9 tight clusters | **k-means++** | **12.20** | **400.44** | 379.80 |
| 4 separated blobs | random | 8.55 | **485.41** | — |
| 4 separated blobs | **k-means++** | **7.15** | 486.23 | — |

19% fewer iterations on the hard layout, and better mean inertia there too.

Two things the single-layout version of this experiment hides. On the **easy**
layout the inertia advantage *disappears* — k-means++ ends marginally worse. With
four well-separated blobs random initialisation rarely goes wrong, so a better
seeding has nothing to fix. The benefit is insurance against bad starts, not a
free improvement, and only a layout where random init can actually fail shows it.

And the **best** run over 40 seeds is the same in both arms: 379.80 against
379.93, a difference of 0.13 in inertia. k-means++ does not find better optima. It
finds the same one more often and sooner. Lloyd's algorithm is identical in both
arms; only where it starts differs.

**2 — single linkage recovers the moons exactly; Ward and K-Means cannot.**

Adjusted-Rand-style agreement against the true labels, 200 points, k=2:

| method | moons | separated blobs |
|---|---:|---:|
| **single linkage** | **1.000** | 1.000 |
| complete linkage | 0.850 | 1.000 |
| Ward | 0.760 | 1.000 |
| K-Means | 0.750 | 1.000 |

The blobs column is why the comparison had to move: on well-separated spherical
clusters *everything* scores 1.000 and the exercise says nothing. Two interleaving
crescents separate the methods, because they are the case where "cluster" means
connected rather than compact.

Single linkage merges on the *closest* pair of points, so it follows a crescent
around its curve. Ward merges to minimise the increase in within-cluster variance
— the same objective K-Means optimises — and a crescent has high variance about
its own centroid no matter how it is cut.

That kinship is directly measurable: **Ward agrees with K-Means at 0.950**, far
more than either agrees with the truth (0.760, 0.750). They are not two
independent methods that happen to fail alike; they encode the same spherical
assumption and fail together for the same reason.

The dendrogram check is the structural one: 200 points down to k=2 requires
exactly 198 merges, and the merge history has 198 entries.

**3 — DBSCAN wins, and "low probability under the GMM" is a trap.**

25 outliers planted uniformly around 200 moon points; the GMM arms flag a fixed
budget of the most suspicious points.

| detector | caught (of 25) | false positives |
|---|---:|---:|
| **DBSCAN** (eps=0.2, min_samples=5) | **21** | **0** |
| GMM by distance to nearest mean | 19 | 3 |
| GMM by lowest max responsibility | **3** | 19 |
| intersection (DBSCAN ∩ distance) | 19 | 0 |

DBSCAN needs no cluster count and its noise label *is* the answer — a point with
fewer than `min_samples` neighbours within `eps` is unreachable from any core
point, which is what "outlier" means on a non-convex shape.

The interesting result is the third row. Reading the exercise's "low probability
in the GMM" as low **max responsibility** inverts the answer: the planted outliers
average **0.9306** max responsibility against the inliers' **0.8885**. They score
*higher*. Responsibility is `P(component | point)` — a posterior over which
component, already normalised over components — so a point far from every
component still gets assigned unambiguously to whichever is nearest, and the
further out it sits, the more unambiguous that assignment becomes. It answers
"which component?", never "does any component explain this?".

Scoring by distance to the nearest component mean, which is what the exercise
meant, recovers 19 of 25 at 3 false positives. Taking the intersection with DBSCAN
keeps 19 and removes the 3, which is the exercise's actual point: two detectors
agreeing is a stronger signal than either alone, because they are wrong about
different points.

## A note on the reference code

`kmeans` and `gmm` print progress while running (`Converged at iteration N`), not
only under `if __name__ == "__main__"`. `harness.parity.quiet()` wraps the calls;
without it one lesson's checks arrive buried in a few hundred lines of transcript.
Not a bug — a lesson module is written to be read and run directly — but worth
recording, since every solution that imports one of these pays for it.
