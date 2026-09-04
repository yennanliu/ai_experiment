<!-- generated:start -->
# 01-math-foundations / 14-norms-and-distances

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/01-math-foundations/14-norms-and-distances/) · upstream spec
`phases/01-math-foundations/14-norms-and-distances/docs/en.md`

```bash
uv run demo practice run 14-norms-and-distances --ex 1
uv run demo explain 14-norms-and-distances --ex 1
uv run pytest demos/phases/01-math-foundations/14-norms-and-distances
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Compute L1, L2, and L-infinity distances between (1, 2, 3) and (4, 0, 6). Verify that L-inf <… | code | T0 | `ex01_norm_ordering.py` |
| 2 | Create two vectors where cosine similarity is high (> 0.9) but L2 distance is large (> 10). E… | code | T0 | `ex02_cosine_vs_l2.py` |
| 3 | Implement a function that takes a dataset and a query point and returns the nearest neighbor… | code | T0 | `ex03_four_metrics_disagree.py` |
| 4 | Compute the Wasserstein distance between [0.5, 0.5, 0, 0] and [0, 0, 0.5, 0.5] by hand using… | code | T0 | `ex04_wasserstein_by_hand.py` |
| 5 | Implement MinHash for approximate Jaccard similarity. Generate 100 random sets, compute exact… | code | T0 | `ex05_minhash.py` |
<!-- generated:end -->

## Answers, and where a single example is not enough

**1 — L1 = 8, L2 = √22 ≈ 4.690, L∞ = 3.** |Δ| = [3, 2, 3]. The exercise asks to
verify the ordering holds "for any pair", which one pair cannot show, so the check
runs 2000 random pairs across dimensions 1–12 (zero violations).

The proof is turned into two testable identities rather than prose. **Upper:**
L1/L2 ≤ √d by Cauchy–Schwarz, and the bound is *attained* only when every |Δᵢ| is
equal — nine equal components give L1/L2 = 3.000000 = √9 exactly. **Lower:** all
three coincide iff Δ has a single non-zero component, since with one term the sum,
the root-of-sum-of-squares, and the max are the same number. That is what makes
the inequalities tight rather than merely true.

**2 — the geometric answer, in one sentence each.** High cosine with large L2:
cosine normalises magnitude away, so scaling one vector moves L2 from 0 to 171
while cosine stays at 1.0 to twelve decimal places. It cannot see magnitude at
all, which is why the first case needs no cleverness — just a scale factor.

Low cosine with small L2 is the reverse, and it is about **length, not angle**:
L2 ≤ |a| + |b|, so two vectors of length ~0.2 cannot be more than 0.402 apart
however different their directions. The pair used is 78.6° apart at L2 = 0.255.

**3 — total disagreement found at trial 22.** Six points in 3-D: L1 picks point 1,
L2 point 0, cosine point 2, Mahalanobis point 4. Found by seeded search rather
than hand-tuned, which is the point — a hand-built example might have been
reverse-engineered from the metrics, whereas the trial count reports how uncommon
the case actually is (uncommon, not exotic).

Each disagreement has its own cause. L1 vs L2 requires the axes to trade off:
L1 sums coordinates while L2 punishes the largest, so a point slightly off on
every axis beats one badly off on a single axis under L1 and loses under L2.
Cosine can pick a point far away in absolute terms if the direction from the
origin matches. Mahalanobis rescales by the data's own covariance, making "near"
depend on the whole dataset rather than the two points.

**4 — pair 1 is larger, by exactly 2×.** By hand via the CDF method:

| | F_p | F_q | \|ΔF\| | W₁ |
|---|---|---|---|---:|
| pair 1 | [.5, 1, 1, 1] | [0, 0, .5, 1] | [.5, 1, .5, 0] | **2.0** |
| pair 2 | [.25, .5, .75, 1] | [0, 0, .5, 1] | [.25, .5, .25, 0] | **1.0** |

The *why* is transport work. Pair 1 moves all of its mass two bins right. Pair 2
already has half its mass at or past the target, so it moves half one bin and half
two — half the work for the same endpoints.

Worth contrasting with KL on the same pair: **inf**, because q is zero where p has
mass. W₁ measures how far mass must travel, so disjoint support gives a large
number rather than an undefined one.

**5 — the error claim worth testing is the rate, not the direction.** MAE falls
0.0390 → 0.0258 → 0.0196 at k = 50/100/200, but "smaller is better" would pass
even for a biased implementation. A MinHash estimate is the mean of k Bernoulli
trials with p = Jaccard, so its standard error is √(J(1−J)/k) and quadrupling k
should halve the error. Measured ratio: **1.989** against a predicted 2. Signed
bias stays under 0.006 at every budget, and absolute MAE tracks the predicted σ at
about 0.8× — which is what MAE/σ should be for a normal.
