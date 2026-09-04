<!-- generated:start -->
# 02-ml-fundamentals / 02-linear-regression

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/02-ml-fundamentals/02-linear-regression/) · upstream spec
`phases/02-ml-fundamentals/02-linear-regression/docs/en.md`

```bash
uv run demo practice run 02-linear-regression --ex 1
uv run demo explain 02-linear-regression --ex 1
uv run pytest demos/phases/02-ml-fundamentals/02-linear-regression
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Implement batch gradient descent, stochastic gradient descent (SGD), and mini-batch gradient… | code | T0 | `ex01_gradient_descent_variants.py` |
| 2 | Generate data from a cubic function (y = ax^3 + bx^2 + cx + d + noise). Fit polynomials of de… | code | T0 | `ex02_polynomial_overfitting.py` |
| 3 | Implement Lasso regression (L1 regularization: penalty = alpha * sum(\|w_i\|)). Train on the… | code | T0 | `ex03_lasso_vs_ridge.py` |
<!-- generated:end -->

## Answers

**1 — "fastest" has two answers, and they disagree.**

| variant | epochs to within 5% of best | parameter updates | uphill steps | \|Δ\| to true params |
|---|---:|---:|---:|---:|
| batch | 38 | 60 | **0** | 0.0672 |
| mini-batch (32) | 6 | 420 | 26 | **0.0532** |
| SGD | **1** | 12,000 | 28 | 0.2484 |

Per *epoch* SGD wins by 38×. Per *update* batch wins by 200×, since SGD makes one
update per row. Reporting either alone makes the comparison a coin flip.

Smoothness needs two measures too. Mean |Δcost| per epoch barely separates them —
0.172 / 0.173 / 0.187 — because within-epoch noise has largely averaged out by the
time the epoch-end cost is read. Counting **cost increases** is decisive: batch 0
of 60, mini-batch 26, SGD 28. A full gradient at a small enough step cannot go
uphill; a sampled one can, and that same noise is what lets SGD escape shallow
minima.

What SGD's speed costs shows in the last column: it reaches the neighbourhood of
the optimum in one epoch and then bounces around inside it, ending furthest from
the true parameters. Batch is still descending at epoch 60, so its residual error
is under-training rather than noise — which is why mini-batch, doing neither,
comes out most accurate.

**2 — overfitting is obvious at degree 10, from the gap.**

| degree | train R² | test R² | gap | max \|coeff\| |
|---:|---:|---:|---:|---:|
| 1 | 0.7885 | 0.7105 | +0.078 | 1.9 |
| **3** | 0.9326 | **0.8728** | +0.060 | 2.2 |
| 10 | 0.9666 | **−0.8907** | **+1.857** | 6.3 |

Two things worth correcting about the usual framing.

**A small gap does not indicate underfitting.** Degree 1's gap (+0.078) is *larger*
than degree 3's (+0.060). What identifies underfitting is the low **training**
score — a straight line fails equally on seen and unseen data, so its errors are
honest rather than hidden.

**Coefficient magnitude is not the early warning here.** Degree 10's largest
coefficient is only 2.9× degree 3's, yet its test R² is **−0.89** — worse than
predicting the training mean for every point, which scores 0. On 30 well-spread
points the predictions diverge long before the coefficients look alarming, so a
"watch for large weights" heuristic would have missed this completely.

**3 — L1 zeroes weights because it subtracts; L2 cannot because it multiplies.**

The implementation detail matters: subgradient descent on |w| oscillates across
zero and never lands on it. **Proximal** gradient descent does, because its
soft-threshold step is `|w| − α·lr` floored at 0. So the checks assert *exact*
`0.0`, which a subgradient implementation cannot pass.

| | exact zeros | noise features zeroed | \|weight\| on noise |
|---|---:|---:|---:|
| unregularised | 0 | 0 of 5 | 0.1362 |
| L1 (α=0.35) | **5** | **5 of 5** | **0.0000** |
| L2 (α=0.35) | 0 | 0 of 5 | 0.3296 |

The mechanism, stated arithmetically rather than as the usual diamond-vs-circle
picture: L1's step subtracts a **constant** 0.007 from each weight's magnitude and
clips at zero, so any weight whose gradient step lands inside that window becomes
exactly zero. L2's step is `w × 0.993` — **multiplicative**, and multiplication
cannot reach zero from a non-zero start.

And a result the exercise does not anticipate: **L2's weights on the useless
features come out 2.4× larger than the unregularised fit's** (0.330 vs 0.136).
Uniform shrinkage pulls the signal 29% low (2.86 against a true 4.0), and the
noise features absorb the residual that leaves behind. At this α, L2 is worse at
ignoring irrelevant features than applying no penalty at all — "less sparse than
L1" undersells the problem.
