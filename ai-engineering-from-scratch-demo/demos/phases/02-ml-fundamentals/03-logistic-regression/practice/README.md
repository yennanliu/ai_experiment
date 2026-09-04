<!-- generated:start -->
# 02-ml-fundamentals / 03-logistic-regression

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/02-ml-fundamentals/03-logistic-regression/) · upstream spec
`phases/02-ml-fundamentals/03-logistic-regression/docs/en.md`

```bash
uv run demo practice run 03-logistic-regression --ex 1
uv run demo explain 03-logistic-regression --ex 1
uv run pytest demos/phases/02-ml-fundamentals/03-logistic-regression
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Generate a dataset that is NOT linearly separable (e.g., two concentric circles). Train logis… | code | T0 | `ex01_polynomial_features.py` |
| 2 | Implement a multi-class confusion matrix for the 3-class softmax model. Compute per-class pre… | code | T0 | `ex02_multiclass_confusion.py` |
| 3 | Build an ROC curve from scratch. For 100 threshold values from 0 to 1, compute the true posit… | code | T0 | `ex03_roc_from_scratch.py` |
<!-- generated:end -->

## Answers

**1 — which polynomial term you add is what matters.**

| features | accuracy |
|---|---:|
| x1, x2 | 54.0% |
| + x1·x2 | 59.3% |
| + x1², x2² | **100.0%** |
| all three | **100.0%** |

The majority-class baseline is 50.0%, so the linear model's 54% is within noise of
chance — and 46 points short of what the right features give. The few points above
chance come from sampling noise in the ring, not from learning; no line separates
an annulus from its centre.

The squared terms alone reach 100% because a linear boundary in (x1², x2²) is a
conic in (x1, x2), which is exactly what separates two radii. The cross term alone
reaches 59.3%: x1·x2 encodes *orientation*, and these two classes differ only in
radius. "Add polynomial features" is not a uniform improvement — it works when the
added term matches the structure.

**2 — class 1 is hardest by both measures, for two different reasons.**

```
            predicted
             0   1   2
actual 0 [  99  20   1 ]
       1 [  21  79  20 ]
       2 [   0  25  95 ]
```

| class | precision | recall |
|---:|---:|---:|
| 0 | 0.825 | 0.825 |
| **1** | **0.637** | **0.658** |
| 2 | 0.819 | 0.792 |

Lowest **recall** means class 1 is missed most often — it sits between the other
two, so it loses points in both directions. Lowest **precision** means that when
the model says "1" it is wrong most often — because 1's neighbours are the classes
that get mistaken *for* it. The two measures happen to agree here, and they need
not: a class can be reliably found and over-claimed, or vice versa.

The asymmetry is the part a single accuracy figure destroys. Classes 0 and 2 are
confused **once** in 360 rows, against 41 and 45 errors for the adjacent pairs.
Accuracy of 75.8% is compatible with any error structure; the matrix says the model
has learned the ordering and only struggles at the boundaries.

**3 — the exercise's 100-threshold recipe approximates a quantity with a closed
form.**

AUC by trapezoid over 100 thresholds: **0.812975**. Exact rank-based
(Mann-Whitney) value: **0.813025**. Off by 5.0e-05, and refining to 10,000
thresholds moves it to 0.813050 — closer, confirming the gap is discretisation
rather than a different definition. The exact value is
P(score of a random positive > a random negative), ties counted at ½, which needs
no grid at all.

One thing worth checking before assuming the grid is fine: it reaches (0,0) and
(1,1) **unaided**, because it spans 0 to 1 and every predicted probability lies
strictly between. A grid stopping at 0.05 and 0.95 reaches only (0.01, 0.04) and
(0.94, 1.0), and loses **0.060** of area — the corners carry real area, and a
grid built from `linspace(0.05, 0.95)` would silently under-report.
