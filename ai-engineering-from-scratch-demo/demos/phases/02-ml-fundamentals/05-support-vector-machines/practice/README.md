<!-- generated:start -->
# 02-ml-fundamentals / 05-support-vector-machines

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/02-ml-fundamentals/05-support-vector-machines/) · upstream spec
`phases/02-ml-fundamentals/05-support-vector-machines/docs/en.md`

```bash
uv run demo practice run 05-support-vector-machines --ex 1
uv run demo explain 05-support-vector-machines --ex 1
uv run pytest demos/phases/02-ml-fundamentals/05-support-vector-machines
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Generate a 2D linearly separable dataset. Train your LinearSVM and identify the support vecto… | code | T0 | `ex01_support_vectors.py` |
| 2 | Vary C from 0.001 to 1000 on a noisy dataset. Plot the decision boundary for each C value. Ob… | code | T0 | `ex02_c_parameter_sweep.py` |
| 3 | Create a dataset where class boundaries are circular (not linear). Show that a linear SVM fai… | code | T0 | `ex03_rbf_kernel_separability.py` |
| 4 | Compare hinge loss vs logistic loss on the same dataset. Train a linear SVM and logistic regr… | code | T0 | `ex04_hinge_vs_logistic.py` |
| 5 | Implement SVR (epsilon-insensitive loss). Fit it to y = sin(x) + noise. Plot the epsilon tube… | code | T0 | `ex05_svr_epsilon_tube.py` |
<!-- generated:end -->

## Answers

**1 — "the points closest to the boundary" is exactly right, and necessarily so.**

5 of 106 points have y·f(x) < 1, and every one of them is nearer the boundary than
every non-margin point (furthest margin point 0.4735, nearest other 0.4824 — the
groups do not interleave). The margin set and the nearest-k set coincide
*exactly*, and this is an identity rather than luck: distance is |f(x)|/‖w‖ with
‖w‖ constant, so ranking by distance and by |f(x)| are the same ranking.

Where they part company is non-separable data. With 12 points misclassified, only
33 of 35 margin points are among the nearest — a misclassified point has
y·f(x) < 0 while its *distance* |f(x)| can be large, so "closest to the boundary"
stops describing the support set.

**2 — C = 0.001 does not underfit. It diverges.**

λ = 1/C = 1000 with the default lr = 0.01 makes the weight-decay factor
(1 − lr·λ) = **−9**, so w flips sign and grows every step until ‖w‖ is nan.
Stability needs lr·λ < 1, i.e. **C > lr**. The exercise's suggested range starts
outside it.

Over the stable range the sweep behaves as described:

| C | margin 1/‖w‖ | support vectors | train | test |
|---:|---:|---:|---:|---:|
| 0.1 | 7.002 | 192/200 | 51.5% | 50.0% |
| 1 | 1.976 | — | 92.0% | **88.0%** |
| 100 | 0.533 | — | 93.0% | 88.0% |
| 1000 | 0.458 | 30/200 | 93.5% | 88.0% |

The margin narrows by 15× and the support count falls from 192 to 30. But the
**overfitting half of the exercise does not happen**: the gap at C=1000 is +5.5%
and test accuracy never drops. A linear model in 2D has too little capacity to
overfit however hard C pushes — that half needs a kernel.

**3 — separability in the lifted space, and why the obvious statistic misleads.**

The raw linear SVM gets 55.0% test against a 65.0% majority baseline. A linear SVM
trained on the kernel *rows* — a linear model in exactly the RBF feature space —
gets **92.5%** at γ=5. Same model, same optimiser; only the representation changed.

Held-out accuracy is U-shaped in γ: **35.0% / 92.5% / 80.0%** at γ = 0.01 / 5 / 50.
So "use RBF" is not the recipe; γ is.

The tempting statistic — within-class kernel similarity over between-class — is
*not* the criterion. At γ=50 that ratio looks excellent (36×) precisely because
both terms are near zero: mean within-class similarity is 0.0091, the kernel has
become the identity matrix, every point resembles only itself, and the model
overfits with an 18.8% train-test gap. A ratio of two vanishing numbers says
nothing.

**4 — 35 points against 200, and the difference is exact.**

Both models reach 94.0% on the same data, so the comparison is about *which*
points matter. The hinge boundary depends on **35 of 200**: the update branches on
y·f(x) ≥ 1, and the other 165 contribute only weight decay, with no data term at
all. The logistic boundary depends on **all 200** — exactly 0 points have p − y = 0,
and the smallest |p − y| is 3.5e-07.

"Faintly" is not "not at all", and the distinction is exact rather than a matter
of degree: hinge is max(0, 1 − y·f(x)), identically zero past the margin, so its
derivative there *is* zero. Logistic is log(1 + exp(−y·f(x))), positive everywhere,
so its derivative never is. Support sparsity is a property of the loss having a
flat region — not of the optimiser, and not of the data.

**5 — the ε-tube, and what happens when it swallows the data.**

| ε | outside the tube | ‖w‖ | mean \|pred − sin(x)\| |
|---:|---:|---:|---:|
| 0.05 | 71/120 | — | 0.0565 |
| **0.15** | 26/120 | 1.559 | **0.0424** |
| 0.4 | **0** | — | 0.1403 |
| 1.0 | **0** | 0.124 | 0.5109 |

Points outside the tube *are* the support vectors, because the ε-insensitive loss
is flat inside it. Past ε ≈ 0.4 there are **none**: every point sits inside, the
data term vanishes for all of them, only weight decay remains, and the weights
collapse to ‖w‖ = 0.124. The model does not fit loosely — it stops learning.

Error against the true sine is U-shaped with the optimum at ε = 0.15, near the
noise scale σ = 0.1, which is what ε is for: ignore residuals you cannot explain.
At that ε the learned weights come out **(1.001, −0.942, 0.736)** against sin's
Taylor coefficients (1, −1, 1). The basis [x, x³/6, x⁵/120] was chosen to make
that checkable — a linear SVR on x alone cannot fit a sine at all, which the
exercise does not mention.
