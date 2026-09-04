<!-- generated:start -->
# 01-math-foundations / 04-calculus-for-ml

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/01-math-foundations/04-calculus-for-ml/) · upstream spec
`phases/01-math-foundations/04-calculus-for-ml/docs/en.md`

```bash
uv run demo practice run 04-calculus-for-ml --ex 1
uv run demo explain 04-calculus-for-ml --ex 1
uv run pytest demos/phases/01-math-foundations/04-calculus-for-ml
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Implement `numerical_second_derivative(f, x)` using `numerical_derivative` called twice. Veri… | code | T0 | `ex01_second_derivative.py` |
| 2 | Use gradient descent to find the minimum of f(x, y) = (x - 3)^2 + (y + 1)^2. Start from (0, 0… | code | T0 | `ex02_gradient_descent_2d.py` |
| 3 | Add momentum to the gradient descent loop: maintain a velocity vector that accumulates past g… | code | T0 | `ex03_momentum_comparison.py` |
<!-- generated:end -->

## Findings

**1 — the lesson's default step is wrong for nested derivatives.** Composing
`numerical_derivative` with itself, as exercise 1 requires, gives **11.923796**
at the default `h=1e-7`, not 12. A central difference carries O(h²) truncation
error and O(ε/h) cancellation error; nest one inside another and cancellation
becomes O(ε/h²), so the optimal step moves from ~1e-5 to ~1e-2. Measured error by
step: `1e-2 → 8.0e-12`, `1e-4 → 1.1e-07`, `1e-7 → 7.6e-02`, `1e-9 → 1.2e+01`.
At `h=1e-9` the answer is 100% wrong. The default is a good default for first
derivatives, and the exercise inherits it without warning.

**3 — momentum is slower on this function, and the exercise does not say
otherwise.** It asks to *compare*, and the comparison comes out against momentum:

| Variant | Steps to 1e-6 | Direction reversals | Lands at |
|---|---:|---:|---|
| plain, lr=0.01 | 101 | 0 | +1.224746 |
| momentum, β=0.9 | 234 | 25 | **−1.224746** |
| plain, lr=0.1 (matched) | **10** | 7 | +1.224745 |

Two things worth noticing. Momentum starting at x₀=+2.5 ends in the *negative*
well — accumulated velocity carried it across the local maximum at x=0, so it did
not just converge slower, it converged somewhere else. And the benefit momentum
is supposed to deliver on this problem is its effective learning rate
`lr/(1−β) = 0.1`; using that rate directly converges in 10 steps, because the
stability limit `2/f''(√1.5) = 0.167` permits it. Momentum earns its keep on
ill-conditioned problems where no single scalar rate works for every direction —
a 1-D quartic is not that problem.
