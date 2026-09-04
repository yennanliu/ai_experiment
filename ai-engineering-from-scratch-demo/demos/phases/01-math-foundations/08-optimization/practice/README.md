<!-- generated:start -->
# 01-math-foundations / 08-optimization

Solutions to all 4 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/01-math-foundations/08-optimization/) · upstream spec
`phases/01-math-foundations/08-optimization/docs/en.md`

```bash
uv run demo practice run 08-optimization --ex 1
uv run demo explain 08-optimization --ex 1
uv run pytest demos/phases/01-math-foundations/08-optimization
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Learning rate sweep. Run vanilla gradient descent on the Rosenbrock function with learning ra… | code | T0 | `ex01_learning_rate_sweep.py` |
| 2 | Momentum comparison. Run SGD with momentum values [0.0, 0.5, 0.9, 0.99] on the Rosenbrock fun… | code | T0 | `ex02_momentum_comparison.py` |
| 3 | Saddle point escape. Define the function `f(x, y) = x^2 - y^2` (a saddle point at the origin)… | code | T0 | `ex03_saddle_escape.py` |
| 4 | Implement learning rate decay. Add an exponential decay schedule to the GradientDescent class… | code | T0 | `ex04_learning_rate_decay.py` |
<!-- generated:end -->

## Answers, and four things the measurements contradicted

**1 — largest learning rate that still converges: 0.005.** But it is not the best
one, and the exercise's phrasing hides that:

| lr | final loss | outcome |
|---:|---:|---|
| 0.0001 | 2.31 | converges, underfits in 5000 steps |
| 0.0005 | 0.0430 | converges |
| **0.001** | **0.00376** | converges — **best** |
| 0.005 | 0.788 | converges — **largest** |
| 0.01 | — | diverges, aborts after 4 steps |

Loss is U-shaped in the learning rate: too small underfits, too large oscillates
in the Rosenbrock valley, and the cliff sits just past 0.005. "Largest that
converges" and "best" are 210× apart. Also worth noting how divergence
manifests: `optimize()` breaks on overflow, so a diverged run is identified by its
*history length* (4 of 5000), not by a large final loss.

**2 — fastest is β=0.9; overshooting is β=0.99.** Two questions, two answers, and
a loss-only reading would get it wrong — β=0.99 reaches the **lowest** final loss
of the four (1.4e-21 vs 8.8e-20) while taking longer to cross the threshold
(1062 steps vs 836), travelling to x=2.04 past a minimum at x=1, and peaking at
loss 26.4, above its own starting loss of 24.2. 2486 of its 5000 steps increase
the loss, against 17 for β=0.9.

**3 — all three escape the saddle, and the ranking flips depending on when you
look.** To leave the flat neighbourhood (|y| > 1): momentum 49 steps, **Adam 76**,
plain GD 233. After 2000 steps: GD and momentum are at |y| ≈ 5e14, Adam at 30.4.

Both halves come from one mechanism. Adam divides by the RMS gradient, so its
step is ~lr *regardless of slope*. Near a saddle the gradient is tiny and that
normalisation is exactly what you want — Adam beats plain GD out by 3×, which is
the property it is recommended for. Once the gradient is large the same cap makes
its escape linear (|y| ≈ steps × lr = 20, measured 30.4) while GD and momentum
accelerate away geometrically.

**4 — decay loses both comparisons.** At lr=0.001 it *hurts*: 0.46 against 0.0038,
because the schedule's total remaining travel is bounded by
Σ lr₀·0.999^k = lr₀/(1−0.999) = 1.0 in parameter distance, nearly all spent in the
first ~1000 steps. After that the run is over however many steps remain.

At lr=0.01 it does not help either, and this is the sharper result: plain GD
diverges after 4 steps, and so does *every* decay tried — 0.999, 0.99, 0.9, even
0.5. Any exponential schedule **starts at lr₀** (0.999⁴ = 0.996), so the first
steps are identical, and that is where this blows up. Decay is a late-stage
refinement tool. It cannot buy back an initially unstable rate.
