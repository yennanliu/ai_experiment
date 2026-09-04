<!-- generated:start -->
# 01-math-foundations / 18-convex-optimization

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/01-math-foundations/18-convex-optimization/) · upstream spec
`phases/01-math-foundations/18-convex-optimization/docs/en.md`

```bash
uv run demo practice run 18-convex-optimization --ex 1
uv run demo explain 18-convex-optimization --ex 1
uv run pytest demos/phases/01-math-foundations/18-convex-optimization
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Convexity gallery. Test these functions for convexity using the checker: f(x) = x^4, f(x) = s… | code | T0 | `ex01_convexity_gallery.py` |
| 2 | Newton vs gradient descent race. Run both methods on f(x,y) = 50*x^2 + y^2 from the starting… | code | T0 | `ex02_newton_vs_gd.py` |
| 3 | Lagrange multiplier geometry. Minimize f(x,y) = (x-3)^2 + (y-3)^2 subject to x + 2y = 4. Veri… | code | T0 | `ex03_lagrange_geometry.py` |
| 4 | Regularization constraint. Implement L1-constrained optimization: minimize (x-3)^2 + (y-2)^2… | code | T0 | `ex04_l1_constraint_sparsity.py` |
| 5 | Hessian eigenvalue analysis. Compute the Hessian of the Rosenbrock function at (1,1) and at (… | code | T0 | `ex05_hessian_eigenvalues.py` |
<!-- generated:end -->

## Answers

**1 — why each verdict makes sense.** x⁴ convex (f″ = 12x² ≥ 0); sin not (curves
both ways); x²+y² convex (Hessian 2I); x·y not; max(x,0) convex.

Two of these deserve more than the sampler's word. x·y is rejected because its
Hessian `[[0,1],[1,0]]` has eigenvalues **±1** — indefinite, a saddle. And sin is
rejected by an *exhibited* chord: on [0, 3π/2] the midpoint value 0.707 exceeds the
chord average −0.5 by 1.207. That matters because the checker is one-sided — it
can fail to find a violation but never prove absence, so a specific
counterexample is stronger evidence than 4,000 samples finding one.

`max(x, 0)` is the case only the chord test can settle: convex, but not
differentiable at 0, so there is no Hessian to examine.

**2 — GD needs 684 steps; Newton needs 1.** Newton's answer is not a race result
but a consequence: on a quadratic the Hessian is exact, so −H⁻¹∇f *is* the vector
to the minimum. The first iterate is `[0.0, 0.0]`, and it stays 1 step at every
condition number — conditioning is precisely what a Hessian corrects for.

What happens to GD as κ grows:

| κ | GD steps |
|---:|---:|
| 1 | 1 |
| 10 | 132 |
| 50 | 684 |
| 500 | 6901 |

A 50× rise in κ costs 52× the steps — linear, within 5% of what the (κ−1)/(κ+1)
per-step contraction predicts. (κ=1 is degenerate: a perfectly round bowl, one
step.) Learning rate is pinned at 1/λ_max throughout, or the sweep would measure
divergence instead.

**3 — the solution is (2, 1), λ = −2.** The perpendicular foot of (3,3) on
x+2y=4, matched to 4.4e-16, with ∇f ∥ ∇g (cross product 1.8e-15).

But the parallel-gradient condition is **necessary, not sufficient** — it holds at
every stationary point of the Lagrangian, maxima included. So the solution also
samples 81 points along the constraint direction (2,−1), all satisfying x+2y=4,
and confirms t=0 is where f is smallest. Without that, "gradients are parallel"
identifies a stationary point and nothing more.

**4 — the zero coordinate comes from the diamond's corners, not from constraining
a norm.** The L1 solution is the vertex (1, 0): budget active, y exactly 0. Run
the identical problem on an **L2 ball** of the same radius and the answer is
(0.832, 0.555) — no zero coordinate, because a sphere has no corners. That
comparison is the actual content of "sparsity from the diamond constraint".

Sparsity is also not free: f = 8.00 under L1 against 6.79 under L2 at the same
budget. The diamond is contained in the ball, so it can only ever do worse.

**5 — the two requested points have identical eigenvalue spectra.** At both (1,1)
and (−1,1) the Hessian is `[[802, ∓400], [∓400, 200]]`; flipping the off-diagonal
sign leaves trace and determinant unchanged, so both give **{1001.60, 0.3994}**.
The eigenvalues cannot distinguish the global minimum from a point four units
above it. Only the gradient does — zero at (1,1), `[−4, 0]` at (−1,1).

What the eigenvalues *do* say is that κ = **2508**: curvature along one direction
is 2508× the other. That single number explains why Rosenbrock is the standard
optimiser benchmark and why exercise 2's GD scaling matters.

And Rosenbrock's Hessian *is* indefinite elsewhere — just not at either point the
exercise names. At (0, 1) the eigenvalues are **200 and −398** (det = −79600), so
−H⁻¹∇f there points toward higher function values. Newton needs a PSD Hessian and
does not get one.
