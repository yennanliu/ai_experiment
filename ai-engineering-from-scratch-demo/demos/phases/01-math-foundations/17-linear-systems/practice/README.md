<!-- generated:start -->
# 01-math-foundations / 17-linear-systems

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/01-math-foundations/17-linear-systems/) · upstream spec
`phases/01-math-foundations/17-linear-systems/docs/en.md`

```bash
uv run demo practice run 17-linear-systems --ex 1
uv run demo explain 17-linear-systems --ex 1
uv run pytest demos/phases/01-math-foundations/17-linear-systems
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Solve the system `[[1,2,3],[4,5,6],[7,8,10]] x = [6, 15, 27]` using your Gaussian elimination… | code | T0 | `ex01_three_solvers_agree.py` |
| 2 | Generate a 50x5 random matrix X and target y = X @ w_true + noise. Solve for w using normal e… | code | T0 | `ex02_least_squares_four_ways.py` |
| 3 | Create a nearly singular matrix by making two columns almost identical (e.g., column 2 = colu… | code | T0 | `ex03_regularization.py` |
| 4 | Implement the conjugate gradient algorithm for a 100x100 random symmetric positive definite m… | code | T0 | `ex04_conjugate_gradient.py` |
| 5 | Time your Cholesky solver vs your LU solver vs `np.linalg.solve` on symmetric positive defini… | code | T1 | `ex05_cholesky_timing.py` |
<!-- generated:end -->

## Answers, and four premises that did not survive measurement

**1 — all three solvers agree to 3.1e-15,** and they agree with a **fourth**
solution computed by exact rational elimination over `Fraction`: x = [3, −3, 3],
with no rounding anywhere. That matters because three solvers agreeing is
necessary but weak — they could agree on a wrong answer. κ(A) = 88 is reported
too, since agreement on an ill-conditioned system would prove much less.

(det = −3. Worth noting `[[1,2,3],[4,5,6],[7,8,9]]` is singular; the 10 in the
corner is what makes this solvable at all.)

**2 — on a random 50×5 matrix the exercise cannot distinguish the four methods.**
All agree to 2.7e-15 with identical residuals, because κ(XᵀX) ≈ 2. The comparison
only becomes informative on an ill-conditioned design:

| method | gap vs `lstsq` at κ(XᵀX) = 5.8e14 | residual |
|---|---:|---:|
| normal equations | **3.06e+03** | 0.7180 |
| QR | 3.3e-05 | 0.7180 |
| SVD | 2.9e-11 | 0.7180 |

κ(XᵀX) = κ(X)², confirmed on both designs — that squaring is the whole reason to
avoid forming the Gram matrix. QR and SVD never do, so they work at κ(X).

And the residuals are **identical to four decimals** across all four. A
near-duplicate column means many coefficient vectors fit almost equally well, so a
small residual is no evidence the coefficients are right. Trust QR or SVD, and
check κ before trusting anything.

**3 — the residual gets *worse* with regularization, and that is not the point.**
Plain 2.6e-06, ridge 0.279 — it must be worse, since ridge solves a different
problem. Comparing solutions and residuals, as the exercise says, suggests
regularization hurts.

What improves is **stability**. Perturbing b by 1e-8 moves the plain solution by
**409** and the ridge solution by **1.5e-08** — a factor of 2.7e10. The plain shift
is bounded by κ(A)·δ = 2.1e03, exactly as theory says. Ridge replaces κ(A) with
κ(A + λI) ≈ σ₁/λ, putting a floor under the smallest singular value; that is why
λ = 0.01 buys so much. The unregularized solution norm is 1.4e10 against ridge's
0.98: the two near-identical columns take large opposing coefficients that nearly
cancel.

**4 — "the theoretical maximum of n iterations" is an exact-arithmetic result.**
Three 100×100 SPD systems, identical but for conditioning:

| | √κ | iterations | bound ½√κ·ln(2/tol) |
|---|---:|---:|---:|
| κ≈10 | 3 | 31 | 30 |
| κ≈1e3 | 32 | **178** | 302 |
| κ≈1e5 | 316 | **728** | 3022 |

Rounding destroys the conjugacy that guarantees termination in n steps, so κ≈1e5
needs 7.3× n. And the lesson's `conjugate_gradient` defaults `max_iter` to n,
which means on two of these three it **returns an unconverged answer** — residuals
1.4e-03 and 7.25 against a tolerance of 1e-08 — and raises nothing. The count
tracks √κ, which is why preconditioning rather than a bigger iteration budget is
the lever.

(The κ≈10 case sits marginally above its bound because the bound is on the A-norm
of the error while the tolerance here is on the residual.)

**5 — Cholesky is not 2× faster, and the reason is more useful than the claim.**
Measured 1.39× at n=500. The 2× is a *flop* ratio (n³/3 vs 2n³/3) and neither
implementation is flop-bound: both are O(n²) Python-level iterations each
delegating one O(n) vector operation to numpy, so what costs is the ~125,000 call
boundaries at n=500, not the arithmetic. The fitted growth exponent confirms it —
**n^1.99 and n^2.06**, not n³. Both algorithms run the same n²/2 iterations, so the
flop saving is invisible; Cholesky wins only on its shorter average inner dot.

`np.linalg.solve` is **71×** faster than either. Choosing the better algorithm in
Python loses to choosing blocked, vectorised LAPACK.

The n=10 timing is excluded from the ordering check on purpose: the decomposition
takes microseconds there, so the ratio is noise and falls either side of 1.0
between runs. Asserting it would be flaky by construction.
