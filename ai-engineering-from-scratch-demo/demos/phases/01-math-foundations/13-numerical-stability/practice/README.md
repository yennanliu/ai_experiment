<!-- generated:start -->
# 01-math-foundations / 13-numerical-stability

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/01-math-foundations/13-numerical-stability/) · upstream spec
`phases/01-math-foundations/13-numerical-stability/docs/en.md`

```bash
uv run demo practice run 13-numerical-stability --ex 1
uv run demo explain 13-numerical-stability --ex 1
uv run pytest demos/phases/01-math-foundations/13-numerical-stability
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Catastrophic cancellation. Compute the variance of [1000000.0, 1000001.0, 1000002.0] using th… | code | T0 | `ex01_catastrophic_cancellation.py` |
| 2 | Precision hunt. Find the smallest positive float32 value `x` such that `1.0 + x == 1.0` in Py… | code | T0 | `ex02_machine_epsilon.py` |
| 3 | Log-sum-exp edge cases. Test your `logsumexp_stable` function with: (a) all values equal, (b)… | code | T0 | `ex03_logsumexp_edge_cases.py` |
| 4 | Gradient checking a neural network layer. Implement a single linear layer `y = Wx + b` and it… | code | T0 | `ex04_gradient_check_linear_layer.py` |
| 5 | Loss scaling experiment. Simulate training with float16: create random gradients in the range… | code | T0 | `ex05_loss_scaling.py` |
<!-- generated:end -->

## Findings

**1 — the naive variance formula does not merely lose accuracy in float32, it
returns a negative number.** `E[x²] − E[x]²` on `[1e6, 1e6+1, 1e6+2]` gives
**−65536.0**. The two terms agree to about 13 significant digits and float32
holds 7, so the subtraction returns a value a variance cannot take.

Two things the exercise does not say. **float64 does not rescue it** — the same
formula still errs by 4.07e-05, roughly 11 of 16 significant digits gone; it
fails less visibly, not less. And **Welford in float32 beats naive in float64 by
2048×** (2.0e-08 against 4.07e-05). Doubling the mantissa does not buy back what
the subtraction destroys; changing the algorithm does.

**2 — the exercise asks for two different quantities and calls them one thing.**
`finfo(float32).eps` = 2⁻²³ is the *spacing above 1.0*: the smallest x with
1 + x ≠ 1. The largest x with 1 + x **==** 1 is eps/2 = 2⁻²⁴, which is a tie that
round-to-nearest-even resolves downward. Bisection pins the boundary to adjacent
float32 values, 5.96046448e-08 and 5.96046519e-08.

And "the *smallest* positive x such that 1 + x == 1" has no answer at all: every
positive float below the tie vanishes, so the infimum is the smallest subnormal,
1.4e-45 — a property of the format, not of 1.0.

**3 — `logsumexp_naive` has two distinct failure modes, and only one is an
overflow.** With one value at 1000 it raises `OverflowError`. With all values at
−1000 every `exp` underflows to 0, the sum is 0, and `math.log` raises
`ValueError`. The exercise groups case (c) with the others as though it were the
same problem; it is the opposite end of the range.

**4 — the check that makes a gradient check mean something** is not that the
gradients agree (1.3e-11 here) but that a *wrong* backward pass is rejected. The
solution feeds it two: a uniformly 5% scaled gradient — the classic missing
`1/batch_size` — and a single flipped sign out of nine entries. Both are caught. A
gradient check that has never rejected anything has not been tested.

**5 — "gradients in the range [1e-9, 1e-3]" decides the entire experiment.**
Sampled uniformly, **0.00%** vanish in float16, because almost every uniform draw
lands within a factor of 10 of the upper bound, far above the format's floor. The
solution samples log-uniformly — how gradients are actually distributed — and gets
**24.3%** vanishing, falling to 0.0% under 1024× scaling.

Both boundaries are probed directly rather than inferred from the sample, since at
1024× nothing in the specified range underflows any more:

| | threshold |
|---|---|
| underflow, unscaled | 2⁻²⁵ = 2.98e-08 |
| underflow, at 1024× | 2.91e-11 |
| overflow, largest gradient | scale 6.71e7 (predicted 65504/1e-3 = 6.55e7) |

Loss scaling is a window, not a fix. Below 2⁻²⁵/scale you underflow, above
65504/scale you overflow, and float16 gives about 41 binary decades to fit the
whole gradient distribution into.
