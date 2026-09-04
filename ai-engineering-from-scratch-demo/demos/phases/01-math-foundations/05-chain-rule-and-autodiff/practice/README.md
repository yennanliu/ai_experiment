<!-- generated:start -->
# 01-math-foundations / 05-chain-rule-and-autodiff

Solutions to all 4 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/01-math-foundations/05-chain-rule-and-autodiff/) · upstream spec
`phases/01-math-foundations/05-chain-rule-and-autodiff/docs/en.md`

```bash
uv run demo practice run 05-chain-rule-and-autodiff --ex 1
uv run demo explain 05-chain-rule-and-autodiff --ex 1
uv run pytest demos/phases/01-math-foundations/05-chain-rule-and-autodiff
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add `__pow__` to the Value class so you can compute `x  n`. Verify that `d/dx(x^3)` at `x=2`… | code | T0 | `ex01_pow_and_cubic_gradient.py` |
| 2 | Add `tanh` as an activation function. Verify that `tanh'(0) = 1` and `tanh'(2) = 0.0707` (app… | code | T0 | `ex02_tanh_derivative.py` |
| 3 | Build a computation graph for a single neuron: `y = relu(w1*x1 + w2*x2 + b)`. Compute all fiv… | code | T1 | `ex03_neuron_gradients_vs_torch.py` |
| 4 | Implement forward-mode autodiff using dual numbers. Create a `Dual` class and verify it gives… | code | T0 | `ex04_dual_number_forward_mode.py` |
<!-- generated:end -->

## Notes

**Exercises 1 and 2 ask for code the lesson already ships.** `Value.__pow__` and
`Value.tanh` both exist in `code/autodiff.py`. Reimplementing them and testing
against the reimplementation would assert nothing, so both solutions verify the
*shipped* operators against independent oracles — the analytic derivative, and
the lesson's own `gradient_check` finite difference — and then probe cases the
exercise does not mention. This is the same reading as exercise 1 of lesson 01.

**What the probes found.** Both operators are correct wherever they are defined.
Two boundaries are worth recording:

- `x ** y` with a `Value` exponent raises `TypeError`. `__pow__` takes a raw
  number, and there is no `__rpow__`, so Python refuses rather than returning a
  result with no gradient path to `y`. Refusing is the right failure.
- `tanh'` underflows to **exactly 0.0** by x=20 (`tanh'(10)` is already 8.2e-09).
  A gradient through a saturated tanh is not merely small; past that point it is
  gone, and no amount of learning rate recovers it.

**Exercise 3 is the first T1 artifact in this repo.** It needs `torch`
(`uv sync --extra llm`), and without it the exercise skips with that remedy
rather than quietly asserting less. Its most interesting check is at the ReLU
kink: with a pre-activation of exactly 0, ReLU has no derivative, so both engines
return a *convention*. They happen to pick the same one — 0 — but torch reports
`-0.0` for one leaf, which is the same number and a reminder that this is a
choice rather than a derivation.

**Exercise 4's cost asymmetry.** Forward and reverse mode return identical
derivatives (worst gap 1.1e-16 across five expressions). They differ in cost:
forward mode needs one pass per *input*, reverse mode one pass per *output*.
Training has millions of parameters and one scalar loss, which is the whole
reason backprop is reverse mode.
