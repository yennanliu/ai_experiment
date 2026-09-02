# Vectors & Matrices — hand-rolled vs numpy

**Lesson:** [01-math-foundations / 02-vectors-matrices-operations](https://yennj12.js.org/ai-engineering-from-scratch/lesson.html?path=phases%2F01-math-foundations%2F02-vectors-matrices-operations)
**Tier:** T0 (instant, no network) · **Install:** `uv sync --extra math`

## What it proves

The lesson writes `Vector` and `Matrix` from lists and loops and says this is
what numpy does underneath. This demo imports *that exact code* and runs the
same eleven operations through numpy, printing the deviation each time.

The largest disagreement across all eleven is `1.8e-15` — floating-point noise,
not a different algorithm.

## Run

```bash
uv run demo run phases/01-math-foundations/02-vectors-matrices-operations
uv run python run.py --explain     # works with nothing installed
```

## Expected output

```
phase 01 / lesson 02: stdlib Matrix vs numpy: 11 check(s)
  ok   matmul (3x4 @ 4x5)      n=15   max|d|=2.22e-16  atol=1e-12
  ...
  worst deviation across all checks: 1.776e-15
```
