<!-- generated:start -->
# 03-deep-learning-core / 01-the-perceptron

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/03-deep-learning-core/01-the-perceptron/) · upstream spec
`phases/03-deep-learning-core/01-the-perceptron/docs/en.md`

```bash
uv run demo practice run 01-the-perceptron --ex 1
uv run demo explain 01-the-perceptron --ex 1
uv run pytest demos/phases/03-deep-learning-core/01-the-perceptron
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Train a perceptron on a NAND gate (the universal gate - any logic circuit can be built from N… | code | T0 | `ex01_nand_gate.py` |
| 2 | Modify the Perceptron class to track the decision boundary (w1*x1 + w2*x2 + b = 0) at each ep… | code | T0 | `ex02_boundary_trajectory.py` |
| 3 | Build a 3-input perceptron that outputs 1 only when at least 2 of the 3 inputs are 1 (a major… | code | T0 | `ex03_majority_vote.py` |
<!-- generated:end -->

## Answers

All three exercises land on the same defect, which is the reason to read them
together: **the perceptron learning rule in this lesson never produces a
separating hyperplane.** It produces a hyperplane the data is sitting on, and
`predict`'s `>=` tie-break does the rest.

**1 — NAND trains in 6 epochs, and its decision boundary has margin exactly zero.**

| input | target | w·x + b | side |
|---|---:|---:|---|
| (0, 0) | 1 | +0.2000 | above |
| (0, 1) | 1 | +0.1000 | above |
| (1, 0) | 1 | **+0.0000** | **on the line** |
| (1, 1) | 0 | −0.1000 | below |

Final `w = [−0.20, −0.10]`, `b = +0.20`; geometric margin **0.00e+00**.

"Verify its weights and bias form a valid decision boundary" is the whole
exercise, and taken strictly the answer is *no*. The point (1, 0) is not on the
positive side of the line — it is on the line. Change `predict`'s comparison from
`>= 0` to `> 0` and this trained NAND gets **1 of 4 rows wrong**.

**MECHANISM.** The update fires only on an error, and `z = 0` already predicts 1.
So the rule halts the instant the last misclassified point *reaches* the boundary
and has no term that would push it across. Nothing in the perceptron rule asks
for margin, and this is what that costs.

The comparison that makes the point is inside the lesson itself. `xor_network`
hard-codes a NAND unit at `w = [−1, −1]`, `b = +1.5` — margin **0.3536**, and it
survives the strict comparison. **The hand-written weights are better than the
learned ones**, on the same four rows.

**ANSWER: the trained unit really is universal.** Wiring copies of it:

| gate | wiring | rows correct |
|---|---|---:|
| NOT | `nand(a, a)` | 2/2 |
| AND | `nand(nand(a,b), nand(a,b))` | 4/4 |
| OR | `nand(nand(a,a), nand(b,b))` | 4/4 |
| XOR | five NANDs | **4/4** |

XOR is the gate a single perceptron provably cannot learn. Five copies of one it
*can* learn produce it — which is the lesson's multi-layer argument, built out of
a unit that was trained rather than assigned.

**2 — the AND boundary stops moving two epochs early, and floating point is why.**

| epoch | w | b | line | updated by |
|---|---|---:|---|---|
| 0 | [0.00, 0.00] | 0.00 | **no line at all** | — |
| 1 | [0.10, 0.10] | 0.00 | x2 = −1.000·x1 − 0.000 | (0,0), (1,1) |
| 2 | [0.20, 0.10] | −0.10 | x2 = −2.000·x1 + 1.000 | (0,0), (0,1), (1,1) |
| 3 | [0.20, 0.10] | −0.20 | x2 = −2.000·x1 + 2.000 | (0,1), (1,0), (1,1) |
| 4 | [0.20, 0.10] | −0.20 | x2 = −2.000·x1 + 2.000 | none — converged |

**There is no boundary to track at epoch 0.** The lesson initialises weights and
bias to zero, so the tracked equation is `0·x1 + 0·x2 + 0 = 0` — satisfied by
every point in the plane. A line exists only from epoch 1.

**MECHANISM: the (0,0) row can translate the line but never rotate it.** The
update is `w_i += lr·error·x_i`, and (0,0) has every `x_i = 0`, so its correction
lands entirely on the bias. (0,0) is corrected in epochs 1 and 2, and neither of
those corrections appears in the weights — every rotation in the table above came
from one of the other three rows.

**FINDING: the run halts on a line that is not an AND boundary.** It stops at
epoch 4 with `b = -0.20000000000000004`, where (1, 0) scores **−2.78e−17**. That
is negative only because `b` accumulated as `−0.1 − 0.1` in binary floating point
and landed a few ulps below `−0.2`. Re-run the identical loop in exact rationals:

| arithmetic | epochs | final b | (1,0) score |
|---|---:|---:|---:|
| float | 4 | −0.20000000000000004 | −2.78e−17 |
| `Fraction` | **6** | **−3/10** | −1/10 |

In exact arithmetic (1, 0) scores exactly 0, `predict` returns 1 for a 0-labelled
row, and training carries on for two more epochs to a genuinely different answer.
**The reported convergence is a float artefact.**

**ANSWER: the line stops at the first one that fits, not the best one.** Final
slope −2.0 at geometric margin 1.24e−16, against slope −1.0 and margin **0.3536**
for the max-margin line `x1 + x2 = 1.5`. The rule has no reason to keep moving
once every point is on the correct *closed* side, so the line it leaves behind is
touching the data — and pointing the wrong way.

**3 — majority is separable, and the reason is not that training worked.**

Trained in 4 epochs to `w = [+0.1, +0.1, +0.1]`, `b = −0.2`, all 8 rows correct.

**WHY: the specification of the function is already a hyperplane.** "At least 2
of the 3 inputs are 1" is `sum(x) >= 2`, which is `w·x + b >= 0` at
`w = [1,1,1]`, `b = −1.5`. There is nothing to search for. Every symmetric
threshold function is separable for the same reason; parity is not one, because
its output is not monotone in the vote count.

**CONTROL.** 3-input parity — the lesson's XOR, one dimension up — runs 1000
epochs without converging, and a linear program confirms no hyperplane exists.

**FINDING: the same zero-margin failure, three times over.**

| input | votes | target | w·x + b |
|---|---:|---:|---:|
| (0,1,1) | 2 | 1 | **+0.0** |
| (1,0,1) | 2 | 1 | **+0.0** |
| (1,1,0) | 2 | 1 | **+0.0** |
| (1,1,1) | 3 | 1 | +0.1 |

Three of the four positive rows score exactly zero and are classified only by the
`>=`. The hand-written `w = [1,1,1]`, `b = −1.5` separates the same eight rows at
margin **0.2887** and needs no tie-break.

**…and separability is the exception, not the rule.** A linear program decides
all 2⁸ = 256 Boolean functions of three inputs: **104 are linearly separable —
40.6%**. At n = 2 it is 14 of 16, so "which gates can a perceptron learn?" reads
like a question about XOR alone. At n = 3 the majority of functions are already
out of reach, and the fraction keeps collapsing as n grows.

## A note on the reference code

Two things about `Perceptron.predict`:

- The `>= 0` tie-break is load-bearing. It is what makes NAND and majority
  "converge" at all, since in both cases training terminates with points exactly
  on the boundary. This is a defensible convention, but the lesson does not say
  it is doing any work, and the exercises ask the reader to *verify* boundaries
  that only hold because of it.
- Because `train` accumulates `bias` by repeated `+= lr * error`, the sign of a
  point's score at the boundary is decided by accumulated rounding. On the AND
  gate that changes the epoch count and the final weights (exercise 2). Comparing
  against a small tolerance, or accumulating an integer mistake count and scaling
  once, would remove the dependence.

The lesson's own `xor_network` hard-codes weights with real margins
(`[-1,-1], +1.5` and `[1,1], -1.5`), so the gap between what it trains and what
it writes down by hand is visible inside a single file.
