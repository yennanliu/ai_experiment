<!-- generated:start -->
# 03-deep-learning-core / 03-backpropagation

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/03-deep-learning-core/03-backpropagation/) · upstream spec
`phases/03-deep-learning-core/03-backpropagation/docs/en.md`

```bash
uv run demo practice run 03-backpropagation --ex 1
uv run demo explain 03-backpropagation --ex 1
uv run pytest demos/phases/03-deep-learning-core/03-backpropagation
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a `__sub__` method to the Value class (a - b = a + (-1 * b)). Then implement a `__neg__`… | code | T0 | `ex01_sub_and_neg.py` |
| 2 | Add a `relu` method to Value (output max(0, x), derivative is 1 if x > 0, else 0). Replace si… | code | T0 | `ex02_relu_xor.py` |
| 3 | Implement a `__pow__` method on Value for integer powers. Use it to replace `mse_loss` with a… | code | T0 | `ex03_pow_and_mse.py` |
| 4 | Add gradient clipping to the training loop: after calling `backward()`, clip all gradients to… | code | T0 | `ex04_clip_to_unit_box.py` |
| 5 | Build a visualization: after training on XOR, print the gradient of every parameter in the ne… | code | T0 | `ex05_smallest_gradients.py` |
<!-- generated:end -->

## Answers

All five exercises edit the same 130-line autograd engine, and between them they
find **four defects in it** — three in `Value` and one in what the exercises
themselves ask you to conclude:

| where | defect |
|---|---|
| `Value.__sub__` / `__neg__` | already implemented; ex 1 asks you to write code you were handed |
| `Value.backward` | `_children` is a `set`, so training is **not reproducible run to run** |
| `Value.backward` | calling it twice compounds rather than doubles — the interior is never cleared |
| `Value.__pow__` (ex 3) | breaks three ways outside integer powers, one of them silently |

And the pedagogy: exercise 4's gradient clip **cannot fire on this network**, and
exercise 5 asks you to look for vanishing gradients at the one moment they are
hardest to see.

**1 — `__sub__` and `__neg__` are already in `code/main.py`.**

The exercise asks for `__sub__` implemented as `self + (-other)` and `__neg__` as
`self * -1`. Both are there, spelled exactly that way. Neither appears in
`docs/en.md`, so the reader is asked to re-derive code they were handed.

The gradients are exact. At `a = 3.0`, `b = 1.25`, for `(a−b)²`:

| | engine | hand derivation | deviation |
|---|---:|---:|---:|
| `d/da` | +3.5000 | `2(a−b)` = +3.5000 | **0.0** |
| `d/db` | −3.5000 | `−2(a−b)` = −3.5000 | **0.0** |

A central finite difference at `h = 1e-05` agrees to 2.293e-11 — the quotient's own
`O(h²)` truncation, not the engine's error.

**MECHANISM: the identity costs five nodes and negates on one side only.** `a - b`
builds 5 nodes joined by `['*', '+']`, because `a + (−1 · b)` materialises a
`Value(-1)` leaf and a multiply a primitive subtraction would not need. And
`1.0 - Value(2.0)` raises `TypeError` — there is no `__rsub__`, so only the left
operand may be a `Value`.

**MECHANISM: `diff * diff` stores a child set of length 1 and is still right.**
`set((diff, diff))` collapses to one element, yet the gradient is correct:
`_backward` captured both operands and runs `self.grad +=` then `other.grad +=`
against the same object, accumulating `2·diff`. **The set decides traversal, not
credit** — which matters for exercise 4.

**FINDING: a second `backward()` compounds rather than doubles.** Three calls on
one graph give `d/da` = 3.5, 10.5, 21.0 — the triangular numbers, not 3.5, 7.0,
10.5. `backward` *sets* the root to 1.0 but never clears the interior, so pass *n*
re-propagates already-accumulated interior gradients on top of a fresh 3.5.
`zero_grad()` resets parameters only. Training survives this because each epoch
builds its graph from scratch.

**2 — ReLU is faster on the 2 seeds where it trains at all.**

| | converges (loss < 0.04) | epochs where both do |
|---|---:|---|
| sigmoid | **8 / 8** | 245, 295 |
| relu | 2 / 8 | **29 (8.4×), 74 (4.0×)** |

"You should see faster training" is true *conditional on training happening*. The
other six relu seeds park for good at constant-output losses `[0.5001, 0.6667,
0.6668, 1.0]`.

**MECHANISM: `relu'(x ≤ 0) = 0` kills units, and `bias = Value(0.0)` kills the
`(0,0)` row at birth.** Units dead at init: `[1, 1, 1, 0, 3, 1, 1, 1]` of 4; after
600 epochs: `[1, 2, 2, 2, 3, 1, 2, 3]`. Five of eight seeds *lose* units mid-training
and none recovers. And because every bias starts at zero, the XOR input `(0, 0)`
sits exactly on the kink at init on every seed — gradient into layer 0 from that
row peaks at **0.0000** for relu against 0.0521 at worst for sigmoid.

**CONTROL: halve the learning rate and sigmoid stops converging too** — 4/8 instead
of 8/8 at `lr = 0.5`. ReLU's derivative of 1 against sigmoid's 0.25 ceiling
rescales the effective step as much as it reshapes the landscape, so part of the
measured speed-up is a learning rate the comparison never controls for.

**3 — `(predicted - target) ** 2` reproduces `mse_loss` to one ulp, and the ulp is the engine's.**

Over 136 parameter gradients (17 params × 8 seeds) the worst disagreement is
**5.551e-17**, and the loss values agree exactly. On a **one-row** graph they agree
*bitwise* on all 8 seeds: `diff * diff` credits `diff` twice by `+=`, `diff ** 2`
once by `2·diff` — the same float.

**FINDING: the residue is not the rewrite's.** Rebuilding the *same* `mse_loss`
graph twice moves the gradients by up to **5.551e-17** as well.

**MECHANISM.** `_children` is a `set`, so `build_topo` visits children in an order
set by object hashes rather than by construction, and all 68 parameter
accumulations (17 params × 4 rows) land in whatever order that gives. Float
addition is not associative, so *this engine's training is not bit-reproducible
between runs of the same script* — exercise 4 measures the same thing again.

**FINDING: "integer powers" is load-bearing, and the rule breaks three ways
outside them.**

| expression | what happens |
|---|---|
| `Value(0.0) ** 0` | forwards to 1.0, then raises `ZeroDivisionError` in backward — `k·x**(k−1)` is `0 · 0.0**-1` |
| `2 ** Value(3.0)` | `TypeError` — no `__rpow__`, as with `__rsub__` |
| `Value(-8.0) ** 0.5` | **silently stores a `complex`**, which `__repr__`'s `:.4f` cannot even print |

**4 — the gradient clip never fires, so there is one loss curve, not two.**

400 epochs on a 2-4-4-4-1 sigmoid net at seeds (42, 7, 99): the clip fires on **0
of 68,400** parameter updates, and the two curves stay within **4.4e-16** — four
ulps — of each other.

**CONTROL: even that residue is not the clip.** Two *identical unclipped* runs from
seed 42 differ by the same 4.4e-16 on about 130 of 400 epochs. It is exercise 3's
`set` again.

**MECHANISM: the clip cannot fire, because every gradient here is bounded by 2.**
The most exposed parameter is the output bias, whose gradient is a sum over 4 rows
of `2(p − t)·σ'(z)` with `|p − t| ≤ 1` and `σ' ≤ 1/4` — at most **4 × 2 × 0.25 = 2**,
and only if all four rows are maximally wrong at once. The largest gradient
observed anywhere is **0.3611**. Scaling every initial weight by 16 still peaks at
0.4445.

**FINDING: a deep sigmoid net does not explode, it vanishes.** Mean |grad| per
layer at init, input to output:

| layer | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| mean \|grad\| | 7.34e-04 | 4.25e-03 | 1.55e-02 | **1.94e-01** |

A **264×** attenuation, and each hidden layer added divides the input layer's
gradient by about 5. **A clip is a ceiling; the problem is a floor.**

**FINDING: neither arm learns XOR at all.** Final loss 0.999707, 0.999933, 0.999988
— that is 1.0, exactly the loss of predicting 0.5 on all four rows (4 × 0.25). The
network has not learned a wrong answer; it has not moved.

**CONTROL: what fixes it is a bigger init.** Multiplying the lesson's own
He-initialised weights by 8 takes the same four-layer net to **0.003071, 0.003208,
0.011993** on the same seeds and the same 400 epochs — 83× better at the worst
seed — with the clip still firing 0 times. Depth was never the obstacle; the
signal reaching layer 1 was.

**5 — layer 1 is smallest, and "after training" is the worst moment to check.**

All 17 gradients of the lesson's own 2-4-1 net after its own 1000 epochs:

| layer | mean | min | max |
|---|---:|---:|---:|
| 1 (hidden) | **3.55e-04** | 1.78e-05 | 7.06e-04 |
| 2 (output) | 5.60e-04 | 1.08e-05 | 1.07e-03 |

Layer 1 is smaller at all three seeds — ratios 0.63, 0.57, 0.43.

**FINDING: measuring after training understates the effect about tenfold.** The
same ratio *before* the first update is 0.087, 0.137, 0.033 — a **7–30×**
attenuation — against 0.43–0.63 after. Training ends at a minimum, where *every*
gradient is small, so the last epoch is the worst moment to look for a gradient
that vanishes with depth.

**FINDING: on a deeper net the question has two answers, and neither is layer 1.**
A 2-4-4-4-1 net after 1000 epochs: by mean |grad| the smallest layer is **2** at all
three seeds; by the single smallest parameter it is **3, 2, 3**. Before training,
both rules agree on **layer 1** at every seed. The ordering the Concept section
describes is real — training erases it.

**MECHANISM: at init the attenuation is monotone and about 1/5 per layer** —
adding a hidden layer divides the input layer's gradient by 4.4, 3.8, 13.9.
`σ' ≤ 1/4` and the He-scaled weights are under 1, so each layer multiplies by less
than one.

**CONTROL: the deep net's gradients are not small because it converged.** Its
final loss is 0.934676, 0.999347, 0.999958 against the 1.0 of predicting 0.5
everywhere — it never left its initialisation. So *its* post-training gradients
are small for the reason the exercise says, unlike the 2-4-1 net's, which are
small because it succeeded.
