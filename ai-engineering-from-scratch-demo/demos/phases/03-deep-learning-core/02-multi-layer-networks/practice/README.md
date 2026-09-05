<!-- generated:start -->
# 03-deep-learning-core / 02-multi-layer-networks

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/03-deep-learning-core/02-multi-layer-networks/) · upstream spec
`phases/03-deep-learning-core/02-multi-layer-networks/docs/en.md`

```bash
uv run demo practice run 02-multi-layer-networks --ex 1
uv run demo explain 02-multi-layer-networks --ex 1
uv run pytest demos/phases/03-deep-learning-core/02-multi-layer-networks
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Build a 2-4-2-1 network (two hidden layers) and run the forward pass on XOR data with random… | code | T0 | `ex01_two_hidden_layers.py` |
| 2 | Change the hidden layer size in the circle classifier from 8 to 2, then to 32. Run the forwar… | code | T0 | `ex02_hidden_layer_width.py` |
| 3 | Implement a `count_parameters` method on the Network class that returns the total number of t… | code | T0 | `ex03_count_parameters.py` |
| 4 | Build a forward pass for a 3-4-4-2 network. Feed it RGB color values (normalized to 0-1) and… | code | T0 | `ex04_rgb_color_classifier.py` |
| 5 | Replace sigmoid with a "leaky step" function: return 0.01 * z if z < 0, else 1.0. Run the for… | code | T0 | `ex05_leaky_step.py` |
<!-- generated:end -->

## Answers

Every exercise in this lesson runs a *forward pass with random weights* and asks
what came out. On this lesson's `Layer`/`Network` the answer is the same five
times: **almost nothing of the input survives the trip.** Each sigmoid layer is a
contraction, so stacking layers destroys the distinctions the network is supposed
to make, and every "classifier" here is a constant function wearing a decimal
point. The second thread running through all five is that the lesson's `forward`
pairs weights to inputs with `zip()`, so a **wrongly shaped layer is silently
accepted** rather than raised on — a defect exercises 1 and 3 each reach from a
different direction.

**1 — a 2-4-2-1 net does not transform the representation, it collapses it.**

The exercise says to print the hidden layers "to see how the representation
transforms at each layer". Measured over 2,000 random nets, what it does is
shrink — this is the distance between the two inputs XOR must separate, `[0,1]`
and `[1,0]`, as it moves through the net:

| after | mean ‖h([0,1]) − h([1,0])‖ | ratio to previous |
|---|---:|---:|
| input | 1.4142 | — |
| hidden 1 (4 wide) | 0.3753 | 0.265 |
| hidden 2 (2 wide) | 0.0644 | 0.172 |
| output | 0.0073 | 0.114 |

**192× closer together than they started.** Every layer contracts, and the deeper
one contracts harder.

**MECHANISM.** `sigmoid` is 1/4-Lipschitz — its derivative peaks at 0.25 — and
the lesson draws weights from `U(−1, 1)`, whose mean |w| is 0.5. The composed map
is a contraction, and depth compounds it. A separate consequence of the same
init: the default biases are all `0.0`, so the input `[0, 0]` leaves the first
hidden layer as *exactly* `[0.5, 0.5, 0.5, 0.5]` — at every seed tried. The
origin carries no information at all.

**ANSWER: 0 of 20,000 random 2-4-2-1 nets get XOR right.** Not a small number —
zero. At `U(−20, 20)`, the magnitude the lesson hand-tunes to in its own worked
XOR weights, 25 of 20,000 do. **Depth alone buys nothing; the weights have to
survive the contraction.**

**CONTROL.** One hidden layer *less* leaves the same pair 5.6× further apart —
2-4-1 ends 0.0411 apart against 2-4-2-1's 0.0073, same init, same seeds. The
collapse is the extra layer, not the fixture.

**FINDING — `forward` never checks a dimension.** Wire a middle `Layer(8, 2)`
into a net that feeds it only 4 values and it returns `0.6925049241102278` —
bit-identical to the correctly wired `Layer(4, 2)`, and nothing is raised. `zip()`
stops at the shorter sequence, so the four unmatched weights per neuron are
dropped in silence.

**2 — width changes the range and the distribution, and neither reaches the classifier.**

*Does the number of hidden neurons change the output range or distribution?* Yes
to both, and the "why" is one line of variance algebra. Pooled over 400 nets ×
200 points:

| h | range one net covers | pooled span | pooled sd | sd(z) / √h |
|---:|---:|---|---:|---:|
| 2 | 0.0747 | [0.212, 0.766] | 0.1041 | 0.306 |
| 8 | 0.1386 | [0.047, 0.933] | 0.1842 | 0.298 |
| 32 | 0.2199 | [0.002, 0.998] | 0.2825 | 0.284 |
| 128 | 0.2568 | [1.6e-06, 0.999933] | 0.3838 | 0.301 |

**WHY.** The output neuron sums `h` terms, so its pre-activation variance is
`Var(z) = h · Var(w) · E[a²] = (h/3) · 0.25`, i.e. sd `0.289·√h`. That last
column is the prediction: constant to within 0.021 across a **64× change in h**.
At `h=2` nothing in the sweep even reaches 0.2 or 0.8; at `h=128` the sigmoid is
saturated at both ends.

**FINDING: none of that reaches the accuracy.** 52.1, 50.3, 51.6, 52.0 % for
h = 2, 8, 32, 128 — a spread of 1.8 points. The decision is `z ≥ 0`, and widening
rescales `z` *symmetrically about that threshold*: it changes how far from the
boundary a net lands, never which side.

**FINDING: every width produces a constant classifier at the majority rate.** The
best of all 1,600 nets scores 82.5%, which is exactly the base rate of always
answering "outside" — and between 270 and 318 of the 400 nets at each width give
one identical answer for all 200 points. The lesson's own printed 17.5% is the
same degenerate net from the other end: "inside" everywhere, collecting its 35
positives.

**3 — 235,146 parameters, and the method will happily count ones that do not exist.**

`sum(n_prev · n + n)` over 784-256-128-10:

| layer | weights | biases | total |
|---|---:|---:|---:|
| 784 → 256 | 200,704 | 256 | 200,960 |
| 256 → 128 | 32,768 | 128 | 32,896 |
| 128 → 10 | 1,280 | 10 | 1,290 |
| | 234,752 | 394 | **235,146** |

Biases are **0.17%** of the model; the first weight matrix alone is **85.5%** of
it. A shallow 784-295-10 costs 234,535 — within 0.3% of the same budget for one
hidden layer instead of two, which is what makes "how many parameters" a poor
proxy for capacity.

The bias count is confirmed *without* the method: monkeypatching `ref.sigmoid` to
tally its calls shows one forward pass invokes it exactly **394** times — once per
neuron, 256 + 128 + 10.

**FINDING: it counts weights the forward pass never reads.** Declare the middle
layer as `Layer(512, 128)` between a 256-wide layer and the output and
`count_parameters` reports **267,914** against the true 235,146 — 32,768 too many,
**13.9% overstated** — while the network still returns 10 outputs and raises
nothing. Same `zip()` as exercise 1, seen from the accounting side.

**FINDING: it will double-count a layer's two contradictory shapes at once.**
`Layer(2, 5, weights=[[1.0, 1.0]])` declares five neurons but supplies one row.
`forward` loops over `len(self.weights)` and returns 1 value; `count_parameters`
takes the weight count from the rows and the bias count from `n_neurons` and
reports `7 = 2 + 5`. Neither half is ever checked against the other.

**COST.** 235,146 pure-Python parameters occupy **7.72 MB** — 32.8 bytes each, an
8-byte list slot plus a 24-byte float object. As `float32` the same weights are
0.94 MB, so the from-scratch representation costs **8.2×** the array a framework
would allocate.

**4 — the colour classifier does not classify colours.**

One 3-4-4-2 forward pass over the 8 corners of the RGB cube: black gives
`[0.8080, 0.4872]`, white gives `[0.8079, 0.4910]`. Across *all* eight corners the
two outputs move by 0.0019 and 0.0111.

**ANSWER: 282 of 300 random nets give one argmax for all 125 colours.** The
decision is `sign(out₀ − out₁)`, and its sd over the colours (0.00686) is **22×
smaller** than its mean per-net offset (0.15254) in 291 of the 300 nets. The net
picks a class when it is initialised and the colour never overturns it.

**FINDING: the two outputs are not a distribution.** They are two independent
sigmoids. Over 300 nets × 125 colours their sum averages 0.9869 — close enough to
1 to be mistaken for a softmax — but ranges `[0.3673, 1.5751]`, and only **2.11%**
of the 37,500 pairs land within 0.01 of 1. Reading them as class probabilities is
reading a coincidence of the mean.

**FINDING: which net you drew matters 5× more than which colour you fed it.** One
net's outputs move 0.0210 over the whole cube; the sd of the per-net mean is
0.1055.

**CONTROL: "normalized to 0-1" is what buys even that much response.** The same
nets on raw 0-255 bytes pin **91.6%** of first-layer activations to within 1e-6 of
0 or 1 (0.0% when normalized), leaving 39.4 of 125 colours still distinguishable
against 125.0. Sigmoid is flat past |z| ≈ 6, and unnormalised bytes are nowhere
near it.

**5 — the leaky step works, and that is the problem.**

*Does it still work?* Yes. On the Step 4 hand-tuned weights the leaky step
returns −0.120, +1.000, +1.000, −0.120 — **4/4**, the same rows sigmoid gets. But
two of them are negative, so the outputs have stopped being probabilities and only
the `>= 0.5` rule still reads them.

It is in fact *more* robust than sigmoid to the scale of the weights. Scaling all
nine parameters by `s`:

| s | 0.05 | 0.1 | 0.2 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|---|---|
| sigmoid | 2/4 | 2/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| leaky step | 2/4 | 2/4 | 4/4 | 4/4 | 4/4 | 4/4 |

Scaling `z` leaves the leaky step's sign pattern intact; below `s = 0.2` it is
sigmoid that gets pulled off its ends and fails XOR.

**WHY the smooth one is preferred: the gradient.** Central differences over all
36 (parameter, row) pairs at `s = 0.1`:

| | max &#124;∂out/∂θ&#124; | pairs with exactly zero gradient |
|---|---:|---:|
| sigmoid | 0.2496 | 8 of 36 |
| leaky step | 0.0100 | **26 of 36** |

The leaky step's slope is 0.01 below zero and **0 above**, and it makes up the
difference by jumping a full 1.0000 at `z = 0` — where sigmoid moves 2.50e-13.
There is nothing for gradient descent to follow: no slope where the unit is on,
and an infinite one at the switch.

**FINDING: at the Step 4 weights themselves, sigmoid has no usable gradient
either.** At `s = 1.0` its largest gradient is 4.55e-05 — **5,489× smaller** than
at `s = 0.1` — because `z` sits at ±10 and ±30, where sigmoid is flat. Leaky's
0.0100 is the *larger* of the two there. Smoothness pays only where a unit is not
already saturated, which is exactly the argument for ReLU in the next lesson.

**CONTROL: what solves XOR is the nonlinearity, not the smoothness.** A hard step
with no leak at all also gets 4/4. Drop the nonlinearity instead — `z → z` — and
the same weights return **370.0 for all four inputs**, 2/4. Two affine layers
compose to one.
