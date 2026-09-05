<!-- generated:start -->
# 03-deep-learning-core / 04-activation-functions

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/03-deep-learning-core/04-activation-functions/) · upstream spec
`phases/03-deep-learning-core/04-activation-functions/docs/en.md`

```bash
uv run demo practice run 04-activation-functions --ex 1
uv run demo explain 04-activation-functions --ex 1
uv run pytest demos/phases/03-deep-learning-core/04-activation-functions
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Implement Parametric ReLU (PReLU) where the negative slope alpha is a learnable parameter. Tr… | code | T0 | `ex01_prelu_alpha.py` |
| 2 | Run the vanishing gradient experiment with 50 layers instead of 10. Plot the magnitude at eac… | code | T0 | `ex02_fifty_layers.py` |
| 3 | Implement the ELU (Exponential Linear Unit): elu(x) = x if x > 0, alpha * (e^x - 1) if x <= 0… | code | T0 | `ex03_elu_dead_neurons.py` |
| 4 | Build a "gradient health monitor" that runs during training: at each epoch, compute the avera… | code | T0 | `ex04_gradient_health_monitor.py` |
| 5 | Modify the training comparison to use the XOR dataset from Lesson 01 instead of circles. Whic… | code | T0 | `ex05_xor_instead_of_circles.py` |
<!-- generated:end -->

## Answers

The lesson's own narrative is that sigmoid vanishes and ReLU fixes it. On the
lesson's own code, **measured, that narrative inverts twice**: over 50 layers
sigmoid is the only activation whose signal survives and ReLU is the only one
that dies (ex 2), and a gradient-health alarm set at the lesson's own thresholds
fires on ReLU and GELU in the lesson's own 10-layer experiment while staying
silent on sigmoid (ex 4).

The other thread is that three of the five exercises ask a comparative question
that this code cannot answer, because `ActivationNetwork.__init__` calls
`random.seed(0)` — every comparison it ships is **one draw its caller has no way
to vary**.

**1 — PReLU's alpha learns its way out of Leaky ReLU's range, and it barely helps.**

Alpha runs **0.0100 → −2.6283** over 200 epochs, crossing zero inside the first.
It is genuinely trained: `Σᵢ dL/dzᵢ · min(zᵢ, 0)` = −6.50939919e-02 against a
central difference of −6.50939919e-02, relative error **5.66e-10**.

A negative slope makes the unit a **V** — `x` for `x > 0`, `2.63|x|` below —
non-monotone and positive on both sides. The circle label is even in each
coordinate, so one V does what two ReLUs would have to.

| | loss | accuracy | epochs to 0.05 |
|---|---:|---:|---:|
| PReLU (learned α) | 0.0097 | 98.5% | 16 |
| fixed Leaky ReLU (α = 0.01) | 0.0099 | **99.0%** | 16 |
| **frozen at the learned α = −2.6283** | **0.0059** | — | **5** |

A 2% edge on loss and half a point of accuracy *the wrong way*, for a parameter
that moved by 2.6. **FINDING: the α it finds is worth 3.2× more than learning
it** — frozen at −2.6283 from epoch 0 the same net converges in 5 epochs, not 16.
PReLU spends the run travelling there, training under a slope that is only good
at the end.

**CONTROL: α = 1 is the ridge the descent runs away from.** There `leaky_relu` is
the identity, the hidden layer is linear, and the net is one logistic regression:
loss stuck at 0.2046, accuracy **71.5%** — exactly the 143/200 base rate of calling
every point "outside". α descends *away* from that ridge, which is why the learned
slope goes negative.

**2 — at 50 layers, only ReLU's signal reaches zero.**

| activation | behaviour through 50 layers | ends at |
|---|---|---:|
| sigmoid | never below 0.0705 | **0.9659** |
| tanh | bottoms 5.69e-06 (layer 15), recovers | 1.12e-03 |
| GELU | below 1e-06 from layer 7, never exactly 0 | 1.03e-18 |
| ReLU | **exactly 0.0 from layer 6** | 0.0 |

**FINDING: that is the lesson's ranking upside down.** The activation blamed for
vanishing gradients is the only one whose signal survives; the one credited with
fixing it is the only one that dies. **MECHANISM:** `sigmoid(z) = 0.5 + z/4 + O(z³)`
pulls a fading signal back to its non-zero centre; `relu(0) = 0` absorbs, and the
next `z` is `0 · Σw`.

**FINDING: the gradient, on the same trace, ranks them the other way** —
`Π act'(z_l)·Σw_l` gives sigmoid **5.181e-38**, gelu 9.815e-17, tanh 5.355e-02,
relu 0.0. Sigmoid is 36 orders below tanh while its *signal* is the healthiest of
the four. **The magnitude the experiment prints is not the quantity its title
names.**

Measured per-layer factor against the textbook ceiling (geometric mean over 400
runs): sigmoid **0.2182** against the lesson's 0.25, tanh 0.5241 against its 1.0,
gelu 0.4851, relu 1.1604 over its 193 surviving layers.

**CONTROL: "layer 6" is a coin flip, not a property of ReLU.** Over 400
independent streams ReLU reaches exactly 0 in **400/400** runs, median layer **2**.
It dies the first time `Σw` is negative, so the index is geometric with p = 1/2 —
seed 42 merely got a long run. GELU dies in 36/400; sigmoid and tanh in 0.

**3 — ELU has no dead units, and the lesson's detector cannot see the difference.**

Units firing for no input after 200 epochs on circles:

| lr | 0.1 | 0.3 | 0.5 | 1.0 | 2.0 |
|---|---:|---:|---:|---:|---:|
| ReLU | **0** | 2 | 2 | 2 | 2 |
| ELU | 0 | 0 | 0 | 0 | 0 |

At the lesson's own `lr = 0.1` the dying-ReLU rate the exercise asks about is
**0%**. The 25% starts at `lr = 0.3`.

**FINDING: `dead_neuron_detector` with its `relu` replaced by ELU gives identical
fire counts on all 20 neurons** over 20,000 pre-activations (10,037 of them ≤ 0) —
0 dead for both. `elu(z) > 0` exactly when `relu(z) > 0`, so its `act(z) > 0` test
is a test of `sign(z)`, not of the activation.

**MECHANISM: the difference is in the derivative, not the output.** `relu'(z)` is
exactly 0.0 on 10,037 of those values; `elu'(z) = α·e^z` never falls below
8.50e-06. A ReLU unit negative on every sample gets zero into `w1` and `b1` and
can never come back; ELU has no such state.

**FINDING: dead units are not what costs accuracy here.** The best fit in the
whole sweep is **ReLU at lr = 0.5** — loss 0.0002, accuracy 100.0% — *with 2 of 8
units dead*. ELU keeps all 8 alive for a best loss of 0.0058.

**CONTROL: at α = 0 this ELU is the lesson's ReLU exactly** — max |difference| over
20,000 values is **0.0**.

**4 — the health monitor never fires where the lesson runs it, and its upper threshold cannot fire at all.**

The monitor reads its per-layer means out of the state `backward` is about to use;
it reproduces `backward`'s own weight delta to **1.05e-15**, so its numbers are the
optimiser's, not a parallel model.

**ANSWER: 0 of 2000 layer-epoch readings** fall outside `[0.001, 100]` — 5
activations × 200 epochs × 2 layers at the lesson's own `lr = 0.1` on its own
200-point circle data.

**MECHANISM: the upper threshold is unreachable.** `|d_out| = |p − t|·p(1−p) ≤ 0.25`,
so the output-layer gradient is at most `0.25|h|` and the hidden one at most
`0.25|w2|·act'·|x|`. Reaching **100** needs `|h| > 400` or `|w2| > 200`, which a
sigmoid output cannot ask for. The largest single gradient over 5 activations × 4
learning rates is **2.2506** — 44× below the alarm.

**FINDING: the low threshold fires where training *succeeded*.** At `lr = 1.0` the
first warning is relu epoch 145 (ending 99.0%), gelu 175 (97.5%), swish 142
(96.5%). The per-layer mean falls under 0.001 because the loss is near zero. On
this network "gradient too small" is a **convergence signal, not an alarm** — the
whole live range is 7.2e-08 to 5.4e-02.

**CONTROL: the threshold is right; its location is not.** Run the same rule over
the lesson's own 10-layer `vanishing_gradient_experiment`:

| | fires at | last three layers |
|---|---|---:|
| relu | **layer 5** | 0.000000 |
| gelu | **layer 4** | 0.000000 |
| sigmoid | never | 0.521604 |

The alarm catches the activation the lesson credits with fixing vanishing
gradients, and stays silent on the one it blames.

**5 — ReLU converges fastest on XOR; GELU and Swish win on circles; and neither answer is stable.**

At the lesson's own settings (`lr = 0.1`, 200 epochs) the comparison is
**degenerate**:

| | final loss | accuracy |
|---|---:|---:|
| sigmoid | 0.2550 | **50%** (chance) |
| tanh | 0.1504 | 100% |
| **relu** | **0.0499** | 100% |
| gelu | 0.0511 | 100% |
| swish | 0.1071 | 100% |

Only ReLU reaches loss < 0.05, and it does so at **epoch 200 — the last one**.

Given room (10,000 epochs), epochs to loss < 0.05: **relu 200**, gelu 203, swish
273, tanh 345, sigmoid 2698.

**WHY it differs from circles — a different winner, in a different unit.** On
circles: gelu 13, swish 13, relu 16, tanh 28, sigmoid 123 epochs. But an epoch is
**4 updates on XOR and 200 on circles**:

| | XOR (updates) | circles (updates) |
|---|---:|---:|
| relu | **800** | 3200 |
| gelu | 812 | **2600** |
| swish | 1092 | **2600** |
| tanh | 1380 | 5600 |
| sigmoid | 10792 | 24600 |

Every activation needs *fewer* updates on XOR — four clean rows against 200 noisy
ones. What changes is which one collects them fastest.

**MECHANISM: XOR zeroes most of the hidden-layer gradient outright.** 93.9% of
ReLU's hidden weight gradients are **exactly zero** on XOR against 36.0% on
circles. `d_h · x[j]` is exactly 0 whenever `x[j]` is, and half of XOR's
coordinates are 0; on circles no coordinate is ever exactly 0, so its 36.0% is
dead units alone.

**FINDING: on XOR the answer is not stable, and the lesson cannot see that.**
Redraw the same init at three seeds and the XOR winner is **relu, tanh, gelu** —
with ReLU failing to reach loss < 0.05 in 2,000 epochs at one of them. On circles
it is gelu, swish, gelu — always one of the two smooth activations, and nothing
fails. `ActivationNetwork.__init__` calls `random.seed(0)`, so the comparison the
lesson ships is one draw and its caller has no way to ask for another.
