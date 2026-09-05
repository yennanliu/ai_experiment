<!-- generated:start -->
# 03-deep-learning-core / 08-weight-initialization

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/03-deep-learning-core/08-weight-initialization/) · upstream spec
`phases/03-deep-learning-core/08-weight-initialization/docs/en.md`

```bash
uv run demo practice run 08-weight-initialization --ex 1
uv run demo explain 08-weight-initialization --ex 1
uv run pytest demos/phases/03-deep-learning-core/08-weight-initialization
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add LeCun initialization (Var = 1/fan_in, designed for SELU activation). Run the 50-layer exp… | code | T0 | `ex01_lecun_vs_xavier.py` |
| 2 | Implement the GPT-2 residual scaling: multiply the output of each layer by 1/sqrt(2*N) before… | code | T0 | `ex02_residual_scaling.py` |
| 3 | Create an "init health check" function that takes a network's layer dimensions and activation… | code | T0 | `ex03_init_health_check.py` |
| 4 | Run the experiment with fan_in = 16 vs fan_in = 1024. Xavier and Kaiming adapt to fan_in, but… | code | T0 | `ex04_fan_in_16_vs_1024.py` |
| 5 | Implement orthogonal initialization (generate a random matrix, compute its SVD, use the ortho… | code | T0 | `ex05_orthogonal_vs_kaiming.py` |
<!-- generated:end -->

## Answers

Every strategy in this lesson is the same random matrix times a scalar —
`forward_deep` re-seeds to 42 on each call and each init draws the same
`random.gauss` values in the same order, so random N(0,1), random N(0,0.01),
Xavier and Kaiming agree to **exactly 0.0** after dividing by 1, 0.01, `√(1/n)`
and `√(2/n)`. **The whole lesson varies one number**, and the five exercises are
five ways of asking what that number has to be.

The answer is a single closed form, and every check below is a consequence of it:

```
per-layer magnitude gain  =  σ · √(fan_in / 2)      (ReLU)
                          =  σ · √(fan_in) · slope  (any activation, slope at 0)
```

**1 — on a square stack, LeCun *is* Xavier, bit for bit.**

`2/(fan_in + fan_out) == 1/fan_in` exactly when `fan_out == fan_in`, and
`forward_deep` only ever builds width × width layers. Over 50 layers the worst
difference between the two runs is **exactly 0.0**, at widths 16, 64, 100, 256 and
1024.

The shared trajectory fails the lesson's own criterion: mean |activation| falls
0.5465 → **0.0576** over 50 layers, and 49 of the 50 layers sit outside the
flowchart's `[0.5, 2.0]` band.

**MECHANISM: tanh contracts, so a gain of 1 has no non-zero fixed point.**

| rms of z | E[tanh(z)²] / E[z²] |
|---:|---:|
| 1.0 | 0.397 |
| 0.5 | 0.697 |
| 0.3 | 0.857 |

Under 1 everywhere, nearing 1 only as the signal dies. `Var(w) = 1/fan_in`
preserves variance across the *linear* map alone. Layer-50 magnitude by forward
gain: 0.0924 at g=1.000, 0.4640 at 1.414, **0.5628 at 1.667** (tanh's usual 5/3).

**ANSWER: LeCun is stable with the activation the exercise names.** LeCun + SELU
holds 0.8256 → 0.7710 — a **7%** change over 50 layers against tanh's 89%. SELU's
negative branch amplifies, which is exactly what repairs the contraction.

**CONTROL: the two formulas do differ — on layers that are not square.** Over 50
alternating 16/64 linear layers LeCun ends at rms 1.1040 and Xavier at
**4.17e-03**, a factor of 265. Xavier averages the forward and backward
requirements; a non-square layer then meets neither.

**2 — 1/√(2N) does not slow linear growth, it stops exponential growth.**

| layer | 0 | 1 | 10 | 25 | 50 |
|---|---:|---:|---:|---:|---:|
| unscaled rms | 9.68e-01 | 1.65e+00 | 1.81e+02 | 8.91e+05 | **3.30e+11** |
| scaled by 1/√(2·50) | 0.9681 | — | — | — | **1.6334** |

**FINDING: the growth is exponential, not proportional to N.** The variance factor
is **2.893 per layer**, so L50 sits 3.41e+11× above L0 — **10.7 decades** past the
`√51 = 7.14×` that "proportional to N" predicts.

**MECHANISM: the sublayer output scales with its input, so the adds multiply.**
`E[f(x)²]/E[x²] = 1.970` for one fresh Kaiming-ReLU-Kaiming block — two Kaiming
layers double the second moment, ReLU halves it — so `x + f(x)` carries 2.97× the
variance.

**CONTROL: pre-normalise the sublayer input and the linear rate appears.** An
RMS-norm in front of the sublayer — what a pre-LN transformer actually has — holds
the *unscaled* stream to **10.347×** against the claimed `√(1 + 2·50) = 10.050`.

**ANSWER: 1/√(2N) bounds the stream at a depth-independent variance.**

| N | 10 | 25 | 50 | 100 |
|---|---:|---:|---:|---:|
| scaled variance ratio | 2.757 | 2.966 | 2.846 | 2.732 |
| unscaled | 1.87e+02 | 9.21e+05 | 3.41e+11 | 1.89e+22 |

The closed form is `(1 + 1/N)^N`, rising to **e = 2.718**.

**FINDING: the 2 in 2N counts two residual adds per block, not one.** A normalised
sublayer scaled by `1/√(2N)` over 50 *single*-add layers gives 2.127 against the
exact `1 + 2N/(2N) = 2`. GPT-2 scales blocks that add twice; one add per layer
wants `1/√N`.

**3 — a health check from one formula calls all six configs, and the lesson's own metric ranks the dead ones top.**

`fan_in · std² · slope²` and nothing else predicts every verdict — zero+sigmoid
symmetric, random(1.0)+relu explodes, random(0.01)+relu dies, xavier+sigmoid dies,
xavier+tanh ok, kaiming+relu ok — each matching the measured 50-layer outcome, and
the magnitude to **0.7 decades** over the +37 and −63 decades the two random inits
actually reach.

**FINDING: the flowchart's `[0.5, 2.0]` band ranks the two dead configs first.**

| config | layers in band | across-sample spread at L50 |
|---|---:|---:|
| zero + sigmoid | **50/50** | **0.0e+00** |
| xavier + sigmoid | 30/50 | 1.8e-17 |
| kaiming + relu | 16/50 | 0.130 |
| xavier + tanh | 1/50 | 0.091 |

**MECHANISM: sigmoid's offset pins mean |a| at 0.5 while the signal dies.**
xavier+sigmoid holds mean |a| at **0.5005** for all 50 layers — sigmoid is centred
on 0.5 — while *deviations* contract by 0.235 per layer, against `sigmoid'(0) =
0.25`, which `2/(fan_in + fan_out)` ignores entirely.

**FINDING: at this depth no init scale rescues sigmoid** — spread at layer 50 with
Xavier scaled ×1, ×2, ×4, ×6 is 1.8e-17, 4.8e-17, 1.4e-11, 7.9e-08. All dead.
**CONTROL: the same rule does fix tanh**, which has no offset — std 0.2083 (gain
5/3) against Xavier's 0.1252 lifts the layer-50 spread 0.0911 → **0.6268**.

**4 — the band of scales that works narrows as 1/√(fan_in).**

| fan_in | 16 | 64 | 256 | 1024 |
|---|---:|---:|---:|---:|
| Kaiming, magnitude at L6 | 0.26 | 0.724 | 0.566 | 0.629 |
| Kaiming, per-layer gain | 0.984 | 1.066 | 1.021 | 1.017 |
| random N(0,1), at L6 | 133 | 2.37e+04 | 1.19e+06 | **8.44e+07** |
| random N(0,1), gain | 2.78 | 6.03 | 11.6 | **23** |

**MECHANISM: the gain is `σ·√(fan_in/2)`** — measured over the closed form it is
0.984, 1.066, 1.021, 1.017, worst deviation **6.6%**. σ = 1.0 and σ = 0.01 give the
same ratio to the last digit, because the two matrices are proportional.

**ANSWER — the gap made precise.** A 50-layer net stays inside `[1e-6, 1e6]` only
while the gain is in `[0.7586, 1.3183]`, so:

| fan_in | admissible σ | width |
|---|---|---:|
| 16 | [0.26820, 0.46607] | 0.19788 |
| 1024 | [0.03352, 0.05826] | **0.02473** |

**8.0× narrower for a 64× wider layer** — and `σ = 1.0` is outside every one of
them.

**FINDING: Xavier is not neutral under ReLU.** Its gain is 0.696, 0.754, 0.722,
0.719 — because `√(1/n)·√(n/2) = 0.7071` with the fan_in cancelling. Over 50 layers
that is `2⁻²⁵ = 3.0e-08`. **That is why Kaiming's extra factor of 2 exists.**

**5 — orthogonal initialization vanishes at gain 1, and beats Kaiming at gain √2.**

At layer 50, width 64, ReLU:

| init | magnitude at L50 | per-layer gain |
|---|---:|---:|
| orthogonal, gain 1 | **1.551e-08** | 0.7181 |
| Kaiming | 0.2572 | 1.0068 |
| orthogonal, gain √2 | **0.5203** | 1.0156 |

0.718 is `1/√2` — the half a ReLU removes, which an orthogonal matrix does nothing
to replace.

**MECHANISM: the factor really is orthonormal.** `‖QᵀQ − I‖_max = 1.8e-15` and
`‖Qv‖ − ‖v‖ = 1.8e-15`. With the activation removed, 50 such layers take the
magnitude 0.7969 → **0.7876** — a per-layer gain of **0.9998 ± 0.0108**. Every bit
of the decay above is the ReLU.

**FINDING: Kaiming's √2 is a ReLU correction, not a general one.** The same stack
with the activation removed has a per-layer gain of **1.4151** — √2 to 0.001 — and
reaches **2.551e+07** by layer 50.

**FINDING: "use the orthogonal matrix U" is the wrong matrix off the diagonal.**
For a 64 → 16 layer, `svd(A, full_matrices=False)` returns U of shape **(16, 16)**
where the weight matrix must be **(16, 64)**; `U @ Vᵀ` supplies it. The recipe
reads correctly only because `forward_deep` is square throughout.

**CONTROL: orthogonal fixes the scale exactly; Kaiming draws it.** Per-layer gain
across the 49 transitions with the activation removed: **0.9998 ± 0.0108**
orthogonal against 1.4151 ± 0.0729 Kaiming — **6.8× tighter**. Under ReLU the
masking dominates and the spreads converge (0.1979 against 0.2221).
