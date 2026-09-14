<!-- generated:start -->
# 08-generative-ai / 02-autoencoders-vae

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/08-generative-ai/02-autoencoders-vae/) · upstream spec
`phases/08-generative-ai/02-autoencoders-vae/docs/en.md`

```bash
uv run demo practice run 02-autoencoders-vae --ex 1
uv run demo explain 02-autoencoders-vae --ex 1
uv run pytest demos/phases/08-generative-ai/02-autoencoders-vae
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Change `β` in `code/main.py` to `0.01`, `0.1`, `1.0`, `5.0`. Record the final reconstru… | code | T0 | `ex01_every_beta_in_the_sweep_is_pareto_optimal.py` |
| 2 | Medium. Replace the Gaussian decoder likelihood with a Bernoulli likelihood (cross-entropy lo… | code | T0 | `ex02_binarising_removes_what_the_likelihoods_disagree_about.py` |
| 3 | Hard. Extend `code/main.py` into a mini VQ-VAE: replace the continuous `z` with a nearest-nei… | code | T0 | `ex03_two_of_thirty_two_is_the_mode_count_not_collapse.py` |
<!-- generated:end -->

## Answers

The lesson is a 200-line pure-Python VAE on 60 points drawn from two Gaussian
centres at ±1 with `gauss(0, 0.2)` noise. That data shape decides all three
exercises: it carries **one bit** of structure — which centre — and eight
dimensions of noise, and the noise puts a floor of `8 × 0.2² = 0.32` under any
reconstruction score. Every number below is measured against that floor.

All three exercises are **T0** — pure stdlib, as the lesson is — and each imports
the lesson's own `forward` and `backward` rather than reimplementing them, so the
two likelihood swaps and the quantiser are substitutions *into* the reference
gradient and cannot silently disagree with it.

### 1 — every β in the sweep is Pareto-optimal

Five seeds per β, the lesson's own settings, final-epoch means:

| β | recon | KL | dominated by |
|---:|---:|---:|---|
| 0.01 | 0.325 | 5.127 | nothing |
| 0.1 | 0.342 | 1.811 | nothing |
| 1.0 | 0.511 | 1.266 | nothing |
| 5.0 | 2.005 | 0.736 | nothing |

**ANSWER: all four.** Reconstruction rises with β and KL falls with it at every
step, so nothing is beaten on *both* axes. Pareto-optimality names the set that
nothing dominates — it does not pick a winner. The question as asked has four
answers, and narrowing it to one requires a scalarisation the exercise never
states.

**FINDING: the left end is pinned by the data, not by β.** β=0.01 reaches
**0.325**, which is **1.02×** the 0.32 noise floor. It is at a wall the data put
there, not fitting harder because it was allowed to — and the two floor-end βs
are not even separable: on **1 of 5** seeds β=0.1 reconstructs *better* than
β=0.01.

**FINDING: the exchange rate falls 556×.** Nats of KL bought per unit of
reconstruction given up: **200.1** between 0.01 and 0.1, **3.2** between 0.1 and
1.0, **0.36** between 1.0 and 5.0. If you must pick one, β=0.1 is the last point
bought cheaply — and saying so means admitting you scalarised.

**CONTROL: the KL axis has no floor.** β=20 drives KL to **6.3e-05** — 11,691×
below the smallest value in the sweep — and reconstruction to **0.98×** that of a
model ignoring its input entirely. `ln 2 = 0.693` sits near the 0.736 that β=5
lands on, but that is where the sweep stops, not a rate-distortion bound.

### 2 — binarising removes what the likelihoods disagree about

| decoder | valid samples | draws leaving [0, 1] | worst excursion |
|---|---:|---:|---:|
| Gaussian | 0.979 | **38.6%** | **+0.143** |
| Bernoulli | 0.987 | 0.0% | 0.000 |

**ANSWER: the same, to inside the measurement.** The 0.009 gap in sample validity
is about one standard error of a 400-draw proportion (0.007). This experiment
cannot separate the two on the axis the exercise names.

**FINDING: the swap changes the support, not the quality.** The Gaussian decoder
is unbounded and puts 38.6% of prior draws partly outside `[0, 1]` — a
probability of 1.14 for a bit. The Bernoulli decoder cannot, its output being a
sigmoid. Both then threshold to the same patterns, which is *why* the quality
numbers agree: the difference is real and it is not on that axis.

**FINDING: the exercise's own binarisation destroys its comparison.** Two
likelihoods differ in how they model the residual around the mean. Binarising
leaves **2 distinct patterns in 400 rows with 0 deviations** — centres at ±1
against `gauss(0, 0.2)` make a sign flip a 5σ event — so no residual survives and
no model of one can matter. To make the comparison read, the noise would have to
survive binarisation: centres nearer 0, or a genuine Bernoulli sampling step.

**CONTROL: the Bernoulli arm is the lesson's own `backward`.** Hand it the target
`t` for which its `2(x_hat − t)` equals `sigmoid(x_hat) − x`; measured, the two
agree to **7.6e-16**.

### 3 — 2 of 32 codes is the mode count, not collapse

| K | 2 | 8 | 32 | 128 |
|---|---:|---:|---:|---:|
| recon | 0.3275 | 0.3273 | **0.3272** | 0.3271 |
| entries used | 2 | 2 | **2** | 2 |

**ANSWER: reconstruction 0.327 on 2 of the 32 entries** — **1.02×** the noise
floor, the same wall the continuous VAE hits, reached with a **one-bit** latent
instead of two continuous dimensions.

**FINDING: 2 of 32 is the number of modes.** The exercise warns that codebook
collapse is real, and it is — but not here. `sample_mixture` draws from two
centres, so two codes is the *correct* answer, and a codebook using all 32 would
be encoding the noise around them. **A usage fraction cannot diagnose collapse on
its own**: 2/32 is a pathology only if the data has more than two modes, which is
a fact about the data and not about the model.

**FINDING: K does not matter.** Across K = 2 → 128 reconstruction spans
**0.0004** and usage stays at 2. The exercise's K=32 is neither better nor worse
than K=2; 30 of its entries are memory that is never addressed.

**FINDING: one bit buys what 5.13 nats bought.** Exercise 1's β=0.01 arm pays
5.13 nats of KL for 0.325. This pays `log2(2) = 1 bit = 0.69 nats` for 0.327 —
**7.4× less rate for the same distortion**. Almost all of the continuous latent's
capacity went on noise it could not use.

**CONTROL: the straight-through estimator is the reference gradient.**
`forward(x, params, eps=0)` returns `z` equal to `mu` to **0.0**, so the lesson's
own `dz/dmu = 1` *is* straight-through, and `beta=0` removes the KL a VQ-VAE does
not have.
