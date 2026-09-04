<!-- generated:start -->
# 01-math-foundations / 16-sampling-methods

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/01-math-foundations/16-sampling-methods/) · upstream spec
`phases/01-math-foundations/16-sampling-methods/docs/en.md`

```bash
uv run demo practice run 16-sampling-methods --ex 1
uv run demo explain 16-sampling-methods --ex 1
uv run pytest demos/phases/01-math-foundations/16-sampling-methods
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Implement inverse CDF sampling for the Cauchy distribution. The CDF is F(x) = 0.5 + arctan(x)… | code | T0 | `ex01_cauchy_inverse_cdf.py` |
| 2 | Use rejection sampling to generate samples from a Beta(2, 5) distribution using a Uniform(0,… | code | T0 | `ex02_beta_rejection.py` |
| 3 | Estimate the integral of sin(x) from 0 to pi using Monte Carlo with 1,000, 10,000, and 100,00… | code | T0 | `ex03_monte_carlo_convergence.py` |
| 4 | Implement Metropolis-Hastings to sample from a 2D distribution p(x, y) proportional to exp(-(… | code | T0 | `ex04_metropolis_hastings_2d.py` |
| 5 | Build a complete text generation demo: given a vocabulary of 10 words with logits, generate s… | code | T0 | `ex05_decoding_strategies.py` |
<!-- generated:end -->

## Answers, and three exercises whose premise needed adjusting

**1 — the heavy tails, made quantitative.** 6.3% of 10,000 samples exceed |10|,
against roughly 1 in 10²³ for a standard normal; the largest draw here is 56,992.
But the property worth showing is not in the histogram at all: the Cauchy has **no
mean**, so the running average does not converge. Measured at n = 100/1000/5000/
10000 it reads +3.52, +1.40, +11.61, +8.19 — wandering, not settling. The median
meanwhile sits at +0.014 and the quartiles land within 0.06 of the exact ∓1.

**2 — the theoretical acceptance rate is 1/M = 40.69%**, measured 40.56% over
49,310 proposals. M must be the PDF maximum, which for Beta(2,5) is 2.4576 at the
mode x = 0.2; since target and proposal both integrate to 1 over [0,1], the
accepted fraction is exactly the area ratio. A 4×-too-tall envelope quarters the
rate to 10.2%, which is why rejection sampling collapses in high dimensions: the
envelope gap compounds per axis.

**3 — three single runs cannot verify a rate.** The error of one Monte Carlo run
is a random draw, and the exercise's three draws came out **0.0116, 0.0025,
0.0063** — not even in order. Repeating each sample count 200 times and comparing
RMS error gives 0.0319 → 0.0106 → 0.0032, ratios of 3.01 and 3.32 against the
√10 = 3.16 that O(1/√N) predicts, and absolute values within 15% of the
closed-form σ(b−a)/√N.

**4 — the proposal-width answer is not "maximise acceptance".**

| σ | acceptance | ESS |
|---:|---:|---:|
| 0.1 | 88.7% | 34 |
| 1.5 | 19.9% | 142 |
| **4.0** | **5.5%** | **273** |
| 12.0 | 0.8% | 77 |

Effective sample size peaks at σ=4.0, where acceptance is only 5.5% — well below
the ~23% rule of thumb. That is because this target is bimodal: the x²y² coupling
term splits it along the anti-diagonal, and the wide jumps that usually fail are
the ones that cross between modes. At σ=0.1 the chain accepts nearly everything
and explores almost nothing (τ = 354). Note also that the sample mean (2.14, 1.58)
sits *between* the modes, so it summarises neither.

**5 — top-p and top-k coincide on the logits the exercise implies.** Support sizes,
read off the lesson's own distribution functions: temperature keeps all 10 tokens
(it rescales but never truncates), top-k=3 keeps 3, and top-p=0.9 also keeps
**3** — because this logit vector happens to hold 0.9 of its mass in exactly three
tokens. The two strategies are indistinguishable on it.

Demonstrating that top-p *adapts* needs a second distribution: on a flat logit
vector it admits 9 tokens while top-k=3 still admits 3. Greedy is the control —
1 distinct token, 1 unique sequence across all 5 runs, deterministic by
construction.
