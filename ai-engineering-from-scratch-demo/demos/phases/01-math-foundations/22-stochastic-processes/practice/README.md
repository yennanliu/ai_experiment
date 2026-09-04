<!-- generated:start -->
# 01-math-foundations / 22-stochastic-processes

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/01-math-foundations/22-stochastic-processes/) · upstream spec
`phases/01-math-foundations/22-stochastic-processes/docs/en.md`

```bash
uv run demo practice run 22-stochastic-processes --ex 1
uv run demo explain 22-stochastic-processes --ex 1
uv run pytest demos/phases/01-math-foundations/22-stochastic-processes
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Simulate 1000 random walks of 10000 steps. Plot the distribution of final positions. Verify i… | code | T0 | `ex01_random_walk_distribution.py` |
| 2 | Build a text generator using a Markov chain. Train on a small corpus: for each word, count tr… | code | T0 | `ex02_markov_text_generator.py` |
| 3 | Implement simulated annealing using Metropolis-Hastings. Start at high temperature (accept al… | code | T0 | `ex03_simulated_annealing.py` |
| 4 | Compare Langevin dynamics at different temperatures. Sample from a double-well potential U(x)… | code | T0 | `ex04_langevin_temperatures.py` |
| 5 | Implement the forward diffusion process. Start with a 1D signal (e.g., a sine wave). Add nois… | code | T0 | `ex05_forward_diffusion.py` |
<!-- generated:end -->

## Answers

**1 — mean 5.9, sd 102.4, and Gaussian by more than its moments.** Mean and
standard deviation would also fit a uniform distribution of the right width, so
normality is tested on its own terms: skew +0.044, excess kurtosis −0.103, and a
KS distance of 0.0369 against a 0.0430 critical value. 69.4% of walks land within
one sd, against the normal's 68.3%.

One detail the exercise omits: **all 1000 endpoints are even.** A ±1 walk of
10,000 steps ends at 10000 − 2·(down steps), which is always even, so the
distribution lives on half the integers. The limit is Gaussian but every finite
walk is lattice-supported, and no continuous test can be exact.

**2 — the transition matrix has a row that is not a distribution.** The last word
of the corpus has no observed successor, so its row is all zeros — 1 of 15 rows
here. A sampler that reaches it has nothing to sample from, and
`random.choices` with all-zero weights raises. The exercise's "sample from the
chain" quietly assumes every row is valid.

And on "generate new sentences": **exactly 0 of 192** generated bigrams are new.
That is provable rather than lucky — a zero transition probability can never be
sampled — so a first-order chain can only recombine observed pairs. "New
sentences" means new *paths* through seen transitions, which is precisely why
n-gram models gave way to neural ones.

**3 — scored on the final state, because best-ever tracking measures the wrong
thing.** Tracking the best point a chain ever visited turns it into random search
with memory, and measured that way a chain that never cools wins **98%** of the
time — which says nothing about annealing, whose claim is that it *settles*.

Final-state results over 40 seeds from x=8:

| strategy | mean final energy | reaches global min |
|---|---:|---:|
| **annealed 3.0 → 0.01** | **−0.431** | **18%** |
| fixed T = 3.0 | +1.613 | 10% |
| fixed T = 0.01 | +2.475 | 0% |
| greedy descent | +2.470 | 0% |

The last two rows are the finding: cold and greedy are **numerically identical**.
At T=0.01 the acceptance probability for any uphill step is effectively zero, so
the Metropolis chain *is* greedy descent. Neither endpoint of the schedule works
alone — the cooling does the work.

**4 — mixing begins at T = 0.3, against a barrier height of 1.**

| T | barrier crossings | right well |
|---:|---:|---:|
| 0.02 | 0 | 100% |
| 0.1 | 0 | 100% |
| 0.3 | 76 | — |
| 1.0 | 621 | — |
| 3.0 | 1421 | 49% |

There is no sharp critical temperature — Kramers' law makes the crossing rate
∝ exp(−ΔU/T), so crossings simply become common as T approaches ΔU = 1. Two
different bars are worth separating: crossing *starts* at T=0.3, but the wells are
only balanced (within 10% of 50/50) from T=1. "Mixes between wells" and "samples
the distribution correctly" are not the same thing, and only the second is what a
sampler needs.

**5 — neither half of the exercise holds as specified.**

100 steps do **not** reach pure noise. With β from 1e-4 to 0.02, ᾱ = 0.364, so
√ᾱ = 0.60 of the amplitude survives and correlation with the clean signal is still
+0.50. That β range is the standard DDPM schedule, designed for **1000** steps —
at 1000 it does reach noise (corr +0.03), as does β up to 0.2 over 100 steps.

And "just subtracts the estimated noise" is where the difficulty lives. The
obvious naive move, dividing by √ᾱ to undo the known scaling, is **not a
denoiser**: RMSE gets *worse*, 0.841 → 1.340, and correlation is unchanged by
construction, since scaling every sample by a constant cannot change it.

What works is a **prior**. Keeping only the strongest FFT bin recovers the sine at
RMSE **0.073** — 11× better than doing nothing, correlation +1.0000. "This is one
sinusoid" is real information, and it is the only reason the noise can be
estimated at all. A diffusion model's network learns that prior instead of being
handed it.
