<!-- generated:start -->
# 08-generative-ai / 06-diffusion-ddpm-from-scratch

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/08-generative-ai/06-diffusion-ddpm-from-scratch/) · upstream spec
`phases/08-generative-ai/06-diffusion-ddpm-from-scratch/docs/en.md`

```bash
uv run demo practice run 06-diffusion-ddpm-from-scratch --ex 1
uv run demo explain 06-diffusion-ddpm-from-scratch --ex 1
uv run pytest demos/phases/08-generative-ai/06-diffusion-ddpm-from-scratch
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Change T from 40 to 10 in `code/main.py`. How does sample quality (visual histogram of… | code | T0 | `ex01_there_is_no_t_at_which_it_collapses.py` |
| 2 | Medium. Switch from ε-prediction to v-prediction. Re-derive the reverse step. Compare final s… | code | T0 | `ex02_v_prediction_is_the_same_net_with_a_rotated_target.py` |
| 3 | Hard. Add classifier-free guidance. Condition on a class label `c ∈ {0, 1}`, drop it 10% of t… | code | T0 | `ex03_guidance_trades_the_mode_for_its_tail.py` |
<!-- generated:end -->

## Answers

The lesson is a 181-line DDPM on a 1-D two-mode mixture. All three exercises are
**T0**, and all three end up pointing at the same line of code:

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
```

The per-step noise range is hard-coded and **independent of T**, so the *total*
noise is whatever T steps of it happen to add up to. That makes `alpha_bar[T-1]`,
not T, the variable everything else tracks.

A shared metric replaces the visual histogram: the share of samples within 1.0 of
either centre (**in-modes**) and the share in the gap `|x| < 0.5`
(**in-valley**). Real data scores **0.980 / 0.000**.

### 1 — there is no T at which it collapses

| T | 5 | 10 | 20 | 40 | 100 | real |
|---|---:|---:|---:|---:|---:|---:|
| `alpha_bar[-1]` | 0.951 | 0.904 | 0.817 | 0.667 | 0.364 | — |
| in-modes | 0.400 | 0.550 | 0.752 | **0.867** | 0.895 | 0.980 |
| in-valley | 0.323 | 0.189 | 0.077 | **0.020** | 0.003 | 0.000 |

**ANSWER: smoothly, and it never collapses.** At T=10 both modes are still
there, just blunt. There is no fall between neighbouring T anywhere in the sweep,
so "at what T does the structure collapse" has no answer to give.

**FINDING: T is not the variable.** The β range is identical at T=10 and T=1000,
so changing T rescales the noise budget rather than re-timing a fixed one.

**FINDING: the forward process never reaches noise, even at the lesson's own T.**
At T=40, `x_T` still carries **81.7%** of the signal — yet `sample()` starts the
reverse chain at `gauss(0, 1)`. The sampler begins off the manifold its own
trainer built, at every T in the sweep. Reaching `alpha_bar ≈ 0` needs ~T=1000.

### 2 — v-prediction is the same net with a rotated target

**FINDING: the reverse step does not need re-deriving, only inverting.**
`v = √ᾱ·ε − √(1−ᾱ)·x₀` and `x_t = √ᾱ·x₀ + √(1−ᾱ)·ε` are a rotation, so
`ε = √ᾱ·v + √(1−ᾱ)·x_t` inverts it exactly — checked to **1e-16** over 500 random
triples. The "re-derivation" is one substitution into the lesson's sampler.

**ANSWER: v-prediction wins here, on both numbers.** Same network, same
optimiser, same 4,000 steps — only the regression target moved.

**FINDING: the difference is scale, not principle.** Because ᾱ never falls below
0.667, `v` stays near `√ᾱ·ε`, and the v target carries **1.17×** the spread of the
ε target across the schedule. What changed is how big a number the regression has
to hit.

### 3 — guidance peaks at w=1 and then goes backwards

| w | 0 | 1 | 3 | 7 |
|---|---:|---:|---:|---:|
| hit rate | 0.527 | **0.562** | 0.520 | 0.490 |
| sample spread | 1.73 | 1.66 | 1.50 | **1.48** |

**ANSWER: not monotone.** Guidance buys **+0.035** at `w=1` and has given all of
it back by `w=3`, ending **below** the unguided baseline at `w=7`. The exercise
asks for 0, 1, 3, 7 as though the rate rose throughout.

**FINDING: what guidance reliably does is narrow the output.** Spread falls
monotonically **1.73 → 1.48** across the same sweep in which the hit rate turns
over — the diversity cost guidance is known for, and the only column that moves
in one direction.

**FINDING: the signal being amplified is weak to begin with.** The unguided hit
rate is 0.527 against a ceiling of 1.0, for exercise 1's reason.

**FINDING: the subtrahend is the least-trained thing in the model.** At 10% drop
the unconditional input is seen **594** times in 6,000 steps against **2,703** per
label, so `w` multiplies the error of the least-trained branch. That is a
mechanism for the turnover, and this experiment does not isolate it.

**CONTROL:** at `w=0` the code takes `eps_cond` without evaluating the
unconditional pass, so the baseline is this same network rather than a separate
run.
