<!-- generated:start -->
# 08-generative-ai / 13-flow-matching-rectified-flows

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/08-generative-ai/13-flow-matching-rectified-flows/) · upstream spec
`phases/08-generative-ai/13-flow-matching-rectified-flows/docs/en.md`

```bash
uv run demo practice run 13-flow-matching-rectified-flows --ex 1
uv run demo explain 13-flow-matching-rectified-flows --ex 1
uv run pytest demos/phases/08-generative-ai/13-flow-matching-rectified-flows
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py` and compare 1-step vs 20-step MSE vs the true data distribution. | code | T0 | `ex01_one_step_lands_every_sample_on_zero.py` |
| 2 | Medium. Switch from uniform `t` sampling to logit-normal (concentrates sampling at mid-t). Do… | code | T0 | `ex02_logit_normal_starves_the_ends_and_helps_nothing.py` |
| 3 | Hard. Implement one reflow iteration: generate paired (x_0, x_1) by integrating the first mod… | code | T0 | `ex03_reflow_copies_the_teacher_not_the_data.py` |
<!-- generated:end -->

## Answers

The lesson is 1-D flow matching on a two-mode mixture `0.5 N(-2, 0.3²) + 0.5
N(2, 0.3²)`, with a 24-unit MLP trained by single-sample SGD. All three
exercises are **T0** and stdlib-only.

Two measuring tools carry all three answers. **Quantile MSE** — sorted samples
against the mixture's exact quantiles, i.e. squared 1-D Wasserstein-2 — is the
"MSE vs the true data distribution". And the **exact velocity field**
`v*(x, t) = E[x₁ − x₀ | x_t = x]` is closed-form for a Gaussian mixture, so the
network's error can be separated from the sampler's.

### 1 — one step lands every sample on zero

| steps | 1 | 2 | 4 | 8 | 20 |
|---|---:|---:|---:|---:|---:|
| lesson network (seed 31) | 1.249 | **0.236** | 0.338 | 0.434 | 0.521 |
| exact field | 4.090 | 0.473 | 0.057 | 0.024 | 0.017 |

**ANSWER: 1-step 1.249, 20-step 0.521.**

**FINDING: one step from the *perfect* field lands every sample on 0.** At
`t = 1` both components have mean 0, so `v*(x, 1) = x` and `x − 1·v*(x, 1) = 0`
(largest |sample| **4e-16**). One step returns the data *mean*, in neither mode.
The network's 1-step row beats the perfect field's only because it is wrong at
`t = 1`. "A single Euler step from t=1 to t=0 would work" holds only once the
paths have been straightened — which is exercise 3.

**FINDING: the lesson's left/right count cannot see this.** At 1 step it reads
**514 / 486** while only **32%** of samples are within 3σ of a mode.

**FINDING: more steps converge to the network, not the data.** The network's
2-step error beats its 20-step one, and its 20-step error is **31×** the exact
field's. The lesson's "4 steps matches 20" is true, but because both are
limited by the model, not by integration.

### 2 — logit-normal starves the ends and buys nothing in the middle

Means over six seeds, same loop (which matches the lesson's `train` to **0.0**
with a uniform draw):

| | v-error, all t | ends | middle | 20-step sample |
|---|---:|---:|---:|---:|
| uniform | 0.508 | 0.613 | 0.446 | 1.150 |
| logit-normal | 0.813 | 1.393 | 0.666 | 1.259 |

**ANSWER: no — worse on every measure.**

**FINDING: the ends are starved.** Logit-normal puts **2.8%** of draws at
`t ≤ 0.1` or `t ≥ 0.9` against uniform's **20%**, and end error rises **2.3×**.
Every Euler sample's first query is at `t = 1`, where logit-normal's density is
zero.

**FINDING: the middle gets 1.5× the draws and still gets worse**, 0.446 → 0.666.
SD3's rationale is that mid-t is where the prediction is hardest in
high-dimensional image space; in this toy it is not the bottleneck.

**FINDING: the seed decides more than the sampler.** Uniform seeds span
**0.276–0.709**; the best logit-normal seed (**0.364**) beats the worst uniform
one. A single run of `main()` could have returned either verdict.

### 3 — reflow copies the teacher, not the data

**ANSWER: one reflow cuts 1-step error from 1.211 to 0.877.**

**FINDING: the student learns the teacher.** Its 1-step samples are **0.266**
from the teacher's 20-step distribution but **0.877** from the data. With the
exact field as teacher, the same student code reaches **0.179** in one step —
against that field's own 1-step **4.09**. Reflow's ceiling is the teacher's
multi-step output; it straightens mistakes along with paths.

**FINDING: the doc's "paths cross" mechanism, counted.** Pair-lines cross
**52.2%** of the time under independent pairing and **0.0%** under reflow
pairing — a 1-D ODE flow cannot reorder points. The mean squared gap between a
sample's 1-step and 20-step landing falls **2.149 → 0.122**.

**CONTROL:** the same loop on *independent* pairs scores **2.465** at one step,
so the gain is the pairing, not the extra training.
