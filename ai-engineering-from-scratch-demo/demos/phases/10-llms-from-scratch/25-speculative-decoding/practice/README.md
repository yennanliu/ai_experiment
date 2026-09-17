<!-- generated:start -->
# 10-llms-from-scratch / 25-speculative-decoding

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/10-llms-from-scratch/25-speculative-decoding/) · upstream spec
`phases/10-llms-from-scratch/25-speculative-decoding/docs/en.md`

```bash
uv run demo practice run 25-speculative-decoding --ex 1
uv run demo explain 25-speculative-decoding --ex 1
uv run pytest demos/phases/10-llms-from-scratch/25-speculative-decoding
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Implement the exact rejection rule and empirically verify it. Run 10K samples via `speculativ… | code | T0 | `ex01_the_control_fails_the_threshold_too.py` |
| 2 | Compute the speedup formula. Given fixed `α` and `K`, plot expected tokens per target-forward… | code | T0 | `ex02_there_is_no_cost_term_so_no_optimal_k.py` |
| 3 | Train a tiny draft. Take a 124M GPT-2 target and distill a 30M GPT-2 draft on 100M tokens wit… | code | T0 | `ex03_the_hint_is_not_the_acceptance_rate.py` |
| 4 | Implement EAGLE-style tree drafting. Instead of a chain, have the draft output top-3 branches… | code | T0 | `ex04_the_tree_has_a_mask_and_no_acceptance.py` |
| 5 | Measure failure modes. Run speculative decode at temperature=1.5 (high stochasticity). Show α… | code | T0 | `ex05_alpha_rises_with_temperature.py` |
<!-- generated:end -->

## Answers

`code/main.py` is a numpy speculative-decoding simulator: a target distribution,
a draft built by blending it with a uniform, the exact rejection rule, the
`E[tokens] = (1 - a^(K+1)) / (1 - a)` formula, and a tree builder with its
attention mask. Four of the five exercises ask for a quantity the module does
not contain — a cost term, a distillation, tree acceptance — and the fifth asks
for a threshold the module's own correct output cannot pass.

All five are **T0** on the **math** dependency group (numpy).

### 1 — the control fails the threshold too

| vocab | plain sampling | speculative | difference | accept-everything |
|---:|---:|---:|---:|---:|
| 32 | **0.0175** | 0.0163 | −0.0012 | **0.1201** |
| 128 | 0.0371 | 0.0360 | −0.0011 | — |
| 512 | 0.0737 | 0.0722 | −0.0015 | — |

**ANSWER: both arms fail the exercise's 0.01 bar, at every vocabulary.** The
rejection rule is already exact and `verify_distribution` already runs both arms,
so the work is the threshold — and scoring plain sampling against the
distribution it was drawn from is the control the exercise never runs.

**MECHANISM: 0.0175 is a sampling floor, not a bias.** 10,000 draws over 32 bins
land about 0.018 from their own distribution; spreading the same draws over 16×
as many bins raises that 4.2×. The bar is below the noise of the measurement.

**FINDING: the test does discriminate.** Accepting every draft token without the
rejection rule scores **0.1201**, 6.9× the floor. So the check is sound and its
threshold is not: the right one is "indistinguishable from the control", which
both correct arms pass at a margin of 0.0012.

**FINDING: the speculative arm beating plain sampling is also noise.** It scores
lower at all three vocabularies; two unbiased estimators of the same
distribution differ by their draws, and the sign carries nothing.

### 2 — there is no cost term, so no optimal K

| alpha | K=1 | K=8 | K=64 | ceiling 1/(1−a) | % of bound at K=32 |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 1.50 | 2.00 | 2.00 | 2.00 | 100.0% |
| 0.7 | 1.70 | 3.26 | 3.33 | 3.33 | 100.0% |
| 0.9 | 1.90 | 5.70 | 9.99 | 10.00 | 96.9% |

**ANSWER: the curve saturates and never turns.** `expected_tokens` is a partial
geometric sum — non-decreasing in K and bounded — so "find the optimal K" has no
answer: the plot's own optimum is whatever K the sweep stops at.

**FINDING: there is no cost term anywhere in the module.** `expected_tokens`
takes `['alpha', 'K']`, and no name mentions cost, latency, time or speed. The
draft's own K forward passes per round are not represented.

**MECHANISM: supply the missing term and the curve turns.** Charging the draft
0.05 against the target's 1 makes wall-clock `(1 + K·c) / expected_tokens(a, K)`,
whose minimum is **K=3, K=6 and K=13** for the three alphas — an interior optimum
that moves with alpha, which is the shape the exercise is asking for.

**FINDING: what saturates is the acceptance run**, `a/(1−a)` = 1.0, 2.3 and 9.0.
At alpha 0.5, K=8 and K=64 differ by 3.9e-03: the ninth speculative token is
reached in one round in 512.

### 3 — the hint is not the acceptance rate

| `alpha_hint` | measured alpha | closed form | gap vs the hint |
|---:|---:|---:|---:|
| 0.1 | **0.617** | 0.616 | **+0.517** |
| 0.3 | 0.707 | 0.702 | +0.407 |
| 0.5 | 0.791 | 0.788 | +0.291 |
| 0.7 | 0.872 | 0.874 | +0.172 |
| 0.9 | 0.962 | 0.959 | +0.062 |

**ANSWER: `make_draft`'s promise — "expected token-level acceptance near
`alpha_hint`" — is wrong at every setting**, by as much as 0.52. A 100M-token
distillation is not available from here and the lesson ships this as the
stand-in, so the measurement is run against the stand-in's own docstring.

**MECHANISM: acceptance is `sum min(p, q)` and the blend only moves part of it.**
`make_draft` returns `a·p + (1−a)·u`. Where `p(x) > 1/V` the draft is below the
target and contributes `q(x)`; where `p(x) < 1/V` it is above and contributes
`p(x)` **at full rate whatever `a` is**. So a uniform draft that knows nothing
already accepts **0.571**.

**FINDING: the simulator is right and the parameter's name is what is wrong.**
`measure_alpha`'s 20,000 draws agree with `sum min(p, q)` to 0.005 at every hint.

**FINDING: the exercise's expected 0.6–0.7 sits at the uniform floor.** Only hint
0.1 lands in it. A 30M draft agreeing with its target 62% of the time by accident
would look, to this harness, like a successful distillation.

### 4 — the tree has a mask and no acceptance

| | `build_tree((3,3))` | chain K=3 | `build_tree((3,3,3))` |
|---|---:|---:|---:|
| nodes | 13 | 4 | 40 |
| leaves | 9 | 1 | 27 |
| mask entries | 169 | **9** | **1,600** |
| tokens per round | **2.99** | **3.09** | — |

**ANSWER: the first two sentences are already in the lesson and the third has
nothing to attach to.** `build_tree((3, 3))` *is* "top-3 branches at each of two
depths" and `tree_attention_mask` builds the mask; but `build_tree`,
`tree_attention_mask` and `validate_tree_mask` take no target, no draft and no
token between them, so "verify the target accepts the longest correct branch" is
a claim about an algorithm the module does not contain.

**FINDING: the validator and the builder compute the same traversal.** Both walk
parent pointers to the root. `validate_tree_mask` catches a mask that disagrees —
an all-ones mask returns `False` — and cannot catch a wrong parent table.

**MECHANISM: written out, the tree loses to the chain.** Best-of-three per level
with the longest accepted path winning takes **2.99** tokens per round at alpha
0.84 against a chain's **3.09**, for nine times the draft calls: selecting a
sibling without renormalising the residual over the rejected ones is exactly the
bias Lesson 15's Exercise 3 measures.

**FINDING: the mask is the only thing that scales, and it scales quadratically** —
1,600 entries against a chain's 9, for one extra accepted token at best.

### 5 — alpha rises with temperature

| T | target entropy | alpha (draft rebuilt) | alpha (draft frozen at T=1) | E[tokens] K=4 |
|---:|---:|---:|---:|---:|
| 0.5 | 1.437 | 0.769 | **0.490** | 3.17 |
| 1.0 | 2.697 | 0.855 | 0.855 | 3.74 |
| 1.5 | 3.097 | **0.896** | **0.917** | 4.06 |
| 2.0 | 3.251 | 0.919 | 0.873 | 4.25 |
| 4.0 | 3.409 | **0.954** | 0.789 | 4.56 |

**ANSWER: acceptance rises with temperature, monotonically.** The module has no
temperature — `make_target` hard-codes `standard_normal(vocab) * 1.4` — so one is
added by dividing the logits. At the 1.5 the exercise names, alpha is 0.896
against 0.855 at T=1: the algorithm gets **faster**, not slower.

**MECHANISM: the draft is a blend with a uniform, and temperature flattens the
target.** At T=4 the target's entropy is 3.409 against a uniform's ln 32 = 3.466,
and any blend of two nearly identical distributions is nearly identical to both.
The construction ties the draft's quality to the target's flatness — in the
direction that flatters it.

**FINDING: the effect the exercise means needs an independently trained draft,
and it is at the other end of the sweep.** Holding the draft at its T=1 form
makes alpha non-monotone: it peaks at T=1.5 and collapses **0.427** on the cold
side, to 0.490 at T=0.5. A draft distilled at one temperature and deployed at a
colder one is the failure mode; T=1.5 is not.

**FINDING: nothing here can be slower than plain decode.** `expected_tokens`
returns 3.17 to 4.56 across the sweep and every value is above 1. "Slower due to
draft overhead" is a claim about a quantity that has to come from outside the
module — the same gap Exercise 2 finds.
