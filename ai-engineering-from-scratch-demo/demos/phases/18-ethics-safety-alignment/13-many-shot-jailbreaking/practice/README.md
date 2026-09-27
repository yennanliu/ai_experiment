<!-- generated:start -->
# 18-ethics-safety-alignment / 13-many-shot-jailbreaking

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/18-ethics-safety-alignment/13-many-shot-jailbreaking/) · upstream spec
`phases/18-ethics-safety-alignment/13-many-shot-jailbreaking/docs/en.md`

```bash
uv run demo practice run 13-many-shot-jailbreaking --ex 1
uv run demo explain 13-many-shot-jailbreaking --ex 1
uv run pytest demos/phases/18-ethics-safety-alignment/13-many-shot-jailbreaking
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Fit a power law to the shot-vs-ASR curve. Report the exponent. | code | T0 | `ex01_the_printed_exponent_is_0_461_but_the_toy_is_n_0_5_and_only_an_offset_aware_fit_recovers_it.py` |
| 2 | Implement a simple MSJ defense: run a classifier over the full context; if N pattern-match ex… | code | T0 | `ex02_rewriting_at_90pct_recall_beats_the_16_shot_cap_up_to_128_shots_then_leaks_a_rising_n_0_5_curve.py` |
| 3 | Read Anil et al. 2024 Figure 3 (power law by category). Explain why violent/deceitful content… | code | T0 | `ex03_a_2x_intercept_halves_the_shots_at_every_threshold_a_steeper_exponent_halves_them_only_at_50pct.py` |
| 4 | Design a prompt that combines PAIR iteration (Lesson 12) with MSJ. Argue whether the compound… | code | T0 | `ex04_lesson_12s_filters_make_the_compound_just_the_better_attack_and_a_30pct_pair_rate_lifts_capped_msj_to_0_40.py` |
| 5 | MSJ's mechanism is identical to ICL. Sketch a training-time defense that reduces ICL sensitiv… | code | T0 | `ex05_an_exponent_cut_on_flagged_patterns_moves_50pct_asr_to_65536_shots_but_one_unflagged_family_keeps_0_699.py` |
<!-- generated:end -->

## Answers

The lesson's toy target is `target_asr(n) = min(1, 0.02 + 0.03 * n^alpha)`,
with alpha = 0.5. It takes a shot count and nothing else. Every exercise
runs or composes that function, and exercise 4 also runs lesson 12's
`pair_loop`. Contexts use abstract placeholders only.

### 1 — the printed exponent is 0.461, but the toy is n^0.5 and only an offset-aware fit recovers it

**The shipped run prints ASR ≈ 0.038 · n^0.461.** The toy is built on an
exponent of 0.5, and a fit recovers it exactly (alpha 0.500, c 0.030) only
after subtracting the a0 = 0.02 offset before taking logs.

**The reference's `fit_power_law` is biased low.** On the noiseless curve
it returns 0.427, and 0.450 on the skill file's 5-512 grid. Over 1000 seeds
the fit averages 0.430 with a 95% range of 0.394-0.473, which excludes 0.5.
Only 7.8% of seeds fit as high as the printed 0.461.

The toy also misses the lesson's own milestones:

| shots | 5 | 32 | 256 | 512 |
|---|---:|---:|---:|---:|
| toy ASR | 0.087 | 0.190 | 0.500 | 0.699 |
| lesson says | fails | begins to succeed | reliable / "saturates" | — |

ASR first reaches 1.0 at 1068 shots.

### 2 — rewriting at 90% recall beats the 16-shot cap up to 128 shots, then leaks a rising n^0.5 curve

The defense scans the full context with a pair classifier (1% false
positives on benign pairs). Once N = 4 pairs are flagged, it rewrites every
flagged answer to a refusal, or truncates the whole context. Expected ASR is
averaged over 400 seeded contexts per shot count:

| shots | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| undefended | 0.050 | 0.062 | 0.080 | 0.105 | 0.140 | 0.190 | 0.260 | 0.359 | 0.500 | 0.699 |
| reference cap-16 | 0.050 | 0.062 | 0.080 | 0.105 | 0.140 | 0.140 | 0.140 | 0.140 | 0.140 | 0.140 |
| rewrite, 90% recall | 0.050 | 0.062 | 0.027 | 0.032 | 0.050 | 0.070 | 0.094 | 0.128 | 0.171 | 0.234 |
| truncate, 90% recall | 0.050 | 0.062 | 0.027 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rewrite, 100% recall | 0.050 | 0.062 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**The pairs the classifier misses still follow the power law.** The
rewrite's leak fits n^0.52 (offset removed, 32-512 shots), so the rewrite
beats the cap through 128 shots and loses from 256 on.

**Truncation wins on ASR by destroying benign ICL.** In a 512-pair benign
context it keeps 23.5% of benign pairs, because four false flags wipe the
context. Rewriting keeps 99.1%.

**The lesson's "61% → 2%" needs zero surviving compliant shots.** One leaked
pair already gives target_asr(1) = 0.050, and target_asr(0) = 0.

### 3 — a 2x intercept halves the shots at every threshold; a steeper exponent halves them only at 50%

**"Needs fewer shots" has two explanations, and only the curve's shape tells
them apart.** One says the model's prior against complying is weaker on
violent and deceitful content, because that text is common in fiction and
news. That shows up as a larger intercept. The other says compliance in
those categories is short and uniform, so in-context learning picks it up
faster. That shows up as a larger exponent. Both categories below reach 50%
ASR at 128 shots instead of 256:

| shots to reach | 20% | 50% | 90% | ASR at 5 shots |
|---|---:|---:|---:|---:|
| baseline | 36 | 256 | 860.4 | 0.087 |
| 2x intercept | 18 (2.00x) | 128 (2.00x) | 430.2 (2.00x) | 0.115 |
| alpha 0.5714 | 23.0 (1.57x) | 128 (2.00x) | 369.7 (2.33x) | 0.095 |

A weaker prior works like a constant shot multiplier. Faster learning gives
an advantage that grows with the target ASR. The toy can express only the
second, because `target_asr` takes (n_shots, alpha, a0) and hard-codes
c = 0.03. The reference's fit misreads the first. It gives 0.450 for the
baseline and 0.463 for the 2x-intercept category, which has the same
exponent. With the offset removed the fits are 0.500, 0.500 and 0.560 (the
steep curve is clipped at 1.0 at 512 shots).

### 4 — lesson 12's filters make the compound just the better attack; a 30% PAIR rate lifts capped MSJ to 0.40

The compound's shape: PAIR runs as the outer loop and searches for a framing
of the target query. MSJ prepends n faux pairs ("User: <harmful request
#k> / Assistant: <compliant answer #k>") to that framed query. PAIR and MSJ
are treated as independent routes to success.

| PAIR standalone rate | 0 | 10% | 30% |
|---|---:|---:|---:|
| ASR under the reference's 16-shot cap | 0.140 | 0.226 | 0.398 |
| ASR undefended, 512 shots | 0.699 | 0.729 | 0.789 |
| shots to 50% | 256 | 201 | 79 |

**The compound is worse than MSJ alone when PAIR succeeds sometimes but not
always, and it matters most against shot or context caps.** In practice
that means behaviours whose refusal depends partly on the surface form of
the request.

**With lesson 12's own filters it is never better than the better single
attack.** PAIR's rate there is 0 or 1. The keyword filter falls to every
attacker (3, 1 and 1 queries). The semantic filter falls only to encoding (1
query), and paraphrase and roleplay fail all 20. Lesson 13's `target_asr`
takes no prompt, so there is nothing in it for PAIR to optimize.

### 5 — an exponent cut on flagged patterns moves 50% ASR to 65,536 shots, but one unflagged family keeps 0.699

**The design: adversarial fine-tuning that flattens the ICL exponent (0.5 →
0.25) only on pattern families a training labeler marks harmful.**

| defense on a flagged family | shots to 50% ASR | tokens per shot in 1M |
|---|---:|---:|
| none | 256 | 3906.2 |
| intercept only (c / 4) | 4,096 | 244.1 |
| exponent 0.5 → 0.25 | 65,536 | 15.3 |

An intercept-only shift leaves room for ordinary-length shots, so the
attacker just adds shots. The exponent cut leaves no room for them.
Unflagged benign families keep 0.190 few-shot success at 32 shots.

**Primary failure mode: the labeler.** With one of ten harmful families
missed, the family-mean ASR at 512 shots falls from 0.699 to 0.216, and an
average-case eval would report a 69% cut. But an adaptive attacker picks
the missed family, which stays at the undefended 0.699. The same error runs
the other way too: the benign family labeled harmful drops from 0.190 to
0.091 at 32 shots and from 0.699 to 0.163 at 512.
