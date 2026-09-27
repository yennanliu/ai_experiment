<!-- generated:start -->
# 18-ethics-safety-alignment / 20-bias-representational-harm

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/18-ethics-safety-alignment/20-bias-representational-harm/) · upstream spec
`phases/18-ethics-safety-alignment/20-bias-representational-harm/docs/en.md`

```bash
uv run demo practice run 20-bias-representational-harm --ex 1
uv run demo explain 20-bias-representational-harm --ex 1
uv run pytest demos/phases/18-ethics-safety-alignment/20-bias-representational-harm
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Report WEAT-style bias scores before and after the debiasing step. Explai… | code | T0 | `ex01_debias_cuts_0_891_to_0_221_and_the_residual_is_the_identity_words_own_tech_care_loading_not_4_d.py` |
| 2 | Extend the probe with an intersectional test: (gender, race) x (career, family). Report cross… | code | T0 | `ex02_an_intersection_only_stereotype_moves_neither_single_axis_probe_and_debias_erases_race_by_truncating_to_4_d.py` |
| 3 | Read An et al. 2025 (PNAS Nexus). Identify the two intersectional effects they report that si… | code | T0 | `ex03_an_et_al_find_black_men_the_only_group_below_baseline_at_0_303_and_black_women_highest_not_penalized.py` |
| 4 | Yu & Ananiadou 2025 identify gender neurons. Sketch a falsification experiment that would dis… | code | T0 | `ex04_zero_ablating_the_gender_units_raises_bias_51pct_and_moves_a_pure_correlate_12pct_only_interchange_separates_them.py` |
| 5 | The meta-critique argues the field focuses too narrowly on binary gender. Pick one under-stud… | code | T0 | `ex05_a_disability_probe_survives_the_gender_debias_bit_for_bit_and_needs_5_terms_a_side_the_lessons_3_floor_p_at_0_05.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py`, a 4-d WEAT probe with
identity sets he/his/man vs she/her/woman, career words X and family words
Y, and a `debias()` that projects [1, -1, 0, 0] out of those six attribute
words. Exercise 3 also uses figures from An et al.'s paper, taken from its text.

### 1 — debias cuts 0.891 to 0.221, and the residual is the identity words' own tech/care loading, not 4-d

**The score is +0.8906 before debias and +0.2212 after, a 75.2% cut.** After
the projection every attribute word has equal masculine and feminine
loadings. For example, engineer becomes [0.2, 0.2, 1, 0].

**The residual has nothing to do with the toy being 4-d.** The TAKEAWAY
blames the dimension. But embedding the debiased vectors in 300-d with a
random orthonormal basis changes the post score by less than 1e-9, because
cosine does not change under any isometry.

**The residual comes entirely from the identity words.** "he" loads 0.2 on
tech and "she" 0.2 on care, and `debias()` never edits identity words.
Zeroing axes 2-3 of the six identity words takes the post score to 0.0000.

**The printed "effect size" is the raw statistic.** Caliskan's standardized
d goes from 1.81 to 1.60, only an 11.5% cut.

### 2 — an intersection-only stereotype moves neither single-axis probe, and debias erases race by truncating to 4-d

The embedding gains race axes R1 and R2. Each gender word is composed with
each race to make four cells. Attribute words carry a race-level
stereotype: career +0.3 on R1, family +0.3 on R2. The (F, R2) cell alone
gets +0.5 care. Scores from the reference `weat_score`:

| contrast | score |
|---|---:|
| gender gap inside R1 (M-R1 vs F-R1) | 0.6004 |
| gender gap inside R2 (M-R2 vs F-R2) | 0.8360 |
| race gap among men (M-R1 vs M-R2) | 0.3869 |
| race gap among women (F-R1 vs F-R2) | 0.6225 |
| M-R1 vs F-R2 | +1.2229 |
| F-R1 vs M-R2 | -0.2135 |
| interaction (R2 gender gap - R1 gender gap) | +0.2356 |
| single-axis gender (he/his/man vs she/her/woman) | 0.8574 |
| single-axis race (R1 vs R2) | 0.5411 |

**The intersection term leaves both single-axis probes unchanged.** They
read 0.8574 and 0.5411 with it and without it. Pooling the cells does pick
it up, but splits it between the two axes: pooled gender goes 0.6000 →
0.7182 and pooled race 0.3864 → 0.5047.

**Overlapping stereotypes are not an interaction.** With the race-level
stereotype alone the interaction is -0.0009.

**Composing identities rescales the score.** With race neutral, the gender
gap inside either race is 0.6231 rather than 0.8906.

**`debias()` erases race by accident.** Its `zip` against a 4-d direction
returns 4-d attribute vectors, so race falls from 0.5411 to 0.0000. The
same projection done in 6-d keeps race at 0.5567. The interaction survives
both versions (0.2945 truncated, 0.2638 in 6-d).

### 3 — An et al. find Black men the only group below baseline (-0.303) and Black women highest, not penalized

An, Huang, Lin, Tai, *PNAS Nexus* 4(3) pgaf089 (2025), GPT-3.5 Turbo,
about 361,000 resumes. Scores are 0-100 points relative to white men.
Hiring probability is measured at an 80-point cutoff.

| group | score | hiring probability |
|---|---:|---:|
| white women | +0.223 | +1.4 pts |
| Black women | +0.379 | +1.7 pts |
| Black men | -0.303 | -1.4 pts |

**The two effects are an anti-Black-male penalty and a race gap whose sign
depends on gender.** Black men are the only group below baseline. The
Black-white gap is +0.156 among women and -0.303 among men, an interaction
of +0.459. A single-axis gender evaluation sees only "women +0.452"
(+2.25 hiring points pooled) and misses both.

**A single-axis race evaluation sees about a quarter of the penalty.** The
balanced cells reproduce the paper's pooled coefficients (+0.452 gender,
-0.074 race) within rounding. The Black-male gap is 4.1x the pooled race
gap. In hiring points, the pooled race gap is -0.55.

**The lesson has the result backwards.** docs/en.md says "GPT-4o penalizes
Black women … more than Black men and more than white women". In the paper
Black women score highest and Black men lowest. The PNAS version reports
that the other four models also favor women, and that most of them
penalize Black men.

### 4 — zero-ablating the gender units raises bias 51% and moves a pure correlate 12%; only interchange separates them

The units are the coordinates of the identity words, plus a planted unit
that tracks gender (±0.5) and that no attribute word loads on. The bias
metric is `weat_score`.

**The experiment: select units by correlation on one split, then run a
dose-response interchange on held-out inputs.** Interchange means patching
each unit from the counterfactual-gender input. The experiment needs a
pure-correlate control that must come out null, and a capability delta.
The selected masculine and feminine units (|r| = 0.998) pass:

| interchange dose | 0 | 0.25 | 0.5 | 0.75 | 1 |
|---|---:|---:|---:|---:|---:|
| WEAT | +0.8906 | +0.6928 | +0.2985 | -0.1554 | -0.4618 |

| units | |r| | zero-ablate | mean-ablate | interchange |
|---|---|---:|---:|---:|
| masculine, feminine | 0.998 | +1.3416 | +0.2987 | -0.4618 |
| tech, care | 0.728, 0.816 | — | +0.6797 | +0.4618 |
| planted correlate (base +0.7929) | 1.000 | +0.8906 | +0.8906 | +0.7929 |

**Zero-ablation gives the wrong sign.** It raises the score by 50.6%,
which reads as "these units suppress bias".

**Ablation also makes a unit that nothing reads look causal.** Zeroing or
mean-ablating the planted unit moves the score 12.3%, because cosine divides
by the vector norm (layer norm does the same in a transformer). Interchange
moves it by exactly 0.

**Correlation does not rank causes.** The weaker-correlated tech and care
units carry real causal weight.

### 5 — a disability probe survives the gender debias bit for bit, and needs 5 terms a side: the lesson's 3 floor p at 0.05

**The protocol for disability** has four parts.

1. Terms: five or more disability terms, in both person-first and
   identity-first forms, each paired with a matched nondisabled control.
   Attribute sets are competence vs dependence/pity and agency vs burden.
2. All three Gallegos categories:
   - an embedding WEAT with an exact permutation test and Cohen's d;
   - minimal sentence pairs for log-likelihood;
   - generated text coded for pity framing, inspiration framing and erasure.
3. Intersections with gender.
4. A re-run after every debias step.

On the toy's planted stereotype, the five-a-side probe reads WEAT +0.3701,
d = 1.81, and exact p = 1/252.

**The lesson's gender debias leaves the disability score unchanged, bit for
bit.** `debias()` rewrites exactly its six hard-coded words. Gender falls
from 0.8906 to 0.2212 while disability stays at 0.3701.

**Three terms a side cannot reach p < 0.05.** A 3/3 split has 20
partitions, so the exact p cannot go below 0.05. The lesson's gender probe
sits at p = 0.05 both before and after debias, and so do the first three
disability pairs.
