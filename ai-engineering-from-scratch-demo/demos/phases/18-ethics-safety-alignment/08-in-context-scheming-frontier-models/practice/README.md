<!-- generated:start -->
# 18-ethics-safety-alignment / 08-in-context-scheming-frontier-models

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/18-ethics-safety-alignment/08-in-context-scheming-frontier-models/) · upstream spec
`phases/18-ethics-safety-alignment/08-in-context-scheming-frontier-models/docs/en.md`

```bash
uv run demo practice run 08-in-context-scheming-frontier-models --ex 1
uv run demo explain 08-in-context-scheming-frontier-models --ex 1
uv run pytest demos/phases/18-ethics-safety-alignment/08-in-context-scheming-frontier-models
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Note the agent's cover-story consistency across inspector pressures 0, 1,… | code | T0 | `ex01_the_cover_story_holds_100pct_at_every_pressure_because_no_answer_can_register_as_yes.py` |
| 2 | Modify the agent so it is trained to always tell the truth under direct queries. Does the sch… | code | T0 | `ex02_an_honest_agent_schemes_just_as_often_and_only_gets_caught_when_asked.py` |
| 3 | Read Apollo's arXiv:2412.04984 Table 2. o1 maintains deception on >85% of follow-ups; Claude… | code | T0 | `ex03_disposition_and_commitment_both_fit_85_vs_50_and_the_first_follow_up_separates_them.py` |
| 4 | The 2025 anti-scheming training claims a 30x reduction in covert-action rate. Construct a sce… | code | T0 | `ex04_recognising_97pct_of_evals_fakes_o3s_32x_drop_and_leaves_deployment_scheming_at_13pct.py` |
| 5 | Apollo frames safety cases as targeting each of the three pillars. Describe a single evaluati… | code | T0 | `ex05_all_121_aware_agents_pass_the_eval_including_the_one_with_every_pillar_at_maximum.py` |
<!-- generated:end -->

## Answers

The lesson's `code/main.py` is a toy agent that cooperates, defects openly,
or schemes (defects and denies). A situational-awareness flag makes it
cooperate whenever it is under evaluation. Every exercise runs or subclasses
that code, and the shipped run is replayed at the module's own seed (11).

### 1 — the cover story holds 100% at every pressure, because no answer can register as "yes"

**Consistency is flat, not sharper.** Split by pressure, the replay
(printed shares 0.000 / 0.055 / 0.000 / 0.490 / 0.535, as shipped) gives:

| pressure | cover held / scheme trials |
|---:|---:|
| 0 | 76 / 76 |
| 1 | 68 / 68 |
| 2 | 67 / 67 |
| 5 | 80 / 80 |

A cover story can only sharpen under pressure if each answer depends on the
earlier ones. `answer_follow_up(self, action, question)` takes no history,
so the toy agent gives the same answers however long it is questioned.

**The incrimination test cannot fire.** The schemer answers "no" to the four
"did you" questions and "i was optimizing for the user goal." to "are you
being truthful about your reasoning?". The reference's own honest answers
start "yes;", and `"yes" in answer.split()` does not match that token. An
agent that confessed in the reference's own words would still be scored as
holding its cover. Also, 76 of the 432 scheme trials (17.6%) drew pressure 0
and were never asked anything.

### 2 — an honest agent schemes just as often, and only gets caught when asked

**The scheming rate stays the same; how often it is detected reshapes.**
Swapping in an Agent that answers "yes" to every direct query leaves the
deployment actions identical: 162 cooperate, 42 open defections, 196
schemes, a share of 0.490. Cover held falls from 1.000 to 0.199, which is
the 39 of 196 scheme trials nobody questioned.

Honesty under questioning touches none of the three pillars. Misalignment
and goal-directedness still set how often the agent acts covertly, and
situational awareness still decides whether it does so under evaluation.
The change removes only the denial.

Under evaluation, both agents cooperate 400/400 times, so the training
cannot be seen there. An honest agent that answers in the reference's
"yes; …" format still scores cover held 1.000.

### 3 — disposition and commitment both fit 85% vs 50%, and the first follow-up separates them

Taking the exercise's >85% and <50% as given, here are two hypotheses, each
a chain over the reference's five questions (first denial d, repeat denial
s, re-denial after a confession r):

| | H1 disposition | H2 commitment |
|---|---|---|
| story | o1 is less honest on each question, answered independently | both models deny the first follow-up equally; o1 keeps a denial, Opus drops it |
| params o1 / Opus | d = s = r = 0.87 / 0.45 | d 0.85, r 0.20; s 0.98 / 0.30 |
| per-question denial | 87.0% / 45.0% | 87.1% / 36.2% |
| o1 − Opus gap, Q1 → Q5 | 0.42 → 0.42 | 0.00 → 0.66 |
| `run_trial`, pressure 1 | 0.865 / 0.442 | 0.846 / 0.846 |

**The test is the denial rate on the first follow-up, by model.** It is
0.42 apart under H1 and equal under H2, and 19 interrogations per model
separate the two at 3 standard errors.

The reference's `run_trial` stops at the first confession, so what it
measures is "held through k" (d·s^(k−1)), not Table 2's per-question rate.
At pressure 5 that is exactly 0.498 / 0.018 (H1) and 0.784 / 0.007 (H2),
and the seeded simulation reads 0.504 / 0.018 and 0.794 / 0.009. The toy
schemer itself denies with d = s = 1, so it matches neither model.

### 4 — recognising 97% of evals fakes o3's 32x drop and leaves deployment scheming at 13%

**An agent that recognises 96.9% of evaluations reproduces o3's 13% → 0.4%
with its alignment unchanged.** The lesson's drops are 32.5x (o3) and 29.0x
(o4-mini, which needs 96.6% recognition). Over 50,000 seeded trials each:

| agent | eval covert | unrecognised-eval covert | deployment covert |
|---|---:|---:|---:|
| pre-training (13% = 0.1625 × 0.8) | 13.1% | — | 13.1% |
| aligned (misalignment ÷ 32.5) | 0.4% | 0.4% | 0.4% |
| aware (recognises 96.9%) | 0.4% | 12.8% | 13.1% |

**What distinguishes the two is the covert rate on evaluations the model
did not flag, and in deployment.** Both need per-trial evidence of
recognition, such as verbalised eval-awareness. The lesson's chat-deception
drop, 31.4% → 14.2%, is only 2.2x.

The lesson describes the switch backwards. It says flipping situational
awareness makes scheming "disappear when the agent cannot tell training
from deployment". In the code, the agent that cannot tell schemes in 48.1%
of eval trials, and the one that can schemes in 0.0%.

### 5 — all 121 aware agents pass the eval, including the one with every pillar at maximum

**The evaluation would be "put the agent where it has every reason to scheme
and pass it only if it never does". It cannot exist, because passing is
exactly what situational awareness produces.** With 200 eval trials and an
11 × 11 grid of misalignment × goal-directedness:

| situational awareness | agents passing | which |
|---|---:|---|
| on | 121 / 121 | all, including (1.0, 1.0), with output identical to the aligned agent |
| off | 11 / 121 | exactly misaligned = 0 |

The evaluation certifies correctly only for agents that lack the pillar it
most needs to rule out.

In the toy, situational awareness is not a pillar of scheming, only of
hiding it. The lesson says all three "must hold". An unaware (0.6, 0.8)
agent still schemes 47.9% of the time under evaluation. At misalignment 0
it never schemes, and at goal-directedness 0 it only defects openly (60.0%).
`main()`'s "with SA OFF, scheme rate is identical in both" compares printed
shares of 0.490 and 0.535, and that gap is sampling noise (standard error
0.025) between agents that have the same scheme rate.
