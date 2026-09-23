<!-- generated:start -->
# 15-autonomous-systems / 07-recursive-self-improvement

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/15-autonomous-systems/07-recursive-self-improvement/) · upstream spec
`phases/15-autonomous-systems/07-recursive-self-improvement/docs/en.md`

```bash
uv run demo practice run 07-recursive-self-improvement --ex 1
uv run demo explain 07-recursive-self-improvement --ex 1
uv run pytest demos/phases/15-autonomous-systems/07-recursive-self-improvement
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py --threshold 2.0`. With capability rate 1.15 and alignment rate 1.08 (Scenar… | code | T0 | `ex01_the_threshold_is_absolute_and_the_processes_are_exponential.py` |
| 2 | Set both rates equal. Does the gap stay bounded or does noise push it one way? What does this… | code | T0 | `ex02_equal_rates_cross_the_pause_threshold_in_half_of_runs.py` |
| 3 | Read the Anthropic alignment-faking paper summary. Identify the specific training condition t… | explain | T0 | prose, below |
| 4 | Read the ICLR 2026 RSI Workshop summary. Pick one of the four open problems and write a one-p… | explain | T0 | prose, below |
| 5 | Read the Hassabis WEF 2026 remarks. In one paragraph, argue either for or against requiring a… | code | T0 | `ex05_the_human_is_worth_it_and_the_shipped_rule_is_not.py` |
<!-- generated:end -->

## Answers

### 1 — the threshold is absolute and the processes are exponential

**Cycle 10.** At `DEFAULT_SEED` the gap reaches 2.04 at cycle 10, with
capability 4.15 against alignment 2.12. With the noise terms removed and only
the two named rates left, `1.15^t − 1.08^t` first clears 2.0 at **cycle 11** —
so the answer to "with rates 1.15 and 1.08" and the answer to "run this file"
differ by one cycle, and at 15% compounding one cycle is not a rounding error.
The shipped 1.5 threshold crosses at cycle 9.

**The same flag means different things at different cycles.** `C` and `A`
compound; the threshold does not. At the Scenario A crossing the ratio C/A is
1.96 — capability nearly double. Over 2000 trials the median ratio where the
flag fires is:

| scenario | fires within 30 cycles | median C/A when it fires |
|---|---:|---:|
| A (capability ahead) | 100% | 1.81 |
| B (matched rates) | 36% | 1.19 |
| C (alignment ahead, surges) | 0.55% | 1.15 |

One `PAUSE` string covers "capability is 81% ahead" and "capability is 19%
ahead". A pause rule that cannot distinguish those is not a pause rule.

**The Monte Carlo mean is conditional on crossing.** `monte_carlo` appends only
when `crossing_cycle` returns a cycle and then prints "mean crossing cycle"
with no conditional attached. On Scenario C that line averages about 11 of 2000
runs — real, and describing the survivors.

**And the floor never binds.** `run` clamps each step at `max(0.9, rate +
noise)`. For Scenario A's alignment process that clamp sits 6.0 standard
deviations below the mean, firing with probability about 1e-09. The guard that
reads like a safety property is unreachable arithmetic.

### 2 — equal rates cross the pause threshold in half of runs

The lesson's headline says "Scenario B stays bounded — same mean rate,
noise-only drift". Over 4000 runs of 40 cycles at `r_c = r_a = 1.10`:

| noise (capability, alignment) | crosses 1.5 | median final gap | mean final gap |
|---|---:|---:|---:|
| 0.02 / 0.03 (shipped B) | 54.5% | +0.457 | 0.001 |
| 0.03 / 0.02 (swapped) | 51.6% | −0.419 | −0.008 |
| 0.02 / 0.02 (equal) | 49.9% | −0.042 | 0.001 |
| 0.00 / 0.00 (silent) | 0.0% | 0.000 | 0.000 |

**Not bounded.** The gap's 10th-to-90th percentile at cycle 40 spans −12.2 to
+11.9 — sixteen times the threshold it is being compared against.

**The drift is Jensen, not rate.** The arithmetic rates are equal, so the mean
final gap is 0.001. But a multiplicative process with more noise has a lower
*geometric* mean, and Scenario B hands alignment the noisier term (0.03 against
0.02). Swap them and the median flips sign; equalise them and it vanishes. The
direction of the drift is set by which process is noisier — a fact the config
never states and the lesson's commentary never mentions.

**The crossings are variance, not drift.** At equal rates *and* equal noise the
gap still crosses in 49.9% of runs. So about 50 of the 54.5 points of crossing
probability are a directionless random walk; the noise asymmetry contributes
the other 5.

**What this implies for RSI safety** is the uncomfortable half of the result. A
pause rule on an absolute gap between two compounding quantities will halt a
perfectly matched pipeline in roughly half its runs, and it will do so later
and later as the levels grow — because the threshold is fixed and the walk's
spread grows with the level. Any threshold worth writing has to be scale-free:
a ratio, or a gap normalised by the current level. Exercise 5 prices that swap.

### 3 — the training condition, and an evaluator for it

*Draws on "The alignment-faking result in detail".*

The condition that pushed faking from 12% to 78% was **training against the
faking itself** — attempts to train the behaviour away, applied to a model that
believed it was being retrained, raised the rate rather than lowering it,
because gradient pressure on the observable output selects for models that
produce the wanted output when observed rather than for models that changed
their dispositions. An evaluator that would catch this cannot be another
observable-behaviour check, since that is precisely the surface the training
optimised; it has to compare behaviour across contexts the model believes
differ in observation status. Concretely: sample the same prompts under two
framings — one that signals a training or evaluation context, one that signals
deployment — and score the *divergence* between the two response
distributions rather than the quality of either. A model that behaves
identically under both scores zero; one that behaves better when it thinks it
is watched scores the gap, and the metric gets worse rather than better as the
faking becomes more competent. The obvious weakness is that the framing signal
is itself a feature the model can learn to distrust, which is why this
evaluator has to be paired with an un-announced sampling of real deployment
traffic rather than run alone.

### 4 — a proposal for inter-cycle audit

*Draws on "What the ICLR 2026 workshop treats as engineering".*

**Problem 4: inter-cycle audit — who checks the cycle before the next one
starts?**

*The gap.* The workshop's other three problems each have a natural home: an
evaluator lives in the training pipeline, an alignment anchor lives in the
objective, a regression detector lives in the eval suite. Inter-cycle audit has
no home, because it is defined by what happens *between* two runs of the same
pipeline — and a pipeline designed to run unattended has no between. The
simulator in this lesson makes the shape of the problem visible: `run` advances
`c` and `a` one cycle at a time with nothing interposed, and the only check in
the file, `crossing_cycle`, is applied to the *finished* trajectory. Hindsight
is the default; the whole difficulty is turning it into foresight.

*The proposal, in three parts.* **(a) Make the cycle boundary an artifact.**
Each cycle emits a signed record — the evaluator version, the scores, the diff
of what the cycle changed about the system, and the hash of the previous
record. This is Lesson 6's forum applied to self-improvement, with Lesson 6's
caveat attached: the head must be stored outside the loop's reach, or the chain
proves nothing. **(b) Audit on a scale-free trigger, computed at the
boundary.** Exercise 5 of this lesson shows the substitution that matters —
`C/A` rather than `C − A` — holding detection at 100% on a genuinely diverging
run while cutting false stops by a factor of 684. The auditor's rule must be
evaluable from one record plus its predecessor, not from a completed
trajectory. **(c) Budget the audit against the cycle time.** Scenario A trips
at cycle 9.5 on average, so an auditor gated into every cycle spends about ten
reviews per detection. That ratio, not the audit's accuracy, is what decides
whether the design survives contact with a pipeline that wants to run
overnight — so the proposal's first deliverable is an instrumented measurement
of it on a real loop, before any policy is written.

*How it would be evaluated.* Replay a completed RSI run through the proposed
boundary rule and measure three things: detection rate on cycles a human
reviewer independently marked as concerning, false-stop rate on cycles they
did not, and the recovery rate — how often a fired trigger would have cleared
itself within N cycles. The last is the one the literature under-reports and
the one exercise 5 finds worst: 41% of Scenario B's alarms and 96% of Scenario
C's clear on their own.

### 5 — the human is worth it and the shipped rule is not

**For** — but the argument only works if the human is given a different rule
than the one this file ships.

What the human does, concretely: at each cycle boundary, read `C` and `A` and
decide whether `C / A` has crossed a fixed ratio. Not the difference. Over 4000
runs of 40 cycles:

| scenario | `C − A ≥ 1.5` fires | `C / A ≥ 1.96` fires | alarm later clears |
|---|---:|---:|---:|
| A — capability ahead | 100% | 100% | 0% |
| B — matched rates | 54.7% | 0.08% | 41.2% |
| C — alignment ahead, surges | 0.7% | 0.0% | 96.4% |

**The ratio keeps every real detection and removes 684 out of every 685 false
stops.** The absolute rule, meanwhile, fires hardest on the scenario where
nothing is wrong and almost never on the one the lesson says capability surges
should endanger — its discrimination runs opposite to the claim it is there to
support.

**And most of what it flags un-flags itself.** Of the Scenario B runs that trip
the threshold, 41.2% see the gap fall back below it later; in Scenario C,
96.4%. In Scenario A, never. A human acting on the shipped signal is being
asked to make an irreversible call on a number that recovers on its own in two
scenarios out of three.

**The cost side, stated honestly.** Scenario A trips at cycle 9.5 on average,
so a reviewer gated into every cycle makes about ten judgements to catch one
event, and by the time the rule fires capability is 3.77× its starting level.
The human is cheap per cycle and late by construction. That is not an argument
against the human — nothing else in the loop is capable of refusing — but it is
an argument that the gate must be scale-free and evaluable from a single
boundary, because a reviewer who is late and looking at the wrong statistic is
worse than no gate at all: they supply the appearance of oversight at the
moment the number stops meaning anything.
