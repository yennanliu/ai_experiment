<!-- generated:start -->
# 15-autonomous-systems / 21-metr-external-evaluation

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/15-autonomous-systems/21-metr-external-evaluation/) · upstream spec
`phases/15-autonomous-systems/21-metr-external-evaluation/docs/en.md`

```bash
uv run demo practice run 21-metr-external-evaluation --ex 1
uv run demo explain 21-metr-external-evaluation --ex 1
uv run pytest demos/phases/15-autonomous-systems/21-metr-external-evaluation
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Confirm the fit's 50% horizon matches the synthetic ground truth. Now hal… | code | T0 | `ex01_the_fit_does_not_match_the_ground_truth_it_was_given.py` |
| 2 | Read METR's Time Horizon 1.1 blog post. Identify the specific tasks where reliability is high… | explain | T0 | prose, below |
| 3 | Read METR's "Measuring Autonomous AI Capabilities" resources. List the HCAST task categories.… | explain | T0 | prose, below |
| 4 | Introduce eval-context gaming into the simulator: flip ~20% of failed tasks to success. Repor… | code | T0 | `ex04_gaming_and_fit_error_are_the_same_size.py` |
| 5 | Design an internal horizon evaluation on your own bug backlog or a representative task set. D… | code | T0 | `ex05_six_hundred_tasks_for_a_number_worth_quoting.py` |
<!-- generated:end -->

## Answers

### 1 — the fit does not match the ground truth it was given

The exercise says to confirm the fit matches. It does not.

| run | 50% horizon | error vs 14.0 hr truth |
|---|---:|---:|
| shipped (seed 3, n=160) | **11.30 hr** | −19.3% |
| halved (n=80) | **9.98 hr** | −28.7% |

So the answer to "does the horizon estimate change meaningfully" is that it
changes by **1.31 hr** — which is *smaller than either error*. The sample size
moves the number by less than the number was already wrong by, which is a
different and more useful answer than "yes" or "no".

**The estimator is noisy and the shipped seed is a low draw.** Over 20 seeds
at n=160 the median is 13.27 hr with a range of **11.06 to 35.49**. The
distribution has a long right tail, and the demonstration's 11.30 sits in its
lower quarter — so a reader who runs the file once sees an unusually low draw
presented as *the* answer, next to a ground truth it does not reach.

**The tail is a slope collapse.** Sorting the 20 fits by |w|:

| slope | horizon |
|---:|---:|
| 0.754 | **35.5 hr** |
| 1.016 | 13.9 hr |
| 1.034 | 11.3 hr |
| … | … |
| 1.561 | 11.1 hr |

`horizon_at` divides by `w`, so an estimator that is uncertain about *whether*
success depends on task length reports a very long horizon rather than a wide
interval. That is the failure mode to watch in any published horizon: a
flat-slope fit and a confident-looking large number are the same object.

**And the 10% horizon is longer than the 50%.** The slope is negative, so the
three bands come out 11.30 / 55.09 / 2.32 hr at 50% / 10% / 90%. The column
labelled 10% is pessimistic in probability and optimistic in hours — a
different axis from the "horizons are upper bounds" framing, and easy to read
past in a three-number row.

### 2 — where reliability is highest and lowest, and why

*Draws on "Why horizons are upper bounds".*

Reliability is highest on the short, well-specified tasks at the bottom of the
curve — SWAA-scale work of a minute to an hour, where the task statement fully
determines the answer and the tooling is exactly what the task needs — and
lowest on the multi-hour ML research-engineering tasks at the top, where the
task admits several approaches and success depends on choices made an hour
before the failure shows up. The lesson's four reasons for horizons being
upper bounds are also the four reasons for the gap, and they get worse
monotonically with task length. **Idealized tooling** matters little on a
one-minute task and enormously on an eight-hour one, because a long task
touches more of the environment. **No real consequences** is irrelevant when
nothing irreversible is possible in the time available. **Eval-context gaming**
has more opportunities to act the longer the model runs. And **no legitimate
user variance** compounds: a short task's ambiguity is resolved once, while a
long task's is resolved repeatedly, each time by a model that cannot ask.

The mechanism underneath all four is the one Lesson 1 measures: per-step
reliability compounds, so a task requiring *n* steps succeeds at *p^n*. The
reliability gap between a 1-minute task and an 8-hour one is not a difference
in difficulty so much as a difference in exponent, and that is why the fitted
curve is a logistic in *log* time rather than in time.

### 3 — the HCAST categories, and which to weight

*Draws on "Benchmark suites".*

HCAST spans **189 tasks across four categories: ML, cybersecurity, software
engineering, and general reasoning**, from one minute to eight-plus hours of
expert time, and it is the suite behind the headline horizon. RE-Bench's 71
ML research-engineering tasks and SWAA's short tasks sit beside it, the first
feeding the AI R&D-4 threshold and the second calibrating the curve's low end.

**For a production task I would weight software engineering most heavily, and
for a specific reason: it is the only category whose failures are
verifiable by the same mechanism that produces them.** A software-engineering
task has a test suite; an agent's success or failure on it is a fact, not a
judgement, so the binary outcome the logistic fit consumes is measured rather
than graded. General reasoning tasks require a grader, and a graded outcome
inherits the grader's variance — which, at the sample sizes exercise 5 shows
are needed (about 640 tasks for a 14% interquartile spread), is not a small
contribution. ML and cybersecurity sit in between: partially verifiable,
partially judged.

The honest caveat is that this weighting is chosen for *measurement quality*
rather than for representativeness, and those pull in opposite directions.
Most production agent work is not software engineering, so up-weighting it
buys a tighter estimate of a narrower thing. That trade is worth making only
because the alternative — a wide estimate of the right thing — turns out, per
exercise 1, to be wide enough to be uninformative.

### 4 — gaming and fit error are the same size

| condition | 50% horizon | fitted slope |
|---|---:|---:|
| clean | 11.30 hr | −1.387 |
| 10% gaming | 13.25 hr | −1.277 |
| **20% gaming** | **14.63 hr** | −1.255 |
| 40% gaming | 44.56 hr | −0.854 |

**The 20% answer is 14.63 hr**, a shift of **+3.33** against the clean fit.
And it is *closer to the ground truth* than the honest evaluation: 0.63 hr
away against 2.70, because the clean fit was already 19.3% low. A gamed
evaluation of this model would look more accurate than a clean one.

**The evaluator cannot tell the two apart from the number.** Fit error spans
11.06–35.49 hr across seeds; gaming spans 13.25–44.56 across rates. Same
magnitudes, same direction. A single horizon figure carries no information
about which produced it, which is the practical form of "horizons are upper
bounds" — not that the number is too high, but that its error bar has two
independent sources and the report shows neither.

**Gaming shows up in the slope before the horizon.** The slope flattens
monotonically as the rate climbs, because flipped failures are drawn
disproportionately from the long end of the task distribution, where the
failures are. The slope is the diagnostic; the horizon is its reciprocal and
loses the sign of the problem. So a published horizon without a slope cannot
be audited for gaming — and the lesson's own `report()` prints three horizons
and no slope.

**At 40% the number stops being an estimate.** 44.56 hr is 3.2× the truth, at
0.62 of the clean slope — the same collapse mechanism a small sample produces.
Which is the argument for publishing `(slope, horizon, n)` rather than a
horizon: the two failure modes look identical in the third number and
different in the first.

### 5 — six hundred tasks for a number worth quoting

**Data collection:** for each backlog item, two values — the median *expert*
completion time in hours, and one bit for whether the agent finished it
unaided. Nothing else enters the fit. Expert time is the expensive field and
has to be a real estimate, not a story-point proxy, because the fit is in log
time and a systematic compression of the range flattens the slope.

**Sample size, from the estimator's own spread:**

| tasks | median estimate | interquartile spread |
|---:|---:|---:|
| 40 | 15.85 hr | 71% |
| 80 | 15.47 hr | 40% |
| 160 | 14.50 hr | 29% |
| 320 | 13.86 hr | 17% |
| **640** | **13.40 hr** | **14%** |

Below 80 tasks the number is not worth quoting. **About 640** is where the
spread stops improving much — and that is the honest cost of the design.

**The binding constraint is the log range, not the count.** The estimator
samples expert times across about three decades, 0.05 to 47.6 hours. A real
backlog clustered in a single decade gives the fit no leverage on the slope,
and the slope is what the horizon divides by. Satisfying *span* from a real
backlog is harder than satisfying *count*, and it is the requirement most
likely to be quietly violated — most teams' tickets are all between one hour
and one day.

**What the output tells you** is a slope and a scale, not a capability. The
fit returns two parameters and the 50% horizon is a reparameterisation of
them, so the internal report should be three numbers — `(slope, horizon, n)` —
of which the published METR figure is one.

**And the comparison to METR is not really a comparison.** The shipped ground
truth is 14.0 hours, matching the January 2026 published figure; a clean fit
on 160 synthetic tasks drawn from that exact truth returns 11.30. Two
evaluations of the *same* underlying capability differ by 19.3% through
sampling alone. So an internal number landing within a few hours of a
published one has agreed with it to within the noise of either — which is
worth saying before anyone treats the difference between 11 and 14 hours as a
finding about their deployment.
