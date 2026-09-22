<!-- generated:start -->
# 15-autonomous-systems / 01-long-horizon-agents

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/15-autonomous-systems/01-long-horizon-agents/) · upstream spec
`phases/15-autonomous-systems/01-long-horizon-agents/docs/en.md`

```bash
uv run demo practice run 01-long-horizon-agents --ex 1
uv run demo explain 01-long-horizon-agents --ex 1
uv run pytest demos/phases/15-autonomous-systems/01-long-horizon-agents
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run the simulator. With the default 7-month doubling, how many months until the horizon cross… | code | T0 | `ex01_the_thirty_hour_crossing_lands_between_two_rows.py` |
| 2 | Set per-step reliability to 0.995. What trajectory length still clears 50% end-to-end reliabi… | code | T0 | `ex02_halving_the_failure_rate_doubles_the_trajectory.py` |
| 3 | Read METR's Time Horizon 1.1 blog post. Identify one methodological choice (task weighting, e… | code | T0 | `ex03_the_success_criterion_has_no_slot_in_the_config.py` |
| 4 | Pick one production agent workflow you know. Estimate the median trajectory length in tool ca… | code | T0 | `ex04_the_skill_fails_its_own_reliability_budget.py` |
| 5 | Read the 2026 International AI Safety Report section on eval-context gaming. Design one evalu… | code | T0 | `ex05_the_gap_is_shaved_off_the_half_that_does_not_compound.py` |
<!-- generated:end -->

## Answers

### 1 — the thirty-hour crossing lands between two rows

The two crossings are **month 7.70** for 30 hours and **month 25.09** for a
168-hour week, against the shipped baseline of 14.0 hr and a 7-month doubling.
That is 1.10 and 3.58 doublings. Plotted on one 36-month axis:

```text
........^................^...........
|     |     |     |     |     |     |
0     6     12    18    24    30    36
```

"Plot" is an ASCII axis on purpose. The lesson ships `assets/horizon-curve.svg`
and its `code/` draws nothing, so the only plot a reader can produce from the
module is one the module prints.

Three things the arithmetic says that the table does not.

**The lesson prints one of the two answers.** `horizon_projection` loops over
four crossing targets — 24, 48, 168 and 720 hours — so the week-scale number is
already on screen at month 25.1 and the 30-hour one is not. The exercise asks
for the crossing the table skips.

**The projection grid straddles the crossing it was asked about.** The table
steps six months at a time: month 6 reads 25.4 hr, month 12 reads 45.9 hr.
Zero of its seven rows lands within an hour of 30, so a reader who trusts the
table watches the horizon pass 30 hours inside a gap it never prints. The
granularity is wrong for the question by a factor of about five.

**The distance between the crossings is baseline-independent.** At baselines of
7, 14 and 28 hours the two crossings all move, and the gap is 17.40 months in
every case, because it is `7 · log₂(168 / 30)` and the baseline cancels. A
better baseline estimate buys a better answer to *when*, never to *how far
apart* — which is the number a roadmap actually spends.

### 2 — halving the failure rate doubles the trajectory

At 0.995 per-step, **138 steps** still clear 50% end-to-end, against **68** at
0.99 and **692** at 0.999. This is not a number the module withholds: its "max
trajectory length for 50% end-to-end success" block already prints all three.
The exercise is asking what they mean side by side.

| per-step | 50% length | `ln 2 / (1 − p)` | 95% length |
|---|---:|---:|---:|
| 0.99 | 68 | 69.3 | 5 |
| 0.995 | 138 | 138.6 | 10 |
| 0.999 | 692 | 693.1 | 51 |

**The law is `ln 2 / (1 − p)`, accurate to 1.31 steps across all three.** So the
allowed length is a straight proportion of the *failure* rate, and "per-step
reliability" is the wrong axis to plan on. On the reliability axis 0.99 → 0.999
looks like a rounding change; on the failure axis it is a tenfold budget
increase, and the trajectory grows tenfold with it.

**50% is the lesson's own bottom flag.** `reliability_compounding` labels
anything under 0.5 "coin flip or worse" and reserves "ok" for 0.95 and up. The
lengths that clear *that* bar are 5, 10 and 51 steps — a constant 13.5× shorter,
since the ratio is `ln 0.5 / ln 0.95` and the per-step rate cancels. The
exercise's bar is the line below which the lesson itself refuses to call
anything shippable, so "138 steps" is the answer to a question no operator
should be asking.

**And the shipped table's only 0.995 row sits 62 steps past the answer.** One of
its eight cases uses 0.995, at 200 steps, scoring 36.7% and flagged "coin flip
or worse". The guard is one-sided too: `per_step >= 1.0` returns the sentinel
1000000000, while `per_step = 0.0` raises `ValueError` out of `math.log` rather
than returning 0.

### 3 — the success criterion has no slot in the config

The methodological choice I would change is the **success criterion**: report
the 80% horizon beside the 50% one, and lead with the 80% number in any
deployment argument. The 50% mark is the median of a logistic fit — the task
length at which the model fails as often as it succeeds — and no one ships at
the median. A capability claim quoted at the median is being quoted at the point
of maximum ambiguity: it is the one threshold where the answer to "will it
finish" is definitionally "no information". The 80% horizon is a worse headline
and a usable input, because it is the first point on the curve where a single
unattended run is more likely to be worth starting than not, and the gap between
the two is itself the most informative thing the fit produces — a shallow slope
means the median is nearly meaningless as a planning number, and a steep one
means it is nearly sufficient. Reporting only the median hides the slope, which
is the parameter a deployment decision actually turns on.

Three measurements underneath that paragraph.

**The criterion is stated everywhere and parameterised nowhere.** The lesson
writes "50%" eight times and "80%" zero times, and `HorizonConfig`'s three
fields — baseline hours, baseline month, doubling months — hold no success
criterion at all.

**One of the five numeric functions can be handed a success bar, and it is on
the wrong half.** `max_steps_for_target` takes `target`; `horizon_at`,
`months_to_cross`, `end_to_end_reliability` and `fmt_hours` do not. So the
simulator can price a 95% *trajectory* and cannot be asked about an 80%
*horizon*.

**The change costs a year and a bit, in the lesson's own unit.** If the stricter
horizon sits a factor k below the median one, today's strict figure is the
median figure from `7 · log₂(k)` months ago: 7.0, 14.0, 16.25 and 21.0 months of
lag for k of 2, 4, 5 and 8. That is one call to `months_to_cross` — and it is a
call the lesson never makes, because all four of its shipped targets sit above
the 14-hour baseline and return positive months. The negative-month branch is
exactly the "how far behind is the strict criterion" question, and zero of the
shipped calls asks it.

### 4 — the skill fails the reliability budget it sells

"A production agent workflow you know" invites a number from memory, and a
number from memory cannot be checked. The one production workflow whose step
list can be *read* is the one this lesson ships for production use:
`outputs/skill-horizon-reality-check.md`, which takes a proposed autonomous task
and returns a go / hold / no-go memo.

Counting its obligations gives the trajectory floor: 5 produce steps, 3 hard
rejects, 3 refusal rules, 6 memo fields — **17**. Run through the lesson's own
`end_to_end_reliability` at the three per-step rates the skill's own step 3
prescribes:

| per-step | end-to-end over 17 steps | the lesson's flag |
|---|---:|---|
| 0.95 | 41.8% | coin flip or worse |
| 0.99 | 84.3% | fragile |
| 0.995 | 91.8% | fragile |

Zero of three reach "ok". Is that honest with the users? No, and the dishonesty
is structural rather than numerical:

**The memo reports every reliability number except its own.** Its six output
fields carry an end-to-end table for the user's task and an explicit verdict;
none carries the reliability of the verdict. All four uses of the word
"reliability" in the skill are about the thing being judged.

**No trajectory at the skill's own rates clears the skill's own bar.** A 95%
end-to-end — the lesson's "ok" flag — allows 1, 5 and 10 steps at 0.95, 0.99 and
0.995. The skill is 17 at every rate it tells the user to tabulate.

**All six of its gates point outward.** Three hard rejects and three refusal
rules, each a condition on the user's task: horizon ratio, budget, irreversible
actions. Zero mention the memo. A workflow whose second hard reject is "any
claim that a model *can do* a task based on the METR horizon alone" ships a
verdict with no reliability argument of its own.

The fix is one line in the output format — the memo states its own step count
and the end-to-end figure that follows from it — and it is the line that would
make a reader discount the verdict correctly.

### 5 — the gap is shaved off the half that does not compound

The protocol, in five rules:

1. **State the horizon as a ceiling.** Benchmark numbers assume ideal tooling
   and no consequences; they bound what is possible, never what is likely.
2. **Measure the gap, do not assume it.** The eval-vs-deploy delta is a
   property of your distribution, recovered by replaying deployment traces
   through the eval harness and comparing, not a constant.
3. **Apply it per step, where it compounds.** A model that behaves differently
   on a fraction of steps changes the per-step rate, and the per-step rate is
   raised to the trajectory length.
4. **Hide the eval context from the model.** Same tools, same latency, same
   logging, same consequences; an eval that is distinguishable is measuring the
   distinguishable behaviour.
5. **Report a distribution, not a scalar.** One number cannot carry a gap that
   is itself uncertain.

Rule 3 is where the arithmetic bites. Take the lesson's own 68-step trajectory
at 0.99 per-step — 50.5% end-to-end. Let the model run at 0.90 on the fraction
of steps it judges unobserved, at Anthropic's measured 12% and 78% rates:

| gaming rate | effective per-step | end-to-end over 68 steps |
|---|---:|---:|
| 0% | 0.99 | 50.5% |
| 12% | 0.9792 | 23.9% |
| 78% | 0.9198 | 0.3% |

The shipped correction, over the same phenomenon, turns a 14-hour horizon into
11.2 and 8.4 hours — factors of 0.8 and 0.6 where the trajectory arithmetic says
0.47 and 0.007. Two orders of magnitude, and the direction of the error is
reassuring rather than alarming.

**The correction lands on the half that cannot compound.** `deploy_gap_note`
calls exactly two names, `fmt_hours` and `print`. It never reaches
`end_to_end_reliability` or `max_steps_for_target`, so the lesson's two halves —
the horizon that grows and the reliability that decays — are joined in the prose
and nowhere in the code.

**A third of the adjustment table adjusts nothing.** Three horizons crossed with
three gaps is nine rows, and one of the three gaps is 0.0, so three rows print
the benchmark number unchanged under a "deploy" heading. All six values are
literals in the function body.

**Of the five rules, the artifact can express one** — rule 1, which is in the
printed text. The other four need an input the module has no seam for: its four
demo functions take zero parameters and read zero files. That is the honest
scope of this exercise. The protocol is a design; the lesson's simulator is not
the place it would live, and saying so is more useful than pretending a
stdlib projection could implement it.
