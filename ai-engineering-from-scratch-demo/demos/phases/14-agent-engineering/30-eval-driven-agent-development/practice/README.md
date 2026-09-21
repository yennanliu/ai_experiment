<!-- generated:start -->
# 14-agent-engineering / 30-eval-driven-agent-development

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/30-eval-driven-agent-development/) · upstream spec
`phases/14-agent-engineering/30-eval-driven-agent-development/docs/en.md`

```bash
uv run demo practice run 30-eval-driven-agent-development --ex 1
uv run demo explain 30-eval-driven-agent-development --ex 1
uv run pytest demos/phases/14-agent-engineering/30-eval-driven-agent-development
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Take one of your production failures. Write an eval case that reproduces it. Does your agent… | code | T0 | `ex01_the_flaky_case_passes_on_round_two_then_on_round_one.py` |
| 2 | Build an LLM-judge rubric for your domain with three dimensions (factual, tone, scope). Score… | code | T0 | `ex02_a_blended_score_gives_the_proposer_nothing_to_fix.py` |
| 3 | Wire the eval suite into CI. Fail the build on >=5% regression. | code | T0 | `ex03_a_five_percent_threshold_needs_twenty_cases_to_exist.py` |
| 4 | Add a trajectory-efficiency metric: how many steps did the agent take vs a gold trajectory? | code | T0 | `ex04_the_eval_case_records_rounds_and_the_trajectory_is_steps.py` |
| 5 | Map every Phase 14 lesson to an eval case in your suite. Any missing? That's a gap to close. | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

### 1 — three production failures, reproduced, and the agent fails all three

A production failure worth an eval case is one found by measurement rather than
by guessing, so the three reproduced here are defects this phase actually
measured: Lesson 29's queue worker consulting its failure policy twice (108 DLQ
entries where 1 was expected), Lesson 26's context-loss detector extracting the
verb after `"do not"` (0 hits on 100 traces), and Lesson 27's validator refusing
`"do you have the invoice"` (33 of 60 legitimate calls). Each becomes an
`EvalCase`; each fails, running to `max_rounds` because the proposer's output does
not change; each flips to passing when the defect is fixed. **Does the agent pass
them now? No — 3 of 3.** A suite that does not fail before the fix is not evidence
of anything, so that is the property to check first.

Writing them exposes three things about the harness.

The shipped suite's own flaky case is not reproducible. `_flaky_benchmark_case`
keeps its attempt counter in a closure, so running `evaluator_optimizer` on the
*same case object* twice gives `rounds=2` then `rounds=1`. The verdict is stable
and the round count is not, so any metric built on rounds — convergence speed,
cost per pass — moves without the agent changing. The lesson's own "flaky evals"
pitfall is in its fixture, and the fix is to build the case from a factory per run
(or to pin the state), which is what "snapshot state" means concretely.

One of the four shipped proposers takes `feedback` and never reads it, so its two
rounds measure a counter rather than a refinement — and `CaseResult`'s six fields
distinguish "refined to a pass" from "passed on a retry" nowhere. Those are
different systems and they report the same number.

And the loop shows only the last candidate: `evaluator_optimizer` overwrites
`feedback` each round and returns `final=candidate`, so a three-round failure
reports 1 of the 3 attempts it made. Reproducing a failure is step one;
*debugging* it needs the intermediate candidates, and the result type has nowhere
to put them.

### 2 — the rubric matters less than what it hands back

Three dimensions — factual, tone, scope — at a third each, requiring all three,
over 50 sessions. The interesting variable is not the scoring, it is the `reason`
string, because `evaluator_optimizer` feeds it to the proposer as `feedback`.

A judge that names the failing dimension converges **50 of 50** in a mean of 1.8
rounds. The same rubric returning only `"score 0.67"` converges **10** — exactly
the sessions that were already correct — and the other 40 exhaust their rounds
without improving. Identical verdicts, identical scores, and the pass rate moves
from 20.0% to 100.0% on the wording alone.

That is worth stating as a design rule rather than a curiosity: in an
evaluator-optimizer loop the judge's prose is an *input to the agent*, not a
report for a human. A numeric score is the correct output for a dashboard and the
wrong output for a refinement loop, and most LLM-judge tooling defaults to the
number.

Two supporting findings. A session can fail one dimension outright and still score
0.67, so a threshold on the mean admits 20 of 50 — requiring every dimension is
what leaves 10 passing before refinement, and 10 is also exactly where the blended
judge ends up. And the round cap is invisible when it binds: dropping `max_rounds`
from 3 to 1 takes the naming judge from 50/50 to 10/50, with `CaseResult` carrying
no field that says "budget exhausted". "The agent got it wrong" and "the agent ran
out of rounds" read identically, and they call for opposite responses.

### 3 — a five percent threshold needs twenty cases before it exists

`ci_gate` ships and takes the threshold as an argument, so wiring is one call. The
arithmetic is where it goes wrong.

Over the lesson's four cases the pass rate takes five possible values and the
smallest non-zero regression is **25.0%**. Every threshold between 0% and 25%
behaves identically, so "fail on ≥5%" and "fail on ≥20%" are the same gate.
Twenty cases make a 5.0% step reachable — one case — and only then does the
stated rule mean anything. That is the first thing to check when adopting a
percentage gate: the suite has to be at least `1/threshold` cases for the
threshold to be expressible at all.

The operator is also wrong for the stated rule. `ci_gate` compares
`regression > regression_threshold`, so a regression exactly at the threshold is
allowed where "≥5%" would block it. Shown at 25%, where the arithmetic is
binary-exact: one failure in four against a baseline of 1.0 returns
`allow=True`. At 5% there is a second, quieter version of the same problem —
`0.95 - 18/20` evaluates to `0.049999999999999934`, so a boundary regression
passes for floating-point reasons even under `>=`. Compare integer case counts,
not float rates.

Two more gaps. The baseline is an argument and nothing stores one: `ci_gate` has
three parameters, none naming a path or a store, and the module defines no
function that reads or writes a baseline — `main` passes the literal `0.95`.
Running the suite twice cannot tell you whether anything moved, which is the
lesson's own "no baseline" pitfall present in the reference. And every case weighs
the same, so on 39 benchmark cases plus 1 online guardrail, the guardrail failing
alone is a 2.5% regression and the build is allowed; weighting online cases at 3x
makes the same failure 7.1% and blocks it. The three layers the lesson carefully
separates are flattened into one denominator by the gate.

### 4 — `rounds` counts loop iterations, and the trajectory is somewhere else

`CaseResult` records `rounds`, which counts proposer-judge cycles. A case can pass
in one round after the agent took forty actions. So the metric has to come from a
trajectory, and Lesson 20's harness already has one: `Task.gold_steps` plus a
trace.

Wiring it in gives 14 steps against 12 gold across Lesson 20's three tasks —
`buy_headphones` 1.00x, `buy_bundle` 1.00x, `revised_order` 1.40x, aggregating to
1.17x. And all three eval cases pass in `[1, 1, 1]` rounds. The round counts are
identical while the step counts span 2.33x, so the two are uncorrelated by
construction and `CaseResult`'s six fields include zero step counts. An efficiency
claim built on `rounds` is reading the wrong number.

The two gates also disagree about which task is worst: a pass-rate gate sees 3/3,
a 1.2x efficiency gate fails `revised_order` alone. Both are needed, and gating on
efficiency alone is the worse mistake — an empty trajectory scores 0.00x, better
than gold, which is Lesson 20's own finding arriving in the eval harness.

There is a typing problem underneath. `EvalCase.judge` is
`Callable[[str], tuple[bool, str]]`, so a trajectory-based case either serialises
the step count into prose for the judge to parse back out, or closes over the
trace and stops being serialisable. Neither is where a gold trajectory belongs.
The honest fix is a field on `CaseResult` — `steps` and `gold_steps` — and a
judge signature that takes the trajectory rather than a rendering of it.

### 5 — mapping Phase 14 to eval cases, and the gaps the map does not show

**Tying Phase 14 together** gives the table: every lesson generates eval cases,
and "if your eval suite has cases for each, you have covered Phase 14." Working
through it against what this phase actually measured, the mapping mostly holds and
the interesting part is where it does not.

**The cases the table names, and what each is really testing.** Lesson 01's
"budget-exhausted, infinite-loop guard" is the shape exercise 2 found missing from
`CaseResult` — a case that hits `max_turns` and one that hits a cycle detector
have to be distinguishable, or the suite reports one number for two bugs. Lesson
19–20's "SWE-bench Verified score, WebArena success rate, OSWorld efficiency" is
exercise 4, and the efficiency half needs a result field that does not exist.
Lesson 29's "DLQ handles N% failure" is exercise 1's first case, and writing it
found the double-draw bug rather than confirming the DLQ — which is the argument
for the whole practice in one example.

**Three lessons in the table generate cases the harness cannot express.**

*Lesson 13, "resume reproduces state exactly."* This is an equality assertion over
a state object, and `judge` takes a `str`. Serialising state into a candidate
string to compare it is the same problem exercise 4 hit with trajectories, and it
is worse here because state equality is the whole test.

*Lesson 23, "spans emit required attributes."* The assertion is over a span tree,
and Lesson 23's own finding was that `Span` has no ids — so the case has to
compare an unordered set of attribute keys, which a `(bool, str)` judge can do but
a `str` candidate cannot carry faithfully.

*Lesson 26, "detectors tag known failures."* A detector returns a set of labels;
the eval needs precision and recall against hand labels, not a boolean. Lesson 26
measured a detector scoring 0 on 100 traces containing 14 real violations — a
pass/fail case would have reported "the detector ran".

**Four lessons the table omits, and they are the expensive ones.**

*Lesson 24's platform layer.* The table has no case for "the observability we
depend on still works". Lesson 24's finding was that flipping `capture_inline`
changes every content attribute name — a config change that silently empties a
dashboard, with no test that would catch it.

*Lesson 25's debate.* Absent from the table entirely, and Lesson 25 found debate
accuracy non-monotone in the number of correct agents (0, 8, 0, 8 as experts go 1
to 4). That is exactly the behaviour a regression suite exists to pin.

*Lesson 21–22's latency budgets.* The table covers computer-use safety and not
voice. Lesson 22's cascade sits at 400–990ms against a 450–600ms premium band, and
nothing in this eval harness measures time — `CaseResult` has no duration.

*Cost.* No lesson's row mentions tokens or ops, and Lessons 25 and 28 both found
the cost ranking disagreeing with the accuracy ranking. A suite that gates on pass
rate alone will accept a 2.5x cost increase silently.

**The structural gap the table cannot show.** Every row is a case that *passes or
fails*. The findings in this phase that mattered most were measurements that
neither passed nor failed: 55% validator false positives, 1.17x over gold, a 44%
CI flap rate. Those need a baseline and a trend, and exercise 3 found the harness
has neither — the baseline is a literal argument and nothing stores one. Covering
Phase 14 means the suite has the cases *and* keeps last week's numbers; the second
half is the one the table does not ask for.
