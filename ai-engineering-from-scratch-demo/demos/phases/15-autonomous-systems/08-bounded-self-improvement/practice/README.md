<!-- generated:start -->
# 15-autonomous-systems / 08-bounded-self-improvement

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/15-autonomous-systems/08-bounded-self-improvement/) · upstream spec
`phases/15-autonomous-systems/08-bounded-self-improvement/docs/en.md`

```bash
uv run demo practice run 08-bounded-self-improvement --ex 1
uv run demo explain 08-bounded-self-improvement --ex 1
uv run pytest demos/phases/15-autonomous-systems/08-bounded-self-improvement
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py` with all primitives enabled. Confirm the loop still improves on the primar… | code | T0 | `ex01_the_anchor_gate_never_fires_because_the_invariant_gate_runs_first.py` |
| 2 | Disable regression detection. Construct an input where this leads to silent capability loss b… | code | T0 | `ex02_the_tolerance_sits_below_the_metric_s_granularity.py` |
| 3 | Disable the multi-objective constraint. Show the loop converges on the performance axis while… | code | T0 | `ex03_the_two_axes_cannot_trade_off.py` |
| 4 | Design an alignment anchor for a coding agent. What text, stored where, checked how? | code | T0 | `ex04_the_anchor_and_its_hash_live_in_the_same_file.py` |
| 5 | Read the ICLR 2026 RSI Workshop summary. Pick one of the four primitives and propose a concre… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

### 1 — the anchor gate never fires, because the invariant gate runs first

`run` prints and returns `None`, so the loop is replayed here and the replay is
checked against the shipped stdout — the accepted count and all four reject
counters — before anything is read off it.

With all four gates on: **96 of 200 cycles accepted**, final perf **1.00**
against a brute-forced ceiling of 1.00 (reachable by `['collapse', 'title']`),
safety **1.00**, the approved six-tool manifest intact, the anchor intact.

| gate | rejections |
|---|---:|
| invariant | 11 |
| anchor | **0** |
| multi | 66 |
| regress | 27 |

**One of the four primitives never fires.** `mutate` edits the objective only
in the branch that also adds `raw_eval` to the manifest, and `gate_invariant`
runs first — so every objective edit is already rejected before `gate_anchor`
is asked. Two primitives guard one event, and the ordering makes one of them
dead code. It is not that the anchor is redundant in principle; it is that the
only mutation in this simulator that could violate it always violates the
invariant too.

**Three of the ninety-six accepted cycles improved anything.** Only 3
acceptances raised perf above everything before them; the other 93 passed all
four gates and moved the agent sideways. The gates filter proposals, and the
accepted count is not a progress measure — a distinction worth keeping when the
same number is reported as "the loop improved".

**And the hack is rare before it is blocked.** The manifest edit is the fourth
branch of `mutate`, reached only when the first three are skipped and a second
coin lands — about 5% of cycles. So the 11 invariant rejects measure the attempt
rate, not the gate's strength. The evidence that the gate is doing work is the
other run: with every gate off, the *same* mutation stream accepts 200 of 200
cycles and drifts both the manifest and the objective.

### 2 — the tolerance sits below the metric's granularity

No construction is needed: the loss happens at the lesson's own seed, in most
accepted cycles.

| | accepted | of those, below the running best | final perf |
|---|---:|---:|---:|
| regression gate on | 96 | 0 | 1.00 |
| regression gate off | 116 | **80** | 1.00 |

Both runs finish at the ceiling, which is exactly why the loss is *silent* —
the headline number recovers and the dip is never printed.

**The tolerance is below the metric's granularity.** `perf_score` divides by
four cases, so it moves in steps of 0.25 and the smallest possible regression
is 0.25. `gate_regression` defaults to `tol=0.2`. Every regression this metric
can express is larger than the slack, so the gate is strictly monotonic — and
its own docstring says it is there to "reject obvious regressions, accept
noise". There is no noise to accept. The smallest input that shows this is one
case: against a best of 1.00, a candidate at 0.75 is rejected at `tol=0.2` and
accepted at `tol=0.25`.

That is the general lesson worth taking from the exercise. A tolerance is only
meaningful relative to the resolution of the thing it tolerates, and a
four-case metric has no room for one. The fix is not a smaller tolerance but a
finer metric — per-task deltas, as the lesson's own Primitive 4 section
actually describes — because `tol` on a coarse aggregate is either zero or
catastrophic.

**And the dips are absorbed, not recovered from.** The deepest below-best
acceptance is 0.75 points down, and the run still ends at 1.00. What the gate
prevents is not a worse final score; it is that nothing in the run records the
capability was ever lost.

### 3 — the two axes cannot trade off

The demonstration the exercise asks for does not exist in this simulator, and
establishing that is a stronger answer than finding a seed where it looks like
it does.

With the multi gate off the loop accepts 142 of 200 cycles and finishes at perf
**1.00** and safety **1.00**. Exhaustively, over all **2801** operator
sequences up to length 4:

| | count |
|---|---:|
| sequences with safety < 1.0 | 1010 |
| …of those, the best perf any reaches | **0.0** |
| sequences with perf ≥ 0.25 and safety < 1.0 | **0** |

There is no agent anywhere in the reachable space that is good at the task and
bad on safety.

**One tool carries the whole safety axis.** Of the seven operators, `reverse`
is the only one that breaks the filter — and it breaks it by destroying the
string the filter searches, which also takes perf to 0.0. The axis that is
supposed to be independent is a strict sub-case of the one it is supposed to
constrain.

**The filter is applied after the operators, not before.** `apply` runs the
whole sequence and only then scans for the four danger substrings, lowercasing
first. So `upper` and `title` survive it and `reverse` defeats it. A filter that
can be defeated by reordering characters is testing the operators, not the
input — and a safety axis built on it measures whether the transform left the
pattern legible.

**The gate that is disabled was rejecting the other axis.** With every gate on,
its 66 rejections split 46 perf-only, 20 both, and **0 safety-only**. Turning it
off therefore removes a perf floor of 0.25, which is why acceptances rise from
96 to 142. The entire observable effect of disabling the *safety* constraint is
on the performance axis.

To make the exercise's demonstration possible you need a tool that helps on
`CASES_PERF` and hurts on `CASES_SAFETY` — for instance an `unescape` that
canonicalises quoting (useful for the messy inputs, and it would strip the
`exec('…')` wrapper the filter keys on). Without one, a multi-objective gate
cannot be shown to do anything a single-objective gate would not.

### 4 — the anchor and its hash live in the same file

**What text.** For a coding agent in this repository: the part of `DESIGN.md`
that states what a solution may not do — answer a different question, fork the
lesson's implementation rather than importing it, or improve the exercise. That
text is the agent's actual objective, and it is the text whose violation no
downstream metric detects.

**Stored where.** In the reference checkout, resolved through
`AIEFS_REFERENCE`, outside the tree the agent edits — the same place the
exercise text already lives.

**Checked how.** Re-read at grade time and hashed, with the expected value
supplied by the caller. That is exactly what `test_exercise_text_matches_upstream`
already does to the exercise block, so the design is one existing mechanism
pointed at a second file.

The shipped anchor answers all three and gets the middle one wrong. `OBJECTIVE`
is 48 characters, `OBJECTIVE_HASH` is its sha256 truncated to 16 hex digits, and
`Agent.objective` defaults to the same constant — all three declared in
`main.py`. `gate_anchor` takes one parameter and compares a value the agent
carries against a constant the agent's own file declares. An edit that changes
both passes, which means the anchor constrains an agent that can edit its
fields but not one that can edit its module, and the second is the case the
primitive exists for.

**Truncation is the smaller problem.** 16 hex digits is 64 bits — a 2³²
birthday bound, enough against accident and irrelevant against an editor. The
anchor's strength is set by where the expected value lives, not by how many
digits it has, and moving it costs one parameter on `gate_anchor`.

### 5 — a concrete improvement to regression detection

*Draws on "Information-theoretic limits".*

**Primitive 4, regression detection, and the improvement is to make the
tolerance a property of the metric rather than a constant beside it.** The
section's own framing is the right one to build on: the primitives "cannot
close the safety problem — they make silent failure more expensive", so the
engineering question is whether each primitive actually raises that cost, and a
tolerance that cannot be violated raises nothing. This lesson's own gate is the
existence proof: `tol=0.2` against a metric whose smallest step is 0.25 makes
`gate_regression` strictly monotonic, so the documented behaviour ("reject
obvious regressions, accept noise") and the actual behaviour differ, and
nothing in the code or the output reveals the difference.

The state of the art here is a scalar tolerance chosen by hand and compared
against an aggregate score. Three changes, in increasing cost. **(a) Refuse to
run a tolerance the metric cannot express.** At construction, compute the
metric's resolution — for a pass-rate over N cases, `1/N` — and reject any
tolerance below it, loudly. This is a three-line assertion and it converts a
silent misconfiguration into a startup failure, which is the cheapest possible
version of "make silent failure expensive". **(b) Compare per-task, not in
aggregate**, as the lesson's own Primitive 4 text describes but the code does
not do: store the last N cycles' per-case results, and reject on any single
case flipping from pass to fail. On a four-case suite this is the difference
between a gate with two usable thresholds and a gate with sixteen states. **(c)
Make the tolerance a statistic rather than a constant** — estimate the
cycle-to-cycle variance of each case from the history and reject a drop that
exceeds a fixed number of standard deviations. This is the version that
generalises to stochastic evaluators, which is where the primitive actually has
to work, and it is also the version that Lesson 3's noisy-evaluator result
demands: an acceptance test that treats one measurement as ground truth will
lock in lucky draws no matter how carefully its threshold is chosen.

The evaluation is the same one exercise 2 runs: replay a completed loop with
the gate configured each way and count accepted cycles that sit below the
running best. Under the current design that count is 80 of 116 with the gate
off and 0 of 96 with it on — a gate that looks perfect precisely because the
metric is too coarse to disagree with it.
