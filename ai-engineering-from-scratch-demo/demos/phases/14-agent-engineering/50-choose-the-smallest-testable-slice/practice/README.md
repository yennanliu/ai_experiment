<!-- generated:start -->
# 14-agent-engineering / 50-choose-the-smallest-testable-slice

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/50-choose-the-smallest-testable-slice/) · upstream spec
`phases/14-agent-engineering/50-choose-the-smallest-testable-slice/docs/en.md`

```bash
uv run demo practice run 50-choose-the-smallest-testable-slice --ex 1
uv run demo explain 50-choose-the-smallest-testable-slice --ex 1
uv run pytest demos/phases/14-agent-engineering/50-choose-the-smallest-testable-slice
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Design three slices for the same outcome at different consequence levels. | code | T0 | `ex01_the_cheapest_slice_scores_highest_and_proves_half_of_what_matters.py` |
| 2 | State the required proof set before scoring them. | code | T0 | `ex02_the_proof_set_disqualifies_the_slice_with_the_best_score.py` |
| 3 | Remove one capability while preserving the decisive evidence. | code | T0 | `ex03_dropping_the_prose_keeps_both_proofs_and_raises_the_score.py` |
| 4 | Add a stop rule for a failed pilot. | code | T0 | `ex04_the_stop_rule_has_five_branches_and_the_artifact_has_none.py` |
| 5 | Identify a reusable platform component that should wait until after the slice. | code | T0 | `ex05_the_platform_slice_stays_ineligible_at_any_score.py` |
<!-- generated:end -->

## Answers

### 1 — the cheapest slice scores highest and proves half of what matters

The outcome is the one Lesson 49 mapped: generate a lesson's answers section
instead of writing it. Three slices, differing in what they put at risk:

| Slice | Value | Unc. | Effort | Cons. | Rev. | Score |
|---|---|---|---|---|---|---|
| generate one lesson, publish nothing | 3 | 3 | 2 | 1 | yes | **2.4** |
| publish one lesson carrying a planted error | 4 | 5 | 3 | 3 | yes | **2.0** |
| generate and publish every remaining lesson | 5 | 5 | 9 | 5 | no | **0.526** |

Ranked by score, the weakest evidence comes first — which is why Exercise 2's gate
exists and why the lesson says the arithmetic matters less than the eligibility
test.

Three properties of that arithmetic, worth knowing before quoting it. The
reversibility flag multiplies consequence by 2 or by 0.5, a 4x swing: marking the
largest slice reversible lifts it from 0.526 to 0.87 without changing any of the
work. Effort and consequence are *added*, so a one-day change that publishes a
wrong number to every reader (effort 1, consequence 5, irreversible) and an
eleven-day change with no consequence at all share a denominator of 11 — the
formula is simple in the direction that flatters the risky option. And `proves` is
a tuple of bare strings, joined to the assumption map of the previous lesson by
convention only: an underscore instead of a hyphen makes a slice silently
ineligible.

### 2 — the proof set disqualifies the slice with the best score

Stated first, from the open assumptions ranked highest last lesson:

> **Required proof:** `numbers-trace`, `wrong-number-detected`.

`choose` filters on `required_proof <= set(item.proves)` *before* it compares
anything, so the 2.4 slice — which only proves that numbers trace — never reaches
the arithmetic. Two of three candidates are eligible and the selected one scores
2.0. That is the gate doing its job, and it only works because the set was written
before the scores were.

Three things about the gate itself. It is a subset test, so proving *more* is free
(the largest slice proves an extra assumption nobody asked for and stays eligible)
while proving *differently* is fatal (a slice offering `wrong-number-detected-late`
is rejected outright). An empty required set is a subset of everything, so a run
with the gate unfilled returns the cheapest slice and looks exactly like a
considered decision. And when nothing qualifies, `decision` raises `ValueError`
rather than returning a document that says so — the same shape Lesson 45's planner
had when a dependency went missing.

### 3 — dropping the prose keeps both proofs and raises the score

The capability to remove is the one that costs effort without carrying proof. In
the selected slice the generator writes both the prose and the numbers, and both
required assumptions are about the numbers.

Removing prose generation: effort 3 → 2, outcome value 4 → 3 (a numbers-only answer
is worth less to a reader), proofs unchanged. Score **2.0 → 2.286**, still eligible,
same selection. Zero of the two required proofs mention prose; the decisive
measurement is a number appearing in a graded check detail, which the previous
lesson ran over 16 values in one lesson.

Keep going and the score does something worth seeing: 2.286 → 2.8 → 2.4. Effort
sits in the denominator beside a consequence term, so the first cuts are worth more
below the line than above it and later ones are not. The score has an interior
maximum that has nothing to do with what the slice proves. Only the eligibility
gate stops a team from optimising a slice down to proving nothing.

And `Slice` has seven fields, none of which lists a capability — "generates prose"
exists only inside the name string. What was removed is recorded in this paragraph
and nowhere in the artifact.

### 4 — the stop rule has five branches and the artifact has none

Written before the pilot, mapping a measured result to one of the five endings the
lesson lists:

| If | Then |
|---|---|
| the detector fires and readers still miss the error | abandon the outcome |
| the rate clears the threshold and nobody reads it | change the user or situation |
| the rate falls more than fifteen points below threshold | test a different mechanism |
| the misses trace to the metric rather than the generator | collect better evidence |
| a planted error survives more than one publish cycle | narrow authority |

None of the five is "keep building", which is the test the lesson sets, and all
five are selected by something measurable rather than by an impression.

The pilot's result is already in hand from Lesson 49: **70.8%** traced numbers
against a 90% threshold — more than fifteen points below it, so the branch that
fires is **test a different mechanism**, not the near-miss one. Had the rate been
86.2% (the best per-lesson rate in the same run), the same rule would say *collect
better evidence*. That the branch depends on one number is exactly why the rule has
to be written before the number is known.

`Slice` has 7 fields and `decision` returns 3 keys; none holds a stop rule. The one
thing written before the pilot that is supposed to bind afterwards is the one thing
that does not travel with the decision.

### 5 — the platform slice stays ineligible at any score

The component: a general answer generator for every phase, rather than one lesson.
It is the "platform minimum" the lesson names — reusable machinery built before one
workflow has earned it.

With plausible numbers it scores 0.231 and loses on merit. The interesting version
is the inflated one: value 9, uncertainty 9, effort 1, consequence 0 gives **18.0**,
the best score in a field of four — and `choose` still returns the one-lesson pilot,
because the subset gate runs before the comparison. No arithmetic rescues a slice
that proves nothing required.

Emptiness is not what disqualifies it: a platform proving *one* of the two required
assumptions is rejected identically, because the gate is a subset test. Partial
coverage buys nothing, which is what makes the proof set the thing worth arguing
about before any slice is drawn.

Three of the lesson's five false minimums are present in this field: the platform
(reusable machinery), the UI-only minimum (generating without publishing, which
removes the publishing uncertainty), and the happy-path minimum (any version
without the planted error). Infrastructure-only and demo are absent because the
outcome already names a reader.

Last: `decision` returns the alternatives with their scores and eight keys each,
and none of them names the proof the loser lacked. A reader sees the platform
scoring 18.0 and losing, with the reason recorded nowhere.
