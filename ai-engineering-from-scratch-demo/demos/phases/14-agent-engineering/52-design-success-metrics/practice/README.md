<!-- generated:start -->
# 14-agent-engineering / 52-design-success-metrics

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/52-design-success-metrics/) · upstream spec
`phases/14-agent-engineering/52-design-success-metrics/docs/en.md`

```bash
uv run demo practice run 52-design-success-metrics --ex 1
uv run demo explain 52-design-success-metrics --ex 1
uv run pytest demos/phases/14-agent-engineering/52-design-success-metrics
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Derive three questions from one outcome goal. | code | T0 | `ex01_three_questions_and_the_third_one_needs_a_metric_the_plan_has_no_kind_for.py` |
| 2 | Add a counter-metric that catches cost shifted to another role. | code | T0 | `ex02_the_counter_metric_shows_ten_of_forty_five_files_over_the_line_target.py` |
| 3 | Define the source, population, and window for every metric. | code | T0 | `ex03_population_is_the_field_the_table_asks_for_and_the_dataclass_omits.py` |
| 4 | Write pass, fail, and ambiguous decisions before generating values. | code | T0 | `ex04_the_boundary_is_inclusive_and_one_way_of_computing_the_rate_fails_it.py` |
| 5 | Identify one metric that is easy to collect but cannot change the decision. Remove it. | code | T0 | `ex05_the_metric_that_cannot_move_is_the_one_the_rule_already_fixes.py` |
<!-- generated:end -->

## Answers

### 1 — three questions, and the third needs a metric the plan has no kind for

The goal is the one this repository has been working to: *a reader can run an
answer to every exercise and see it check its own claims against the lesson's
code.* Three questions fall out of it, and the third is about somebody else:

1. **Does every shipped answer still pass?** → `shipped_answers_passing`, at-least
   1.0, over every finished lesson. Measured: **45 of 45** across nine lessons.
2. **Does any answer make a claim its own run does not support?** →
   `traced_number_rate`, at-least 0.9, over one lesson's answers section.
3. **What does an answers section cost the reader?** → `answer_section_lines`,
   at-most 150. Measured mean: **132.7** lines.

The third is a counter-metric, and `validate` has no kind for it. It enforces
exactly two — refuse a plan with no `outcome` and no `guardrail` — while `kind` is
otherwise a free string, so a metric declared `counter` satisfies neither
requirement and raises no issue of its own. Two of the three kinds the lesson
describes are enforced; the third exists only as a value.

Two further properties of the artifact. A plan is valid before any value exists:
reporting with an empty value map returns status `valid` with three rows marked
`missing`, which is correct and worth saying out loud — validity is a property of
the plan, never of the measurement. And `MeasurementPlan` holds goal, questions and
metrics with nothing joining the last two, so a plan with three questions and one
metric passes, as does one with no questions and three metrics. The derivation this
lesson teaches leaves no trace in the artifact that records it.

### 2 — the counter-metric shows ten of forty-five files over the line target

The outcome metric measures the reader. The cost of reaching it lands on two other
people, and both are measurable from the tree:

- **10 of 45** shipped solution files sit above this repository's own 120-line
  target, the largest at 145 — maintainer cost.
- The answers sections average **132.7** lines, worst case 146 — reviewer cost.

Both come from the same nine lessons whose answers pass 45 of 45. A plan carrying
only the outcome metric reports a perfect result while the cost it created sits
entirely outside the report.

The counter-metric has to be `at-most` on a number that improving the outcome
pushes *up*: longer answers make a claim easier to support, so
`answer_section_lines at-most 150` is the one that protects against the outcome
metric. It currently passes by four lines — a counter-metric with slack is still
the metric that will fail first.

And the structural note: adding it leaves the plan valid with three metrics across
three distinct kinds, while `report` renders one row per metric with `passed` per
row and no aggregate. A run where the outcome passes and the counter-metric fails
renders as two passing rows and one failing row with no verdict, which is what
Exercise 4 exists to supply.

### 3 — population is the field the table asks for and the dataclass omits

Two of the three inputs are fields on `Metric`; the third is not. `Metric` carries
`name`, `direction`, `threshold`, `window`, `source`, `kind` — six against the
docs' seven. So "define the population for every metric" means first deciding where
it goes: a seventh field (taking each metric to seven) or the name, which is what
the shipped example does with `median_identification_seconds`.

`validate` refuses a metric whose `source` or `window` is blank and has no third
test, because there is no third field. Two of the three reproducibility inputs are
enforced.

Why it matters concretely: `shipped_answers_passing` is 1.0 over the **45** answers
in nine lessons and also 1.0 over the **5** in any single one. Same threshold, same
direction, same source, different evidence. A plan that records the window but not
the population lets the smaller claim be read as the larger one — which is the
failure mode that makes a metric unreproducible even when it is written down.

And `window` and `source` are free strings. A plan citing a window of "vibes" from
a source of "memory" returns zero issues. The schema guarantees the fields are
present, never that they are real.

### 4 — pass, fail and ambiguous, written before the values

> **pass:** every shipped answer passes *and* the traced rate reaches 0.9.
> **fail:** any file over the 150-line hard ceiling, or a traced rate below 0.75.
> **ambiguous:** anything else — buy a larger replay set.

Then the values: 45 of 45 answers passing, a traced rate of **0.875**, zero files
over the ceiling. Neither the pass rule nor the fail rule fires. The decision is
**ambiguous**, and the consequence was written down before the number was known,
which is the entire point of writing it first.

`report` cannot make that call. It returns four keys and one row per metric with
`passed` per row — on these values, two passing rows and one failing row and no
aggregate. The mixed result is exactly what a decision rule exists to interpret,
and it lives outside the artifact.

Two boundary notes. `evaluate` compares with `<=` and `>=`, so the shipped example's
`correct_service_rate` of 0.9 against a threshold of 0.9 passes *by equality* —
every boundary case lands on the generous side, which should be a deliberate choice
rather than an inherited one. And how the number is computed decides the boundary:
nine tenths as `9 / 10` passes an at-least 0.9, while the same nine tenths summed
from `0.3` gives `0.8999999999999999` and fails. A decision rule has to name the
arithmetic, not just the number.

### 5 — the metric that cannot move is the one the rule already fixes

"Cannot change the decision" is a property of the data, so the test is to compute
the metric across the whole window and count distinct values.

`solution_files_per_lesson` is **5** in every one of the nine finished lessons.
One distinct value, variance zero — because `DESIGN D10` requires one file per
exercise and each of these lessons has five exercises. The metric is a restatement
of a rule that lives elsewhere in the system, not evidence about this build. It
comes out, taking the plan from four metrics to three.

Removing it leaves the plan valid: still one outcome and one guardrail, zero
issues. Nothing notices that a question lost its answer, because nothing ever
linked questions to metrics.

The contrast is what makes the test useful. `files_over_line_target` is computed
over the same 45 files, reads **10** against a threshold of 0, and fails. Two
metrics over one population: one could never have moved, one did. And in the report
they are indistinguishable — the constant renders as
`{"name": "solution_files_per_lesson", "value": 5, "passed": true}`, exactly like a
measured pass. One of the four rows was decoration and the document gave a reader
no way to tell which.
