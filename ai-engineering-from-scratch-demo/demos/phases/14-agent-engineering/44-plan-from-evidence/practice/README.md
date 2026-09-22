<!-- generated:start -->
# 14-agent-engineering / 44-plan-from-evidence

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/44-plan-from-evidence/) · upstream spec
`phases/14-agent-engineering/44-plan-from-evidence/docs/en.md`

```bash
uv run demo practice run 44-plan-from-evidence --ex 1
uv run demo explain 44-plan-from-evidence --ex 1
uv run pytest demos/phases/14-agent-engineering/44-plan-from-evidence
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a migration item that requires explicit human approval. | code | T0 | `ex01_the_irreversible_item_schedules_into_wave_two_beside_the_docs.py` |
| 2 | Create a cycle and explain the hidden product disagreement behind it. | code | T0 | `ex02_the_cycle_report_names_every_blocked_item_not_the_two_in_the_loop.py` |
| 3 | Split one item that has two proof commands. | code | T0 | `ex03_two_proofs_in_one_string_pass_the_check_that_exists_to_count_them.py` |
| 4 | Add a work item that can run in the second wave without touching either existing branch. | code | T0 | `ex04_the_new_wave_two_item_is_not_something_the_gate_waits_for.py` |
| 5 | Render the plan as Markdown while keeping JSON as the source of truth. | code | T0 | `ex05_the_blocked_plan_renders_no_schedule_in_either_format.py` |
<!-- generated:end -->

## Answers

### 1 — the irreversible item schedules into wave two, beside the docs

The item is easy to write. Where the scheduler puts it is the lesson.

Adding `migration` ("add the unique index on normalized email") with
`depends_on=("contract",)` produces
`[['contract'], ['docs', 'implementation', 'migration'], ['integration']]` —
validate returns no issues, status `ready`. The one step in this plan that cannot
be undone by editing a file runs concurrently with the item still deciding how the
public contract is worded.

`WorkItem` has five fields — `id`, `change`, `evidence`, `depends_on`, `proof` — and
none of them says "ask first". Approval, reversibility and blast radius have
nowhere to live, so the dependency graph is the only enforcement available. And
`validate` implements 5 of the 6 rejection rules the docs list: duplicate ids,
missing evidence, missing proof, unknown dependency, cycle. The sixth — *the first
irreversible action occurs before the relevant uncertainty is resolved* — is
exactly this exercise, and has zero lines behind it.

Modelling approval as a node the migration depends on does move it, and turns up a
second bug on the way:

```
[['contract'], ['docs', 'implementation'], ['approval', 'integration'], ['migration']]
```

The migration now runs last, after its approval. But the integration gate sits in
wave 3, *before* the schema change it is supposed to verify — because nothing in
the plan says the gate depends on the migration. The scheduler is right and the
plan is wrong, which is the failure mode this whole lesson is about.

### 2 — the cycle report names every blocked item, not the two in the loop

The cycle worth building is the one the lesson's example exists to prevent: `docs`
waits to learn which status code shipped, `implementation` waits to be told which
status code was promised. Neither party is being unreasonable. The deadlock is a
disagreement about **who owns the public contract**, and writing it as a graph is
what makes it visible instead of a slow argument in review.

The resolution is the lesson's own shape: one `contract` node both depend on. Same
two items, now scheduling together in wave 2, 3 waves total, 0 issues. The fix for
a cycle is usually a missing node, not a deleted edge.

Two things about the diagnosis itself:

**The report names the blocked set, not the cycle.** Add one innocent item
downstream of the loop and `validate` says "dependency cycle among: docs,
implementation, integration" — three names for a two-item cycle, because
`execution_waves` prints whatever is left in `remaining` when nothing is ready.
`integration` is blocked, not looping, and a reader has to re-derive which edge to
cut.

**A cycle costs every other check its output.** `plan_document` returns
`"waves": []` for any issue at all, so the one item that was schedulable loses its
ordering too. The first failure hides the rest of the report.

### 3 — two proofs in one string pass the check that exists to count them

`proof` is a single string, so "an item with two proof commands" is something the
model permits rather than something it records.
`python3 -m unittest && python3 scripts/check_links.py` validates with zero issues.

Split, the plan has 5 items, still 3 waves, and a last wave of
`['integration-links', 'integration-tests']`. Wave widths go from `[1, 2, 1]` to
`[1, 2, 2]`: the split buys a verdict, not a step.

Why bother, precisely: the two commands fail for unrelated reasons. One is the
acceptance suite, one is a link checker. As a single string the plan cannot record
"tests pass, docs link is broken" — and a session resuming mid-item cannot tell
which half already ran, which is the resumability the lesson asks for.

The shipped example has a sharper version of the same problem. Its `contract`
item's proof is `"review contract"` — prose, not a command — and that is the item
`implementation` and `docs` both depend on. 1 of 4 proofs is unexecutable, and it
is the root of the graph. `validate`'s only proof rule is an emptiness test, so
nothing notices.

### 4 — the new wave-two item is not something the gate waits for

"Second wave without touching either branch" pins the shape: the item may depend on
`contract` and nothing else. A telemetry item ("count duplicate rejections by
normalized domain") fits, and schedules into
`[['contract'], ['docs', 'implementation', 'telemetry'], ['integration']]` with no
issues.

Then look at the other end. `integration` depends on 2 of the 3 items in wave 2.
The gate closes the plan without ever proving the third.

**The plan now has two terminal nodes where its author believes it has one.**
Nothing depends on `telemetry`, so it is a sink exactly like `integration`. "Run
the complete acceptance gate" is the name of one of two ways this plan can end.

**Wave position is an accident, not a guarantee.** There are 0 edges between
`telemetry` and `integration` in either direction; the gate's ancestors are
`['contract', 'docs', 'implementation']`. The only thing putting telemetry first is
the wave it happened to land in, and a resuming session reading the graph is free
to run them in either order — both orders satisfy every dependency.

The fix is one edge. Adding `telemetry` to the gate's dependencies keeps 3 waves
and the same wave-2 width; the only change is that the plan can no longer finish
without the item it just added. That is the property you wanted from "add an item",
and the graph does not give it to you for free.

### 5 — the blocked plan renders no schedule, in either format

"JSON as the source of truth" is testable. The renderer reads `plan_document` and
nothing else; the Markdown reproduces all 4 top-level keys and all 20 item fields,
renders 3 wave rows, and parses back to the same 4 ids in the same order. No field
in the Markdown is absent from the JSON.

What the round trip exposes is what the JSON does not carry.

**A plan blocked on a missing proof renders with no waves at all.**
`plan_document` sets `"waves": []` whenever `issues` is non-empty. A plan whose only
fault is one empty proof string — a fault that changes no edge — loses all 3 of its
computable waves. The Markdown inherits the hole, correctly, because the JSON is
the source of truth. The renderer is not the place to fix this; `plan_document` is.

**The waves are computed twice and discarded once.** `validate` calls
`execution_waves` to detect cycles, and `plan_document` calls it again to report
them. On a blocked plan the second call never happens and the first call's result
is thrown away. Keeping it would cost nothing and would leave a blocked plan with a
schedule a human could read.

**Evidence renders as a string and nothing resolves it.** All 4 receipts in the
example name files that do not exist in the lesson directory — the same fiction as
the previous lesson's frame. A Markdown table makes them look like links, which is
the one way rendering can make a document less honest than its source.
