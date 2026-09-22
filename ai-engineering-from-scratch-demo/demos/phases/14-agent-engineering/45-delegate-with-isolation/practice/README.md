<!-- generated:start -->
# 14-agent-engineering / 45-delegate-with-isolation

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/45-delegate-with-isolation/) · upstream spec
`phases/14-agent-engineering/45-delegate-with-isolation/docs/en.md`

```bash
uv run demo practice run 45-delegate-with-isolation --ex 1
uv run demo explain 45-delegate-with-isolation --ex 1
uv run pytest demos/phases/14-agent-engineering/45-delegate-with-isolation
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Decompose a real change into two independent work units and one integrator. | code | T0 | `ex01_the_only_shared_path_is_the_one_a_script_writes.py` |
| 2 | Find a proposed parallel split that only looks independent. State the shared decision. | code | T0 | `ex02_the_shipped_split_is_two_workers_writing_the_same_status_code.py` |
| 3 | Add a read-only research worker whose output is a fact table. | code | T0 | `ex03_read_only_is_the_one_thing_the_contract_cannot_say.py` |
| 4 | Add a merge gate that checks the final changed-file set against all unit contracts. | code | T0 | `ex04_the_merge_gate_finds_the_file_no_contract_claims.py` |
| 5 | Define a cancellation rule for a worker whose dependency becomes invalid. | code | T0 | `ex05_cancelling_by_deletion_makes_the_planner_raise.py` |
<!-- generated:end -->

## Answers

### 1 — the only shared path is the one a script writes

"Real" means the decomposition has to survive the repository's own history. The
change decomposed here is the one this repository performs constantly: ship the
practice solutions for two lessons.

- **lesson-43** (worker-a) owns `…/43-frame-the-task-before-code/practice`.
- **lesson-44** (worker-b) owns `…/44-plan-from-evidence/practice`.
- **integration** (integrator) owns `README.md`, depends on both, and runs the
  phase-wide audit.

Zero path conflicts, waves `[['lesson-43', 'lesson-44'], ['integration']]`, status
`ready`. The interesting part is the third unit's path: the repository's top-level
`README.md` is touched by **6 of the last 6** lesson commits, because
`scripts/coverage.py` rewrites it from the manifests. It is not a file either
worker should own — it is a generated artifact, and the integrator is the only
party that can regenerate it once.

Three gaps in the artifact while we are here. `WorkUnit` has 5 fields where the
docs table asks for 6: `goal` and `handoff` are missing — the two that say what the
unit is for and what came back, which is what an integrator reads. A unit owning no
paths validates `ready`, so a read-only worker and a decomposition error are
indistinguishable. And `conflicts` compares only *distinct* units, so one worker
claiming the phase directory produces two findings against a neighbour that owns
two nested paths, while those two nested paths overlap each other and nothing
reports it.

### 2 — the shipped split is two workers writing the same status code

"Only looks independent" means the path check passes and the work still collides.
The lesson ships one: `api` owns `app/api`, `docs` owns `docs/api.md`, zero
conflicts, both in wave 1.

**The shared decision is the response contract** — what a duplicate signup returns.
Both workers write it down, in different files, from different worktrees. Neither
proof can catch the disagreement: the api worker runs `python3 -m unittest
tests.test_api`, the docs worker runs `python3 scripts/check_links.py`, and a link
checker cannot notice that the page promises 409 while the handler returns 422.

The first command that reads both surfaces is `integration`, in wave 2 — after both
workers have finished. The lesson's own text says the merge contract "must resolve
shared interfaces before work begins", and `WorkUnit` has no field to write one in.

Filesystem isolation makes this *more* likely, not less: two worktrees mean neither
worker sees the other's draft, so the contract gets decided twice and the conflict
surfaces at integration instead of at the keyboard. Three isolation layers are
named in the docs; the one that feels most like safety is the one hiding the
problem.

The fix is an ordering, not a check. A `contract` unit owning `docs/contract.md`
that both workers depend on gives `[['contract'], ['api', 'docs'], ['integration']]`
with zero conflicts — the shared decision made once, in wave 1, by someone
accountable for it.

### 3 — read-only is the one thing the contract cannot say

A research worker is defined by what it may *not* do, and the contract describes
only what a unit may write. So the worker is expressible as "owns
`outputs/facts.md`" and nothing more, and the fact table is what makes the slot
worth spending.

The table built here answers five questions about the planner itself, each row
carrying a `path:line` receipt — and all 5 resolve against the line they cite:

| Question | Claim | Receipt |
|---|---|---|
| What does a unit promise? | five fields, no goal | `code/main.py:14` |
| How is overlap decided? | a `PurePosixPath` parent test, not a glob match | `code/main.py:25` |
| What happens to a bad dependency? | `waves` raises instead of reporting | `code/main.py:43` |
| What does the plan report? | status derived from three lists | `code/main.py:63` |
| How big is the shipped split? | three units | `code/main.py:72` |

Two of those rows change the plan rather than describe it, which is the test Lesson
44 sets for evidence.

`WorkUnit` cannot express read-only: five fields, all about writes, no `reads`
field, no flag — and the module never executes a worker at all, so nothing is
constrained either way. State isolation does work, but exactly to the depth of the
strings: `paths_overlap("outputs", "outputs/facts.md")` is `True`, so a worker
claiming `outputs/` collides with the researcher, while two distinct files under it
do not. And the fact table only pays for itself if it runs first — wired as a
dependency of `api` and `docs` it gives 3 waves; left unattached it is a second
sink whose findings arrive after the code they were meant to inform.

### 4 — the merge gate finds the file no contract claims

The planner checks contracts against each other before the work. The merge gate
checks the work against the contracts afterwards — a different question needing a
different input: the actual changed-file set, from git.

On a real temp checkout with three workers' edits, `git status --porcelain` reports
6 paths and the gate flags 2:

- `scripts/release.sh` — claimed by no contract at all.
- `tests/test_api.py` — owned by the docs unit, edited by the api worker.

`delegation_plan` said `ready` before any of this happened, and it was right to:
the contracts are mutually consistent. Consistency is not compliance.

**The gate needs the owner, not just the union of paths.** Membership in the union
catches 1 of the 2 problems. Attributing each change to the worker that made it
catches the other, and `WorkUnit` already carries an `owner` field to compare
against — it is simply never used after planning.

**`paths_overlap` is blind to globs.** `paths_overlap("app/**", "app/api/routes.py")`
is `False` where the directory-prefix form is `True`. A contract written the way
most people write contracts passes the pre-work check and then owns nothing at
merge time.

**And the gate is the first thing in this lesson that reads the repository.**
`delegation_plan` takes one argument — the list of contracts — and the module never
imports `subprocess`. Everything before the merge is a check on the plan's internal
consistency. A perfectly consistent plan can describe work nobody did.

### 5 — cancelling by deletion makes the planner raise

**The rule: cancel by marking, never by deleting, and cascade to the transitive
dependents while keeping the cancelled unit's paths reserved.**

Each clause earns its place:

**Marking, not deleting.** Dropping `api` from the example makes
`delegation_plan` raise `ValueError("unknown dependency")` instead of returning a
blocked document — `waves` is guarded only against duplicate ids. Marking keeps 3
units, cancels `api` and `integration`, leaves `docs` running, and still renders a
plan a human can read. More generally: two of the three failure modes (unknown
dependency, cycle) escape as exceptions while overlap and missing proof come back
as fields. A caller that wants a report has to wrap the call in `try`, which is the
shape of an API that never considered the blocked case its job.

**Cascading transitively.** Cancellation is a graph question. Cancelling one unit
stops 1 more in the shipped three-unit example and 3 in a five-unit chain. A rule
that only stops direct dependents stops the wrong amount of work, and the amount it
misses grows with the depth of the plan.

**Keeping the paths reserved.** Freeing `app/api` the moment `api` is cancelled
lets another worker claim a half-finished tree while `integration` still holds
state derived from it. Keeping the reservation costs zero conflicts here and is the
difference between a cancellation and a race. The reservation lifts when the
cancelled unit's dependents have all settled — not when the unit stops.
