<!-- generated:start -->
# 14-agent-engineering / 53-prototype-pilot-or-production

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/53-prototype-pilot-or-production/) · upstream spec
`phases/14-agent-engineering/53-prototype-pilot-or-production/docs/en.md`

```bash
uv run demo practice run 53-prototype-pilot-or-production --ex 1
uv run demo explain 53-prototype-pilot-or-production --ex 1
uv run pytest demos/phases/14-agent-engineering/53-prototype-pilot-or-production
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Classify three current projects by learning stage, not deployment status. | code | T0 | `ex01_the_stage_ignores_consequence_whenever_the_data_is_synthetic.py` |
| 2 | Write pilot exit criteria that include a stop decision. | code | T0 | `ex02_exit_criteria_is_one_string_in_a_list_of_five.py` |
| 3 | Add a technical control that prevents a prototype from reaching production data. | code | T0 | `ex03_the_control_that_keeps_the_prototype_out_is_a_monkeypatch_in_25_files.py` |
| 4 | Identify the first operational responsibility that makes the build production. | code | T0 | `ex04_the_first_production_responsibility_is_the_one_nobody_can_skip.py` |
| 5 | Design a rollback receipt for the bounded pilot. | code | T0 | `ex05_the_rollback_is_one_commit_and_the_receipt_has_to_name_the_dependents.py` |
<!-- generated:end -->

## Answers

### 1 — the stage ignores consequence whenever the data is synthetic

Three real pieces of work in this repository, classified by the lesson's own
decision function rather than by where they run:

| Work | Real users | Real data | Cons. | Rev. | Ready | Stage |
|---|---|---|---|---|---|---|
| the practice solutions | yes | yes | 2 | yes | yes | **production** |
| the answers generator (Lessons 49–52) | yes | yes | 3 | yes | no | **pilot** |
| the scaffolder fallback | no | no | 2 | yes | yes | **prototype** |

The third is the point of the exercise: it is classified `prototype` and it ships
in `scripts/`, imported by every lesson since. Deployment status and learning stage
are independent, and only one of them is in the decision.

Then the mechanism underneath. `choose_stage` tests "no real users and no real
data" *first*, so a decision at consequence 5, irreversible, with no operational
readiness still returns `prototype` and draws 3 controls. Stage drift is not a
failure of discipline here — it is the function's first line, and any prototype that
later acquires users inherits a classification made when it had none.

Two smaller things. `BuildDecision` has six fields and `choose_stage` reads five:
the `unknown` — the field that states the learning question — is never consulted,
while the prototype control list names "learning question" as a control. And the
controls are prose: the three stages draw 3, 5 and 8 strings, of which zero mention
configuration, access control or telemetry. The doc says a warning banner is not
enough; a list of three phrases is a banner with three lines.

### 2 — exit criteria is one string in a list of five

The pilot's control list already contains the words "exit criteria", so the work is
writing what the string stands for:

> **expand** — every shipped answer passes and the traced rate is at least 0.9.
> **stop** — any shipped answer fails, or the traced rate falls below 0.75.
> **revise** — everything else; the pilot continues in its bounded form.

The real numbers: **50 of 50** shipped answers passing, a traced rate of **0.875**.
Neither expand nor stop fires, so the pilot **revises** — it does not get to expand
on a near miss, and the reason was written before the number.

Three notes on the artifact. `required_controls("pilot")` returns five strings of
which one is "exit criteria", and `plan` returns three keys — nothing that can hold
a threshold, a window or a decision, so a pilot with three written criteria and one
with none produce identical plans. The stop path is the branch that needs its
number first: one failing answer moves this same run from `revise` to `stop`, and
stop is the decision somebody will want to relitigate. And `BuildDecision` carries
no date, window or budget, so "bounded duration" is unenforceable — a pilot running
a week and one running a year are the same record.

### 3 — the control that keeps the prototype out is a monkeypatch in 25 files

The control the exercise asks for already exists here, and it is worth naming
because it is the kind that works: a solution that needs to exercise a lesson's
*writer* points the module's output path at a temporary directory before calling
it.

**25 shipped solutions redirect a path into `tempfile`**, four of them by rebinding
the pack assembler's own `PACK` global before running it. The receipt is that the
upstream reference tree reports **0** modified paths — the prototypes ran, wrote,
and reached nothing real.

It works because the production path is a module global computed from `__file__` at
import time. A writer that computed its destination inside the function would leave
nowhere to intervene, and every solution that needed to exercise it would have to
fork the code it is supposed to import — trading a data-safety problem for a
correctness one.

And `required_controls("prototype")` names none of this: "synthetic or recorded
inputs", "discardable implementation", "learning question" — zero mentions of a
path, a directory or a credential. The check that the control held (`git status` on
a tree the tests never write to) is cheaper than the control itself, and it is the
artifact the stage decision should carry.

### 4 — the first production responsibility is the one nobody can skip

Of the eight production controls the docs list, this repository carries four:
continuous monitoring (`audit_practice.py` and the test suite), rollback (one
commit per lesson), recovery (`git revert`) and a retirement path (delete the
directory). It is missing the service level objective, the on-call owner, the
security review and cost controls — and it has **0** CI workflows, so even the
monitoring runs when somebody remembers to run it.

**Continuous monitoring is the first responsibility**, because five of the eight
controls presuppose that something is watching. A rollback nobody notices the need
for is not a control; an SLO with no measurement is a sentence; a retirement path
with no signal is a plan for a conversation nobody starts. That is why the stage's
question is "can we own it continuously" rather than "is it deployed".

The model flattens all of this into one boolean. `operational_readiness` is a
`bool`, so four controls present and eight controls present are the same input, and
at consequence 2 with a reversible change that single flag decides `production`
against `pilot`. It is the field a team will be tempted to set early, and the
function gives them no reason not to.

### 5 — the rollback is one commit, and the receipt has to name the dependents

The pilot's unit here is a lesson and the rollback is `git revert` of its single
commit, so both halves of a receipt are computable.

Reverting lesson 47: commit `f77cab9`, **9 files**, all but the generated README
inside that lesson's directory. It looks local. It is not — **7 solutions** in
lessons 48, 49, 51 and 52 name that directory and read its files, so the revert
takes their measurements with it.

The shape of the dependency matters as much as the count. The two earliest finished
lessons have 7 dependents each; the last five have 0 between them — 29 edges across
the ten finished lessons, all pointing backwards. Reverting an early lesson is
expensive and reverting a late one is free, which is the opposite of how a commit
log reads, and nothing in the stage plan records it.

`required_controls("pilot")` names "rollback" and "audit trail"; neither is a record
of what a rollback would *cost*, and `plan` returns three keys, so the receipt
lives outside the document that requires it. Recording the commit is one line.
Recording the dependents is one pass over the tree. The second half is the one that
decides whether the rollback is a revert or a project, and it is the half nobody
writes.
