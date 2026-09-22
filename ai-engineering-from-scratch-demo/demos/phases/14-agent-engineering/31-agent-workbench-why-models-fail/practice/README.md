<!-- generated:start -->
# 14-agent-engineering / 31-agent-workbench-why-models-fail

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/31-agent-workbench-why-models-fail/) · upstream spec
`phases/14-agent-engineering/31-agent-workbench-why-models-fail/docs/en.md`

```bash
uv run demo practice run 31-agent-workbench-why-models-fail --ex 1
uv run demo explain 31-agent-workbench-why-models-fail --ex 1
uv run pytest demos/phases/14-agent-engineering/31-agent-workbench-why-models-fail
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Pick a repo where you already run an agent. Score the seven surfaces from 0 (missing) to 2 (h… | code | T0 | `ex01_this_repo_scores_ten_of_fourteen_and_handoff_is_zero.py` |
| 2 | Extend `main.py` so the prompt-only run also produces a fake "success" claim. Verify the veri… | code | T0 | `ex02_both_runs_already_declare_success_and_only_one_earns_it.py` |
| 3 | Add an eighth surface for your own product. Justify why it does not collapse into one of the… | code | T0 | `ex03_budget_is_the_eighth_surface_because_a_correct_run_can_cost_anything.py` |
| 4 | Re-run the script with a different stub agent that hallucinates an extra file write. Which su… | code | T0 | `ex04_scope_catches_it_first_and_verification_never_does.py` |
| 5 | Map the five industry-recurring failure modes from Phase 14 · 26 onto the seven surfaces. Whi… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

### 1 — this repo scores 10 of 14, and handoff is the weakest surface

The repo an agent actually runs in, for these files, is the one they live in, so
the scoring is done against `ai-engineering-from-scratch-demo` with evidence a
reader can check. Each surface gets 0 when nothing implements it, 1 when something
does but nothing enforces it, and 2 when a script the build runs will fail on it.

| surface | score | evidence |
|---|---|---|
| instructions | 2 | `DESIGN.md`, enforced by `scripts/audit_practice.py` (D10, D12, D14) |
| state | 1 | `practice.yaml` per lesson, read at build time, never written by a run |
| scope | 2 | one file per exercise, index-checked by the audit |
| feedback | 2 | `practice.selfcheck` prints every failing check's measured value |
| verification | 2 | `pytest` plus `scripts/check_deps.py` |
| review | 1 | `scripts/coverage.py` reports; nothing gates on it |
| handoff | 0 | no file, no script, no convention |

Total 10/14, weakest **handoff**.

The scoring rule turns out to be a count of *enforcement*: the four surfaces
scoring 2 are exactly the ones with a script that fails the build, and the two
scoring 1 have an artifact with no gate. A surface nobody can fail the build on is
documentation. Handoff's zero is the honest one — a session that ends mid-lesson
leaves the next one to re-derive where it was from `git log`, which is a real cost
this repo pays and has no artifact to point at.

Two things about the demo itself fall out. `stub_agent` branches on `scope`,
`state`, `verification` and `feedback`; `instructions`, `review` and `handoff` are
in `WORKBENCH_SURFACES` and never read, so passing them changes zero result
fields — three of the seven surfaces are labels. And `state` is the one surface
whose absence changes nothing observable: removing it alters one note and no
fields, so a run scored on `failure_report` cannot tell a stateful session from a
stateless one. That is precisely the surface whose failure is free in this run and
expensive in the next, which is why it is hard to score honestly.

### 2 — the fake success claim already ships; the gate is what is missing

`stub_agent` sets `declared_success = True` in both branches, so the exercise's
first half is already done. Both runs claim success; only the workbench run
reports `actually_passing=True`.

What is missing is the gate. `has_verification` *assigns* `actually_passing =
True` — the branch reads zero of the task's fields, not `acceptance`, not
`allowed_files`, not `forbidden_files`. Verification in the demo causes passing
rather than checking it, and the test is decisive: hand it a run that touched zero
files and it still reports `actually_passing=True`. A gate that cannot fail is not
a gate.

A real gate is a function, deterministic over inputs, that fails closed. Written
over the three inputs `RunResult` already carries — acceptance criteria, files
touched, whether the command ran — it returns False for the prompt-only run and
True for the workbench run: two runs, two correct verdicts. The prompt-only run
fails on two of the three, and the interesting one is *not* the acceptance file:
acceptance names `test_app.py` and the prompt-only run does touch it, so file
coverage alone passes. It is `tests_run=False` and the two forbidden writes that
sink it. Checking only that the right files were touched would have let the fake
claim through.

The last observation is the one to carry into production: `failure_report` returns
seven keys including both `declared_success` and `actually_passing`, so the
divergence is visible *here*. The agent's own output is the claim. A caller that
logs only `declared_success` sees two successes where there is one, which is
exactly Lesson 26's success-hallucination mode with the evidence sitting one field
away.

### 3 — the eighth surface is budget, and it earns its place by refusing

A surface earns its place when there is a run that all seven existing surfaces
accept and it refuses. Budget — a declared ceiling on steps, tool calls and
tokens, checked per step and failing closed — clears that bar.

The run: stays inside `allowed_files`, runs its tests, passes acceptance, writes a
handoff note, and takes **412 steps against a 6-step plan**. It passes 7 of 7
existing surfaces and fails a ceiling of 60. That is 68.7x over plan, with zero
other refusals — nobody would ship it, and nothing in the seven says so.

The three collapse arguments, taken seriously:

*Into scope?* No. Scope is an authorization policy over paths — it answers
*where*. The over-budget run touches exactly the two allowed files and zero
forbidden ones, so an ACL has nothing to object to. Widening scope to cover cost
would mean keying a permission on a number nobody knows until the run ends, which
is a quota, not a permission. Different primitive.

*Into verification?* No, and this is the ordering argument exercise 4 makes
separately. Verification is triggered on task close; by then 412 steps are spent.
It can report the overrun and cannot prevent it. Budget has to be a trigger on
*every step*, which puts it at position 1 alongside scope rather than at the end.

*Into feedback?* This is the closest call, and it is a clean split. Feedback
records every invocation — the data a budget needs is already there, all 412
entries. What feedback lacks is a ceiling and the authority to stop the worker.
In the lesson's own primitive vocabulary, feedback is a queue and budget is a
function over that queue with a kill. A queue is half of a quota.

Worth noting what budget is *not*: it is not observability. Lesson 28 measured
cost and accuracy disagreeing, and Lesson 25 found debate costing 2.5x for an
identical answer. Both are visible in traces and neither stops anything. The
surface is the refusal.

### 4 — scope catches it first, and verification never sees it

"First" is an ordering question, so the surfaces go in the order a run reaches
them: scope before the write, feedback during, verification at close, review
after. A stub that adds `config/prod.yaml` to an otherwise correct run is refused
by **scope at position 1 of 4**. Feedback records the write without judging it,
the shipped verification branch returns `actually_passing=True` regardless, and
review would flag it at position 4 — after the file exists.

That ordering is the useful output. Stopped at scope, the repo has the two files
that were allowed; stopped at review, it has three. Both surfaces "catch" the same
write, and only one leaves nothing to revert. Prevention and detection are
different *positions*, not different strengths, and a workbench that reports "we
caught it" without saying where is hiding the difference between a refused call
and a rollback.

Two defects in the reference worth naming. `failure_report`'s `off_scope_writes`
compares against the hardcoded set `{"app.py", "test_app.py"}` rather than
`task.allowed_files`. On this task it happens to be right; on a task whose allowed
files are `["src/api.py"]` it reports the legitimate write as off-scope alongside
the real violation. The check is correct for exactly one task, which is the kind of
bug that survives every run of the demo.

And of the report's seven keys, exactly one (`off_scope_writes`) is derived from
observing what happened; the other six are the agent's own account of itself. A
hallucinated write is invisible to any surface that reads the `RunResult` instead
of the filesystem — which is why scope, feedback and review all have to be
implemented against the world, and why "the agent said it stayed in scope" is not
a scope surface.

### 5 — the five failure modes, mapped onto the seven surfaces

**Reasoning from primitives, not from vendor taxonomies** is the section that makes
this mapping tractable: each surface reduces to a distributed-systems primitive —
instructions to policy plus function metadata, state to session persistence, scope
to authorization policy, feedback to an invocation log in a queue, verification to
a function triggered on task close, review to a separate worker with read-only
authz, handoff to a durable record emitted by a session-end trigger. Failure modes
map onto surfaces cleanly once you ask *which primitive was missing*.

Phase 14 · 26's five industry-recurring modes:

**Hallucinated actions → scope.** The agent invokes a tool that does not exist or
fabricates arguments. Scope is the authorization policy: an allowlist refuses a
call to a function not on it, before execution. Exercise 4 is this mode in
miniature — a write nobody asked for, refused at position 1. The surface that
*detects* it afterwards is review; the surface designed to absorb it is scope,
because the primitive is a permission check and permission checks run first.

**Scope creep → scope, with instructions as the upstream half.** The agent expands
the task beyond the ask. This is the mode the surface is named for, and Lesson 26
found the shipped detector gating on whether the *request text* contained a write
verb rather than on what the agent did. Scope as an ACL over paths and approvals
is the enforceable version. Instructions carry the half scope cannot: "do not open
extra PRs" is policy attached at startup, not an ACL entry.

**Cascading errors → feedback and verification, in that order.** One wrong call
triggers downstream effects. Feedback is the invocation log that makes the cascade
*visible* — Lesson 26's cascade-radius exercise needed exactly that log, and found
the shipped detector discarding the radius it computed. Verification is what
*stops* it, because it fails closed at task close. Neither is sufficient alone: the
log without a gate is Lesson 24's "expensive logging", and the gate without the log
cannot say which step went wrong.

**Context loss → state, with handoff for the cross-session case.** Long-horizon
tasks forget early-turn constraints. State is session persistence — the keyed store
the runtime re-reads at every step — and handoff is the durable record that carries
a constraint across a session boundary. This is the pair this repo scores worst on
(1 and 0 in exercise 1), and it is not a coincidence: both surfaces' failures are
invisible in the run that causes them and expensive in the next one.

**Tool misuse → verification, with instructions upstream.** Right tool, wrong
arguments. Verification as a deterministic function over inputs is the gate;
Lesson 06's argument validation is its concrete form. Instructions carry the
function metadata that makes the right call likelier, and Lesson 27's measurement
is the warning about relying on it — a validator with 55% false positives gets
turned off, so the gate has to be precise before it is strict.

**What the mapping leaves over.** Two of the seven surfaces absorb none of the five
modes directly. **Review** is a second worker reading the first's artifacts — it
catches everything late and nothing early, which exercise 4 priced as the
difference between a refused call and a revert. And **handoff** only matters across
sessions, which none of the five modes is scoped to. That asymmetry is itself the
finding: the modes were catalogued from single-run traces, so a taxonomy built
from them will systematically under-weight the two surfaces whose failures show up
in the *next* run. Exercise 1's score on this repo — handoff at 0 — is what that
blind spot looks like from inside.
