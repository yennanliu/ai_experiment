<!-- generated:start -->
# 14-agent-engineering / 47-outcomes-before-output

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/47-outcomes-before-output/) · upstream spec
`phases/14-agent-engineering/47-outcomes-before-output/docs/en.md`

```bash
uv run demo practice run 47-outcomes-before-output --ex 1
uv run demo explain 47-outcomes-before-output --ex 1
uv run pytest demos/phases/14-agent-engineering/47-outcomes-before-output
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Rewrite a feature request from your backlog as an outcome frame. | code | T0 | `ex01_the_request_that_started_this_repository_names_its_output_twice.py` |
| 2 | Add one constraint that changes which solutions remain possible. | code | T0 | `ex02_the_determinism_constraint_removes_the_solutions_the_lesson_would_reach_for.py` |
| 3 | Add two non-goals that keep the first slice small. | code | T0 | `ex03_every_file_the_phase_shipped_is_inside_a_practice_directory.py` |
| 4 | Identify the earliest observation that would disprove the desired outcome. | code | T0 | `ex04_the_next_question_asks_for_confirmation_and_the_frame_has_no_field_for_doubt.py` |
| 5 | Write three different outputs that could satisfy the same outcome. | code | T0 | `ex05_three_outputs_satisfy_the_outcome_and_one_constraint_picks_between_them.py` |
<!-- generated:end -->

## Answers

### 1 — the request that started this repository names its output twice

The backlog item rewritten here is the one that produced the files around this
answer: *"complete all of the lessons in phase 14, commit and push once a lesson is
done."* Both clauses are outputs — a unit of work and a delivery mechanism — and
neither says who is better off.

The frame:

- **User.** A reader working through a phase 14 lesson.
- **Situation.** After reading the lesson and running its code, facing its exercises.
- **Current behavior.** Reads the exercise, guesses at an answer, and has no way to
  tell whether the guess matches what the lesson's code does.
- **Desired outcome.** A reader can run an answer to every exercise and see it check
  its own claims against the lesson's own code.
- **Constraints.** Answers import the lesson's code rather than forking it; the same
  command produces the same verdict on any machine; no dependency outside the
  standard library for a T0 lesson.
- **Non-goals.** Rewriting the lesson text; translating anything; changing the
  reference curriculum.

It validates with zero issues and never mentions a practice file — which is the
test, since a solution file is one of several outputs that could satisfy it
(Exercise 5).

The leak check underneath is weaker than it looks. `proposed_output` defaults to
`""` and the rule is `proposed_output.lower() in desired_outcome.lower()`, so a
frame that leaves the output blank cannot fail no matter what the outcome names —
the original request, framed with a blank output, returns no issues while naming
the artifact outright. And the substring test misses paraphrase: "triage with the
assistant" against a proposed "incident assistant" is clean, while "use the
incident assistant to triage" is flagged. It tests spelling, not leakage.

### 2 — the determinism constraint removes the solutions the lesson would reach for

A constraint earns its place by deleting options, so the measure is how many
candidate outputs stop being available.

| Candidate output | Depends on | Survives |
|---|---|---|
| time the run and report the speed-up | wall clock | no |
| ask a model to grade the answer | sampling | no |
| fetch the upstream file and diff it | network | no |
| import the lesson's code and assert on what it returns | the repository | yes |
| compare output against a checked-in fixture | the repository | yes |
| count the artifacts the lesson produces | the repository | yes |

**3 of 6 deleted.** And the constraint is visible in the work that followed it: the
15 answers the three most recent lessons ship import `time`, `random` or `urllib`
zero times between them.

The constraint also converts the outcome from a claim into a test. "A reader can
run an answer and see it check its own claims" is satisfied by an answer that
passes on the author's laptop; with determinism attached, the sentence has a
procedure — grade every answer twice and compare the verdicts. They agree 15 of 15.

Two notes on the artifact. `validate` reads `frame.constraints` exactly once, in
the emptiness test, so a frame with three real constraints and one carrying
`["be good"]` are equally valid; whether a candidate violates a constraint is a
judgment the frame records and the program cannot make. And a constraint that
deletes nothing is a preference: adding "prefer clear names" leaves all six
candidates available, and the lesson's own examples (no production writes, no new
runtime dependency) are all of the deleting kind.

### 3 — the non-goals held in five of six commits

Two non-goals that a diff can cross: **do not translate anything**, and **do not
touch the tooling**.

Checked against the tree: the finished lessons put **80** files in this repository
and **0** of them sit outside a `practice/` directory. The one crossing is a single
file the slice does not own — `scripts/scaffold_practice.py` — which still carries
the `except FileNotFoundError` branch the phase needed.

The crossing was correct. The scaffolder raised `FileNotFoundError` on lessons that
ship English docs only, which is every lesson from 43 onward, so shipping anything
at all required the edit. That is exactly what a non-goal is for: without it the
extra file is an unremarked line in a diff; with it, it is a decision that has to
be named in the commit message. A non-goal that is never crossed and a non-goal
that is crossed silently are both failures of the same mechanism.

The translation non-goal is the larger of the two. The reference phase carries 42
Chinese documents, and the generated manifests carry 212 `zh:` blocks. Declaring
translation a non-goal removes that surface from the slice in one line rather than
deferring it file by file — which is the difference between a boundary and a
backlog.

And as with constraints, the artifact only counts: `validate` reports "non-goals
are empty" and nothing else, so two concrete non-goals and the single word
"nothing" are equally valid. The check that matters — does the diff stay inside
them — is the `git status` receipt built in Lesson 43, and it lives in the gate.

### 4 — the next question asks for confirmation, and the frame has no field for doubt

"What would disprove this?" and "what would show it was achieved?" are different
questions, and the lesson's artifact asks only the second: `next_question` is one
template reading *"What evidence would show that the desired outcome was achieved
for the {user}?"*. `decision` returns four keys, `OutcomeFrame` has seven fields,
and none of them is a falsifier.

**The earliest disproof: one graded answer whose checks fail against its lesson's
code.** Not a survey, not a reading — a file, a check, and a measured value.
Graded now, the three most recent lessons return 15 passes and 0 failed checks, so
the observation has been looked for and not found, which is the only form a passing
disproof takes.

Two things follow from naming it. The disproof costs 1 file where confirmation
costs 15 — 6.7% — so it is available after the first lesson rather than at the end
of the phase, and a frame that records only the confirming question quietly invites
the opposite schedule. And because the 15 answers carry 60 individual checks, the
outcome has sixty chances to be wrong rather than one impression to argue about.

### 5 — three outputs satisfy the outcome, and one constraint picks between them

Three outputs for the same outcome:

1. **Imported answers** that assert on the lesson's own functions.
2. **A fixture-diff harness** comparing saved output against a checked-in file.
3. **A quiz** with worked answers in prose.

All three validate against the frame with zero issues, which is the property that
makes the outcome an outcome: it does not contain its own solution.

Scored against the frame's three constraints they read **3, 2, 2**. The deciding
clause is "answers import the lesson's code", and it eliminates two candidates for
a reason that is written down: the fixture harness's fixtures drift silently the
moment the lesson's code changes, and the quiz never executes anything. A
constraint that eliminated only one candidate would not have been worth writing.

Two observations about the machinery. The leak check fires on whichever output is
*named* in the sentence — naming each candidate flags 3 of 3 frames, leaving the
outcome alone flags none — so it protects the sentence, never the choice. And
nothing in the artifact records why the survivor won: `OutcomeFrame` has 7 fields,
`decision` returns 4 keys, and neither holds the alternatives considered or the
constraint that removed them. The reasoning that makes this choice reviewable lives
outside the thing the lesson says to keep.
