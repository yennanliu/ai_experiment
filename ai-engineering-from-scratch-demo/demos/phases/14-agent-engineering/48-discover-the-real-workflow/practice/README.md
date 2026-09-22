<!-- generated:start -->
# 14-agent-engineering / 48-discover-the-real-workflow

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/48-discover-the-real-workflow/) · upstream spec
`phases/14-agent-engineering/48-discover-the-real-workflow/docs/en.md`

```bash
uv run demo practice run 48-discover-the-real-workflow --ex 1
uv run demo explain 48-discover-the-real-workflow --ex 1
uv run pytest demos/phases/14-agent-engineering/48-discover-the-real-workflow
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Reconstruct one workflow from a log without interviewing anyone. | code | T0 | `ex01_a_workflow_rebuilt_from_the_log_can_never_be_grounded.py` |
| 2 | Interview a user and mark every claim that still lacks direct evidence. | code | T0 | `ex02_four_reported_claims_and_the_log_corroborates_two.py` |
| 3 | Add one authority boundary and one failure-recovery step. | code | T0 | `ex03_the_recovery_step_is_a_loop_and_the_model_is_a_list.py` |
| 4 | Model two workflow variants without merging them. | code | T0 | `ex04_the_two_variants_differ_at_one_step_and_merging_them_breaks_the_order.py` |
| 5 | Identify a proposed feature that removes a visible step but leaves hidden work untouched. | code | T0 | `ex05_the_scaffold_removes_the_typing_and_leaves_the_measuring.py` |
<!-- generated:end -->

## Answers

### 1 — a workflow rebuilt from the log can never be grounded

The log available here is this repository's git history plus the artifacts each
commit left behind. Reconstructed from it, the per-lesson workflow is eight steps:
scaffold the manifest, write one solution per exercise, run each until its checks
pass, fill the manifest's threshold, regenerate the lesson README, write the
answers by hand, commit the lesson as one change, regenerate the coverage table.

Every step is supported by something checkable — a path in the tree or a commit in
the log — and 8 of 8 receipts resolve. And `audit` calls the result
**`needs-evidence`**, because `direct_evidence_ratio` is `0.0`. Artifacts are not
direct behaviour, and the status rule is `not issues and direct > 0`. A workflow
reconstructed entirely from a log — which is exactly what this exercise asks for —
cannot reach "grounded" by construction.

That follows from a flattening: the docs rank evidence on four rungs (direct
behaviour, artifact, reported, inference) and `Evidence.direct` is a `bool`. Three
of the four rungs store identically, so an incident log and somebody's hunch are
the same value, and the ratio that drives the status cannot separate them.

Two smaller notes. Reconstruction from a log buys addresses: 7 steps point at a
file that exists and 1 at a commit, so every claim can be re-checked by someone who
doubts it. And `confidence` is stored and never used — dropping all eight to 0.01
leaves the ratio and status unchanged, because `audit` range-checks the number and
then ignores it.

### 2 — four reported claims, and the log corroborates two

The "interview" available here is the instruction that started this work. Four
claims come out of it, and the repository can be asked about each:

| Claim | Corroboration | Verdict |
|---|---|---|
| one commit per completed lesson | 8 recent lesson commits, 8 distinct lesson numbers | corroborated |
| push after each lesson rather than at the end | the branch is on its remote | corroborated |
| the pull request is where this lands | nothing in this repository | reported |
| a reviewer reads the answers before the code | nothing in this repository | reported |

**2 of 4.** Recording the corroboration moves `direct_evidence_ratio` from 0.0 to
0.5 — the mechanism already exists; what the interview adds is the discipline of
writing the claim down *before* going to look for the artifact, so the ones that
find nothing stay visible.

The corroborated claims are also narrower than the sentences they came from.
"Commit and push once a lesson is completed" is supported only as "one commit whose
subject names one lesson". Whether the lesson was *completed* is a different check —
that lesson's own audit — and collapsing the two is how a reported claim quietly
becomes a requirement.

Finally, `audit` returns five keys and a single ratio. A reader gets one number for
the whole workflow and has to walk `steps` to find which two claims are unsupported,
which is the opposite of the lesson's advice to keep uncertain claims visible.

### 3 — the recovery step is a loop, and the model is a list

The authority in this workflow is the audit: the thing allowed to refuse a lesson.
It fits, but only by being the *actor* of a step — the same trick the lesson's own
example uses for the incident commander — because of the four fields the docs' step
table asks for and `WorkflowStep` lacks: **authority, trigger, input, output**. Five
fields against the documented eight.

The recovery step does not fit at all. "The audit refuses" and "the author rewrites
the refused solution" is a return to a step that already happened. Written down,
the orders read `[1, 2, 3, 4, 9, 2]` and `audit` reports "workflow order must be
contiguous from one". A workflow with a loop is a graph; `order` is an integer in a
list. Exceptions and recovery — the two places the lesson says AI features fail —
are the two things this model cannot express.

The loop is not hypothetical. Lesson 46's fifth solution spells a banned marker at
run time, with the comment "the audit refuses the literal": one file in this
repository carrying the scar of a refuse-and-rewrite cycle, which is direct
evidence of a step the happy path does not contain.

And the loop has no cost in the model. `friction` is prose, there is no attempt
count, no duration, no retry number — so a step that ran once and a step that ran
four times are indistinguishable in the JSON.

### 4 — the two variants differ at one step, and merging them breaks the order

The variants are real: 42 of this phase's 54 lessons ship documentation in two
languages and 12 ship English only, so the scaffolding step behaves differently.
The bilingual path reads `en.md` and `zh.md`; the English-only path reads one file
and takes the scaffolder's missing-file branch.

Audited separately, each returns zero issues. Concatenated into one list, the
orders read `[1, 2, 3, 4, 1, 2, 3, 4]` and the audit reports an ordering issue.
There is no way in this model to say "these are two paths" — a variant is only
expressible as two separate documents.

The population split is what makes averaging wrong rather than merely imprecise.
This is not an edge case or an expertise difference; it is a policy that changed
partway through the curriculum. An averaged workflow would describe reading **1.8**
documents at step 2, which is a step nobody performs.

The tempting fix is worse than the problem: renumbering the second variant to 5–8
makes the audit pass with zero issues, at the cost of asserting that eight steps
happen in sequence. A clean report bought by making the model wrong is exactly what
the lesson means by averaging away disagreement — and note that `audit` returns no
field naming a variant, so a reader has to diff two `steps` arrays to notice.

### 5 — the scaffold removes the typing and leaves the measuring

The feature already shipped, which is the best kind of example:
`scripts/scaffold_practice.py` writes the manifest and the stubs, so nobody creates
a practice file by hand.

It writes every field except the one nobody can generate. The template emits a
placeholder for `verifies:` — the threshold the answer asserts — and across the five
most recently finished lessons, **0 of 5** manifests still carry it. Twenty-five
exercises, twenty-five thresholds, each written by hand *after* running the answer
and reading what it measured.

The system's response to that hidden work is to block it, not to do it: the
placeholder is one of the audit's banned strings, so a manifest that still has it
cannot ship. That is the right design and it is not the same as removing the work.

Modelled as two workflows, the pre-scaffold path has 5 steps and the scaffolded
path has 4 — and both carry the same 2 friction entries, attached to the steps they
share ("the threshold is only knowable after the first run"). The feature deleted a
step that had no friction on it. It changed the count, not the experience.

The last part is the one worth keeping: both workflows report
`direct_evidence_ratio` 1.0 at status `grounded`. The artifact that exists to
describe the workflow reports identical numbers before and after the feature. What
changed is the step list itself, which a reader has to read rather than compare —
so a feature evaluated by this audit's summary would look like it did nothing, and
a feature evaluated by the step count would look like it did more than it did.
