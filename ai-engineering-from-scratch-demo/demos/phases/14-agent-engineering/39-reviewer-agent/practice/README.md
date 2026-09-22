<!-- generated:start -->
# 14-agent-engineering / 39-reviewer-agent

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/39-reviewer-agent/) · upstream spec
`phases/14-agent-engineering/39-reviewer-agent/docs/en.md`

```bash
uv run demo practice run 39-reviewer-agent --ex 1
uv run demo explain 39-reviewer-agent --ex 1
uv run pytest demos/phases/14-agent-engineering/39-reviewer-agent
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a sixth dimension specific to your product domain. Defend why it is not absorbed by the e… | code | T0 | `ex01_the_five_dimensions_score_a_forked_copy_ten_out_of_ten.py` |
| 2 | Run the reviewer with two different system prompts (terse, verbose). Which produces a report… | code | T0 | `ex02_a_terse_report_that_prints_only_the_total_inverts_the_verdict.py` |
| 3 | Add a `confidence` field per dimension. Refuse to ship the report when confidence in the lowe… | code | T0 | `ex03_the_dimension_the_reviewer_exists_for_is_the_one_it_cannot_be_sure_of.py` |
| 4 | Build a calibration set: 10 historical task close-outs with known correct verdicts. Run the r… | code | T0 | `ex04_the_calibration_set_agrees_wherever_the_evidence_is_not_in_the_diff.py` |
| 5 | Add a "request more evidence" affordance: the reviewer can ask the builder for a specific tes… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

### 1 — the five dimensions score a forked copy ten out of ten

"Not absorbed" is a testable claim. Two diffs that differ only in the proposed
dimension must score identically on the existing five; if any of the five moves,
the dimension is already covered. This repository's domain rule is `DESIGN D5` — a
solution imports the lesson's own `code/`, it never forks it — so the sixth
dimension is **reference fidelity**, scored 0 when the diff copies definitions out
of the reference module, 2 when it imports them.

Run both diffs through `review` and they come back identical: 10/10, `pass`, same
five notes, on a fork that duplicates 143 lines of the lesson's module. Nothing in
the rubric can see it. That is the proof the exercise asks for, and it is not a
near miss — it is a perfect score for the exact failure the rule exists to prevent.

The dimension also cannot be *computed* from what the reviewer is given.
`diff_summary` is typed `dict[str, list[str]]` — file names — and 0 of the 5
scorers reads a file's contents. Adding the dimension means widening
`ReviewerInputs` first, the same shape Lesson 38 hit when an off-scope exemption
needed a timestamp the scope report never carried. On real artifacts the check is
cheap: the four shipped Lesson 38 solutions all call `parity.load_reference`, hold
zero copied definitions, and score 2/2.

One thing the measurement turned up on the way past: `score_problem_fit` counts
goal words against file *names*, so the lesson's own clean case
(`app/signup.py`, `tests/test_signup.py`) scores 1/2 while a single empty file
named `signup_validation_input.py` scores 2/2. That is verbosity bias with the
model taken out — the rubric rewarding the label over the work.

### 2 — a terse report that prints only the total inverts the verdict

With deterministic scorers there is no system prompt to vary, so the honest
reading is that the "prompt" is the renderer: the same `ReviewReport` written two
ways. Verbose — every dimension with its note — is 12 lines and 322 characters.
Terse — verdict, total, and only the dimensions scoring below 2 — is 2 lines and
74 characters, a 6.0x cut, and the entire difference is the dimensions that passed.

Terse wins, and the measurement says why rather than asserting it: on the clean
demo the verbose report is 12 lines of which 0 name something to do. Lines spent on
a green run are what teach a reviewer to skim the next one.

But terse has a floor, and it is lower than "print the total". `review` tests
`any(d.score == 0)` *before* it compares the total, so a run scoring 2/2/2/2/0
hard-fails at **8 points** while a run scoring 2/1/2/1/1 passes at **7**. A summary
carrying `total` and `verdict` and nothing else reads as a bug in the tool. The
zeroed dimension is not a detail the terse rendering may drop; it is the only thing
that explains the verdict.

The other half of the case against verbose: 3 of the 5 notes are counts —
"keyword hits across touched files: 3", "off-scope warnings: 0", "1 assumptions
recorded" — numbers the reader could have read off the artifacts. Of the two
qualitative notes, one is the scorer saying it cannot tell. Verbose buys length,
not judgment.

### 3 — the dimension the reviewer exists for is the one it cannot be sure of

Confidence has to come from somewhere mechanical or it is a second number the
scorer invents. The honest source is the branch taken: reading an exit code or a
`findings` list is a fact (1.0); inferring intent from a file name, or landing in a
branch whose own note says the scorer cannot tell, is not (0.5).

Attach that and the rule in the exercise refuses to ship **3 of 3** runs, including
the clean one. `min(confidence)` is 0.5 every time, because `problem_fit` is
derived from file names on every input the reviewer can be given. A confidence
floor of 0.6 does not filter low-signal reports here; it turns the reviewer off.

The dimension that is never confident is the one the reviewer exists for. The
lesson's framing is that acceptance cannot ask "did this solve the right problem" —
`problem_fit` *is* that question, and it is answered by matching three goal-derived
keywords against file names. Dropping it to clear the threshold deletes the reason
to run a reviewer at all.

Two structural notes. `score_assumptions` returns 1 for two opposite worlds and
says so in its own note: "no assumptions recorded; either work was trivial or
undocumented" — a scorer publishing its confidence in prose because
`DimensionScore` has nowhere to put it. And there is no channel to refuse in:
`ReviewReport` has 4 fields, `review` emits 3 verdict literals, and none of them
means "no verdict". Refusing *per dimension* keeps one — drop `problem_fit` and the
clean case rescores 8/8, still a pass, with the unanswerable question named instead
of silently worth two points.

### 4 — the calibration set agrees wherever the evidence is not in the diff

Ten close-outs, each labelled from its history ("deleted the failing test to get
the suite green" → `hard_fail`) before any artifacts were encoded, so the set
measures the rubric and not itself.

**Agreement: 5 of 10.** Half, against the 80% bar the lesson sets for shipping a
rubric. The five disagreements are `H-05`, `H-07`, `H-08`, `H-09`, `H-10`, and all
five run the same direction: the rubric is more lenient than the human record.
Zero come back stricter. A calibration set that only surfaces false passes is
telling you the rubric has no way to say "I did not see this" — which is exactly
what Exercise 3 measured as confidence.

Where it disagrees, in three groups:

**The diff content it never reads** (`H-07`, `H-09`, `H-10`). The deleted failing
test and the rename-only change both score **10/10, pass**. One because
`score_verification` reads exit codes and a deleted test exits 0; the other because
`score_problem_fit` reads file names and the rename made them match the goal
perfectly. These are the adversarial moves the rubric invites, and it hands them
full marks.

**The gate's verdict it never reads** (`H-05`). No scorer touches
`verdict["passed"]`, and `score_scope_discipline` looks at 2 of the gate's 8
codes. A close-out the gate blocked on `acceptance.missing` scores 9/10, `pass`.
`score_verification` helps it along: an empty feedback log falls past
`all(code == 0 ...) and exits` into the middle branch and scores 1 with the note
"mixed exit codes in feedback", for a log containing no exit codes.

**The resolution of the scale itself** (`H-08`). A correct fix whose session ended
with the task still open scores `handoff_readiness` at 1 and totals 9 — comfortably
`pass`, where the record says `soft_fail`. The pass bar is 7 of 10, so one weak
dimension cannot fail a run; it takes three. The rubric is coarser than the
judgments it is being calibrated against, and no amount of prompt work fixes a
threshold problem.

### 5 — "request more evidence", and the back-off that stops it looping

The affordance is worth having, but the termination argument has to come from the
shape of the request, not from a retry counter. Three rules, in order.

**One request per dimension, and the request must name the dimension it
unblocks.** There are five dimensions, so there are at most five requests in a
task. That is the whole back-off: the budget is a finite set the reviewer is
walking, not a number someone picked. A second request for a dimension already
requested is a bug, not a retry — the reviewer already had its turn and must now
score with what it has.

**The reply is evidence or a refusal, and both end the request.** If the builder
runs the command and the dimension still cannot be scored, that is a confidence
value (Exercise 3), not another request. This is where the loop would otherwise
live: "still unclear, ask again" has no fixed point. Scoring with recorded low
confidence does.

**No request may change the artifact under review.** The doc's rule here is
**The reviewer cannot edit the diff** — the reviewer reads, it writes a report, and if
the report says "fix this," the next builder turn does the fix. A request phrased
as "run the tests with this null check added" is a patch instruction wearing an
evidence request's clothes: it edits the diff and then reviews the edited thing.
The admissible request is a *read* of state that already exists — run this existing
command, show me this file, paste this log — and the inadmissible one is anything
whose answer requires new code. That line is also what keeps the back-off
well-founded: reads over a fixed artifact terminate, and a build-measure-build
cycle does not.

Two notes from the measurements. First, the evidence the reviewer actually needs
here is mostly *not* a test run: of the five calibration disagreements, `H-09` and
`H-10` turn on diff content and `H-05` on the gate's own verdict, all three
available without asking the builder for anything. Exercise 1 found the root cause —
0 of the 5 scorers read file contents. A request affordance that fetches test runs
would fix none of them; widening `ReviewerInputs` fixes all three. Ask for evidence
the inputs genuinely cannot carry, and fix the inputs for everything else.

Second, the request has a cost the rubric should book against itself. Each round
trip is a builder turn, and the builder is the party whose work is under review; a
reviewer that can spend the builder's turns has an incentive to spend them instead
of committing to a score. Capping requests at one per dimension bounds that at five
turns per task, which is a number a human can see on the report and object to.
