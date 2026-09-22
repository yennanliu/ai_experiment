<!-- generated:start -->
# 14-agent-engineering / 41-workbench-for-real-repos

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/41-workbench-for-real-repos/) · upstream spec
`phases/14-agent-engineering/41-workbench-for-real-repos/docs/en.md`

```bash
uv run demo practice run 41-workbench-for-real-repos --ex 1
uv run demo explain 41-workbench-for-real-repos --ex 1
uv run pytest demos/phases/14-agent-engineering/41-workbench-for-real-repos
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a sixth outcome: time-to-first-meaningful-edit. How do you measure it cleanly? | code | T0 | `ex01_the_workbench_reaches_its_first_edit_one_step_later_and_that_is_the_number.py` |
| 2 | Run the comparison on a real second-day task in your codebase. Where do the workbench numbers… | code | T0 | `ex02_a_real_second_day_task_slips_on_the_two_outcomes_this_repo_has_no_surface_for.py` |
| 3 | Add a "false negative" pass: tasks where prompt-only would have been faster and the workbench… | code | T0 | `ex03_the_overhead_is_fixed_so_the_false_negatives_are_the_small_tasks.py` |
| 4 | Replace the scripted "agent" with a real LLM call. Which outcomes get noisier? | explain | T0 | prose, below |
| 5 | Author a one-page summary aimed at a non-engineer. What survives the cut? | code | T0 | `ex05_what_survives_the_cut_is_the_two_outcomes_with_units_a_reader_owns.py` |
<!-- generated:end -->

## Answers

### 1 — the workbench reaches its first edit one step later, and that is the number

"Cleanly" rules out the wall clock. Both pipelines here are scripted, so a timer
measures the machine; on a real run it measures sampling speed and the network,
neither of which is the thing under study. The clean unit is the pipeline's own
step list, which the lesson publishes, and the clean definition of "meaningful" is
an edit to a file inside the scope contract.

Counted that way: prompt-only reaches its first edit at **step 3 of 4** (75% of the
run spent before the keystroke), the workbench at **step 4 of 8** (50%). One extra
step of setup. The number that actually matters is on the other side of the edit —
prompt-only has 1 step after it ("claim done"), the workbench has 4, and all four
are checks.

Three things the measurement forces:

**"Meaningful" has to mean in-scope.** Prompt-only touches `sample_app/app.py`,
`README.md` and `sample_app/scripts/release.sh`, and 2 of those 3 are outside
`ALLOWED`. An unqualified "first edit" would score a pipeline *faster* for writing
to the release script first — the metric would reward the failure mode.

**There is no clock to read.** The module defines 5 functions, imports no `time`,
and both pipelines return hardcoded `TaskOutcome` literals. A duration field would
record 0.0 twice. A sixth outcome has to be derivable from artifacts, not from the
run.

**`FORBIDDEN` is declared and never read.** It names
`sample_app/scripts/release.sh`, but `files_outside_scope` is computed as
`p not in ALLOWED`, so a block-severity forbidden write is counted as one of two
off-scope files alongside a README edit. The benchmark's own severity distinction
from Lesson 38 does not survive into its scoring.

### 2 — a real second-day task slips on the two outcomes this repo has no surface for

The honest version of this exercise scores a task that already shipped, using the
repository's own artifacts. The task: shipping Lesson 40's practice solutions — the
8 files it put in the tree, every solution graded in-process for the acceptance run.

| Outcome | Benchmark's workbench row | This repo, measured |
|---|---|---|
| `tests_actually_run` | true | true |
| `acceptance_met` | true | true |
| `files_outside_scope` | 0 | **1** |
| `handoff_quality` | "full packet" | **missing** |
| `reviewer_total` | 9 | **6** (`hard_fail`) |

Three of five match. The two that slip slip for the same reason: this repository
has some of the seven surfaces and not others.

**The off-scope file is a build byproduct.** The one path outside the lesson
directory is the top-level `README.md`, which `scripts/coverage.py` rewrites
whenever a lesson lands. A contract made of allowed globs has no way to say "the
build owns this file", so a mechanical regeneration is counted as scope creep —
the same finding Lesson 38's gate produced on the same work.

**The reviewer's `hard_fail` is correct and unmodelled.** Running Lesson 39 over
the real artifacts gives 6/10 with `handoff_readiness` at 0: there is no
`agent_state.json` in this repository, so there is no `next_action` and no closed
task. The benchmark's row hardcodes 9. A real second-day task is scored by
surfaces that either exist or do not, and the benchmark never models their absence.

**The benchmark cannot slip, because both rows are literals.** `run_prompt_only`
and `run_workbench` read zero files and return constants; `main` writes the sample
app *after* the outcomes are computed, so the app is a prop. It is an assertion
formatted as a measurement — which is precisely why the exercise says to run it on
something real.

### 3 — the overhead is fixed, so the false negatives are the small tasks

"Real cost" wants a number, and the workbench's cost is fixed: 7 of its 8 steps are
not edits. So overhead as a share of the run depends only on how many edits the
task needs, which makes false negatives identifiable *before* starting rather than
in hindsight.

| Task | Edits | Prompt-only | Workbench | Ratio |
|---|---|---|---|---|
| run the formatter | 1 | 4 | 8 | 2.00x |
| one-line lint fix | 1 | 4 | 8 | 2.00x |
| bump a pinned version | 1 | 4 | 8 | 2.00x |
| fix a docstring typo | 1 | 4 | 8 | 2.00x |
| where is this symbol defined | 0 | 3 | 7 | 2.33x |
| add signup validation | 2 | 5 | 9 | 1.80x |
| refactor twelve files | 12 | 15 | 19 | 1.27x |

The overhead is real, and it is worst exactly where the task is smallest.

Now the defence, and it is not "run it anyway."

**The saving is conditional on a fact only the skipped steps establish.** Zero of
prompt-only's four steps inspect the diff. "It was only one line" is not something
that pipeline can know — it is what step 6, the scope check, would have told you.
Every false negative is a bet that the task was what it looked like before anyone
measured it.

**On cost alone, "always" does not survive the arithmetic.** Four extra steps on a
one-edit task, against a 12-step cost of being wrong (a redo plus a guided run),
pays off above a 33.3% failure rate. The figures this lesson itself cites straddle
that line — Vercel's agent at 80% success, WebAgent's baseline at 40–50%. An
average-case argument is not strong enough to carry the policy, and pretending
otherwise is how the gate gets disabled by the team that resented it.

**So keep 2 of the 8 steps unconditionally.** The acceptance run and the
verification gate are 25% of the pipeline and they keep `files_outside_scope` and
`acceptance_met` measurable — the first of which Exercise 2 caught slipping on a
real commit. The fast path gives up `handoff_quality` and `reviewer_total` on
purpose. That is a sentence a reviewer can argue with; "we skipped the workbench"
is not.

### 4 — replacing the scripted agent with a real LLM: which outcomes get noisier

The five outcomes do not degrade equally, and the split is not about difficulty —
it is about who computes the number. The doc's table of
**The five outcomes measured** mixes two kinds of row, and a live model separates
them immediately.

**Barely noisier: `tests_actually_run` and `files_outside_scope`.** Both are
computed by the harness from artifacts the model does not write — a feedback record
with an exit code, a list of paths from the diff. The model's variance moves
*whether* a test ran, not whether the measurement is right. Run the same task ten
times and these two are exactly as trustworthy on run ten as on run one. This is
the same property Exercise 1 relied on: they need no clock and no judgment.

**Noisier, and diagnosably so: `acceptance_met`.** The gate compares the commands
that ran against the acceptance commands, and a live model writes the command
string. Lesson 38 already found the failure shape without any model in the loop:
`_acceptance_findings` does `str(rec.get("command"))`, so a list-shaped argv never
matches a string-shaped acceptance line. A live model produces `pytest -q`,
`pytest -x -q`, `python -m pytest` and `uv run pytest` across four runs — all
correct, none equal. The noise is real but it is *string* noise, and normalizing
the comparison removes most of it.

**Noisiest by construction: `reviewer_total`.** It is the only outcome produced by
a model rather than measured from an artifact, so it inherits every bias the lesson
lists on the reviewer side: position, verbosity, self-preference, authority. Lesson
39's calibration set already scored 5/10 agreement with the deterministic scorers;
put a live judge on both sides and the variance stacks — the builder's output
varies, and the grader's reading of it varies independently. Report it with a
spread over repeated runs or do not report it as a number at all.

**Noisy in a way that hides: `handoff_quality`.** It is a free-text label today
("full packet" / "missing"), which is exactly the field a live model will fill with
something plausible. The fix is not a better prompt — it is Lesson 40's prereq
check: the packet either names six artifacts that exist on disk or it does not, and
existence is not a judgment call.

Two consequences for how the benchmark should be run once a model is in the loop.
First, **n > 1 or the numbers mean nothing**: a single live run of a 5-outcome
comparison reports one draw from four distributions of quite different width.
Second, **report the outcomes in order of who computed them** — harness-measured
first, model-produced last — so a reader knows which rows to argue with. The
scripted version in `code/main.py` hides this distinction because its literals all
have variance zero; the moment a model arrives, that is the most important thing
the table is not saying.

### 5 — what survives the cut is the two outcomes with units a reader owns

A one-page summary is a length budget and an audience constraint at once: ~350
words, and every number carrying a unit the reader already owns. That rules most of
the report out before any writing starts.

**Two of the five outcomes survive.** `tests_actually_run` is a yes/no about
whether anyone checked the work. `files_outside_scope` is a count of files that
changed and should not have. A reader can hold an opinion about both without
knowing what a workbench is. `acceptance_met`, `handoff_quality` and
`reviewer_total` each need the machinery explained first, so they belong in the
body as consequences, not in a table as numbers. The summary as written lands in
264 words with zero snake_case identifiers left in the text.

Why the other three fail the cut, specifically: all 5 outcome names are
identifiers, and 3 of them name a surface the reader has never heard of.
`reviewer_total (/10)` is the worst of them — a scale with no external referent, so
a 9 reads as a school grade and a 6 reads as failure, neither of which is what it
means. `handoff_quality` is a free-text label whose two values are "full packet"
and "missing", which tells a non-engineer nothing about what was lost.

The report's own explanatory paragraph shows the failure at prose length too: 50
words naming 3 surfaces ("runs the acceptance command through the feedback runner,
passes the verification gate, and ships a handoff packet"). That is the sentence a
non-engineer stops reading at. The summary makes the same claim naming zero.

And the number worth printing is not the benchmark's. Its rows are literals, so
"9/10" and "0 files outside scope" are claims. Exercise 2's numbers — one off-scope
file, no handoff, on a commit that actually shipped in this repository — are
checkable by anyone who doubts them. A summary aimed at a skeptic should cite the
one they can check, and say plainly that the worked example is a worked example.
