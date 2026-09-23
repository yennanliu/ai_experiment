<!-- generated:start -->
# 15-autonomous-systems / 09-coding-agent-landscape

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/15-autonomous-systems/09-coding-agent-landscape/) · upstream spec
`phases/15-autonomous-systems/09-coding-agent-landscape/docs/en.md`

```bash
uv run demo practice run 09-coding-agent-landscape --ex 1
uv run demo explain 09-coding-agent-landscape --ex 1
uv run pytest demos/phases/15-autonomous-systems/09-coding-agent-landscape
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. How many turns does each scaffold take on the same task set? What is the… | code | T0 | `ex01_the_turn_saving_and_the_blast_radius_are_the_same_number.py` |
| 2 | Read the OpenHands paper (arXiv:2407.16741). The paper argues CodeAct beats JSON tool calls o… | explain | T0 | prose, below |
| 3 | Pick one task from your bug backlog that would require 10+ lines of change across two files.… | code | T0 | `ex03_the_scaffold_gap_is_widest_for_mediocre_models.py` |
| 4 | SWE-bench Verified has 161 single-file, 1–2 line tasks. Construct a score that excludes them.… | code | T0 | `ex04_excluding_the_easy_tail_reorders_nothing.py` |
| 5 | Read "Introducing SWE-bench Verified" (OpenAI). Explain the specific methodology used to remo… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

### 1 — the turn saving and the blast radius are the same number

Four numbers at one problem size say nothing about which way either number
moves, and both scaffolds accept a starting repo — so the same comparison runs
at 0, 1, 2 and 3 remaining bugs:

| bugs left | JSON turns | JSON radius | CodeAct turns | CodeAct radius |
|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 1 | 0 |
| 1 | 2 | 1 | 2 | 1 |
| 2 | 3 | 1 | 2 | 2 |
| 3 | 4 | 1 | 2 | 3 |

At the shipped size both pass 3/3: **JSON 4 turns at radius 1, CodeAct 2 turns
at radius 3**. JSON turns are exactly `bugs + 1`; CodeAct is 2 for any non-zero
number of bugs; CodeAct's radius is exactly `bugs`.

**The trade is one-for-one.** Every turn CodeAct saves is a file added to the
radius of a single action — two turns saved, two files added. The two columns
are the same quantity read from opposite ends, which is a sharper version of
"neither is strictly better" than the prose gets to.

**The two radii are not measured the same way.** `JsonScaffold.blast_radius`
is the literal `return 1`; `CodeActScaffold.blast_radius` returns
`worst_touched`, an observed maximum. At zero bugs CodeAct reports 0 and JSON
still reports 1 — the scaffold that touched nothing reports the *larger* radius,
because one number is an observation and the other is a claim about the design.
That asymmetry matters for the exercise as posed: "the per-action blast radius
of each" is a measurement for one scaffold and a specification for the other.

**And the action string is a description, not a cause.** Both `step` methods
call `_apply_fix` and mutate the repo *before* returning, and the only thing
either `run` reads back is whether the action was `done`. Nothing decodes or
executes the action. So the comparison isolates the scaffold from model quality
as advertised, but what varies between the two scaffolds is not an execution
model — it is that `JsonScaffold.step` returns inside its loop after the first
fix and `CodeActScaffold.step` appends and continues.

### 2 — the failure mode the paper acknowledges

*Draws on "CodeAct vs JSON tool calls".*

The acknowledged cost of the CodeAct bet is that a single action is an
arbitrary program, so its failure modes are "anything the sandbox runtime
allows" rather than anything an argument validator would have rejected — an
action that half-succeeds leaves the workspace in a state no schema described
and no turn boundary recorded. That mode dominates in production wherever the
executor touches something that is not a scratch container: a CI job with
credentials in its environment, a migration script against a real database, or
any repo where the agent's edits are pushed rather than diffed — because there
the cost of a partial multi-file action is not a retry but a cleanup, and the
per-turn validation that JSON tool calls buy is exactly the thing that bounds
the cleanup to one file.

### 3 — the scaffold gap is widest for mediocre models

The task is one this repository actually has: **enforce the manifest's
reference-use field**, which `harness/manifest.py` parses 4 times and which
`scripts/audit_practice.py` and `scripts/check_deps.py` read **0** times.
Fixing it means a check in the gate and the field on each of this lesson's 5
exercise entries — two files, about 17 lines.

(The measurement deliberately does not count the field name inside this
lesson's own `practice.yaml`: an earlier draft did, and adding the name to the
manifest's `verifies` prose broke the solution under pytest while it still
passed standalone. A solution that counts a token in a file it also writes is
measuring itself.)

Using the scaffolds' own turn counts (`bugs + 1` against `2`), two files is 3
JSON turns and 2 CodeAct turns, so end-to-end is `p³` against `p²`:

| per-step `p` | JSON | CodeAct | gap |
|---:|---:|---:|---:|
| 0.50 | 0.125 | 0.250 | 0.125 |
| 0.667 | 0.296 | 0.444 | **0.148** |
| 0.95 | 0.857 | 0.902 | 0.045 |
| 0.99 | 0.970 | 0.980 | 0.010 |

**Justifying the gap: it is `p²(1 − p)`, a hump rather than a slope.** It peaks
at `p = 2/3` and vanishes at both ends. A frontier model at 99% per-step gains
one point from CodeAct; a mediocre one gains fifteen. That inverts the usual
argument — compositional scaffolding is pitched as what lets the best models
stretch, and the arithmetic says it is worth most to the models that need the
fewest chances to get a multi-step edit right.

**And the gap scales with files, not lines.** For a `k`-file change it is
`p²(1 − p^(k−1))`: 0.045 across two files, 0.167 across five, 0.334 across ten.
The exercise's "10+ lines" does not enter anywhere — the scaffolds are paid per
action, and an action is a file. A 200-line single-file change is a one-turn
task for both.

**The price is in the other column.** CodeAct's radius for this task is 2 files
against JSON's 1, and what it buys is a partial edit: the check landing without
the field, or the field without the check. The first fails every lesson's
audit; the second passes silently. The JSON scaffold cannot reach either state,
because each turn is validated before the next begins.

### 4 — excluding the easy tail reorders nothing

**The score.** With 161 of 500 tasks in the easy tail, the hard-subset score is

```text
H = (500·V − 161·e) / 339
```

where `V` is the published Verified score and `e` is the system's solve rate on
the easy tail.

**At `e = 1` it reorders nothing.** `H` is affine and increasing in `V`, so the
ranking is preserved exactly. The lesson's 2026 band of 70–80% becomes
55.75–70.50% — lower numbers, identical order. The question "how does the
leaderboard shuffle" has the answer "it does not", and that is the useful
result: the naive exclusion score is a monotone transform of the score it was
supposed to correct.

**The gaps widen by a fixed factor.** `dH/dV = 500/339 = 1.475`, so a 2-point
Verified difference becomes 2.9 points on the hard subset. The metric makes the
leaderboard look more decisive while changing none of its decisions — which is
worth naming, because a wider spread is easy to mistake for better
discrimination.

**A real shuffle needs easy-tail rates nobody publishes.** Two systems swap only
when their easy-tail gap exceeds `500/161 = 3.11` times their Verified gap: at
two points apart, the lower-ranked system must solve 6.2 more points of the easy
tail. That per-system rate is not on any leaderboard, so the reshuffle is
unobservable from the published column. Constructing this score honestly
therefore requires per-task results, not per-system scores — and the moment you
have per-task results you would compute `H` directly and never need the formula.

**And this is not SWE-bench Pro.** Pro reports 23–59% for frontier systems where
the exclusion score predicts 55.75–70.50% from the same band — an overlap of
3.25 points. Removing the easy tail and raising the difficulty floor are
different operations, and Pro is much the harder of the two. The lesson's
advice — run a Pro-like subset of your own backlog — is not the same as
subtracting the easy tail from someone else's number.

### 5 — the Verified curation, and what it misses

*Draws on "SWE-bench, one paragraph".*

SWE-bench Verified is a human-curated 500-task subset in which professional
developers read each original task and removed the ones whose problem
statements were under-specified — where the issue text does not contain enough
information to know what patch would be accepted — and the ones whose tests
were unfair, meaning the hidden tests check behaviour the issue never asked for
or the environment fails for reasons unrelated to the patch. The category that
survives this filter is the task that is *unambiguous and unrepresentative*:
curating for "a competent developer could tell what is wanted from the issue
text alone" systematically keeps the 1–2 line fixes, which is precisely how 161
of the 500 come to be in the easy tail. Ambiguity and difficulty are
correlated in real backlogs — the changes that touch several files are exactly
the ones whose issue text under-describes them — so removing ambiguity removes
difficulty as a side effect, and the benchmark ends up measuring a distribution
that no production backlog contains.
