<!-- generated:start -->
# 14-agent-engineering / 19-benchmarks-swebench-gaia

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/19-benchmarks-swebench-gaia/) · upstream spec
`phases/14-agent-engineering/19-benchmarks-swebench-gaia/docs/en.md`

```bash
uv run demo practice run 19-benchmarks-swebench-gaia --ex 1
uv run demo explain 19-benchmarks-swebench-gaia --ex 1
uv run pytest demos/phases/14-agent-engineering/19-benchmarks-swebench-gaia
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Port the toy harness to run on a real repo (pick one of yours). Write 3 FAIL_TO_PASS tests fo… | code | T0 | `ex01_ftp_pre_is_computed_and_never_used_in_the_verdict.py` |
| 2 | Add a step-count metric. On your 3 tasks, how many agent steps per resolution? | code | T0 | `ex02_steps_per_resolution_divides_by_the_tasks_it_gave_up_on.py` |
| 3 | Read the SWE-bench+ paper. Implement a solution-leakage check (pattern-match the issue text a… | code | T0 | `ex03_quoting_the_bug_and_naming_the_fix_look_the_same_to_overlap.py` |
| 4 | Download a GAIA question from the public split. Trace what a GPT-4-class agent would do. What… | explain | T0 | prose, below |
| 5 | Read AgentBench's per-environment breakdown. Which environment mirrors your product surface?… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

### 1 — the verdict never checks that the tests failed first

`run_task` computes `ftp_pre` and then leaves it on the floor: the verdict is
`ftp_post == len(fail_to_pass) and ptp_broke == 0`. Porting three real bugs onto
the shipped `Task` — a `slugify` that drops non-ASCII letters, a `clamp` off by
one at the upper bound, a `truncate` that cuts before the ellipsis rather than
after — gives three FAIL_TO_PASS tests that really fail at version 0 and really
pass at version 1, two PASS_TO_PASS tests each, and 3/3 resolved. Then a fourth
task whose FAIL_TO_PASS tests were *already green*, with an identity patch,
resolves too: 2/2, nothing changed. A gate that never asserts the red state
cannot distinguish a fix from a no-op, which is the mechanism behind every
"resolved" task in a contaminated split.

Two smaller holes fall out of the same read. `ftp_fixed` is computed and
discarded, so the six fields of `TaskResult` contain no before-count — the
harness knows how many tests the patch *changed* and reports only how many pass.
And `ptp_broke` is a difference, so a PASS_TO_PASS test that was already failing
contributes zero however the patch behaves: a permanently red guard silently
stops guarding.

### 2 — steps per resolution moves in both directions

The shipped harness applies `task.patch` in one call, so there are no steps to
count until something produces patches. Wrapping an edit-and-retest agent around
it — one step is one candidate constant evaluated against the task's own tests —
gives 3, 7 and 16 steps on three real functions, 3/3 resolved in 26 steps: 8.7
steps per resolution, median 7.

The mean is the least useful of those numbers. The slowest run is 5.3x the
fastest and 1.8x the mean, so a budget set at the mean finishes two tasks of
three; the tail is what decides whether a harness terminates, and a single mean
is exactly the statistic that throws it away. Worse, the ratio is not monotone in
the budget: capping at 12 steps gives 11.0 per resolution on two resolutions, and
capping *harder*, at 8, gives 9.0 on the same two. Tightening the budget improved
the number because the denominator falls with the numerator. Steps per resolution
is unreadable without the resolve rate printed beside it — and the shipped
harness has nowhere to print either, since `Task` and `TaskResult` carry twelve
fields between them and none names a step, a token or a cost.

One incidental result worth keeping: the agent stops at `k=7` on a task whose
intended constant is 8, because `24 // 7` is also `3`. The first constant the
tests accept is not the right one. That is exercise 3's weak-coverage finding
arriving from the other direction.

### 3 — the check has to score what the patch *introduces*

SWE-bench+ reports that 32.67% of successful patches had the solution present in
the issue text. Implementing the check means deciding what "present" means, and
the deciding detail is that a good bug report quotes the broken code. Tokens on a
diff's removed lines are therefore *expected* in the issue text; only tokens that
appear on added lines and nowhere in the pre-image are evidence that the reporter
wrote the fix.

Scoring added-only tokens over twelve hand-labelled issues flags 4 — 33.3%,
against the paper's 32.67% — and agrees with the labels 12/12, precision 1.00 and
recall 1.00. Matching against the whole diff instead flags 10, 83.3%, at
precision 0.40; all six false positives are reports that quoted the failing call,
which is what a good report does. One line of filtering separates a 33.3% headline
from an 83.3% one, and only the first is a measurement of anything.

The harness itself is blind to all of it. Replacing every task's `description`
with its own patch — total leakage — leaves all twelve verdicts byte-identical,
because `run_task` reads `description` zero times. Contamination cannot be
measured by the thing being contaminated; it needs a second pass over the corpus.
And SWE-bench+'s other number, the 31.08% flagged for weak test coverage, needs
no text at all: a task with one FAIL_TO_PASS test and zero PASS_TO_PASS tests is
resolved by a patch that satisfies the one test and corrupts everything else,
because `ptp_broke` over an empty list is 0.

### 4 — a Level 3 GAIA question, traced, and the six tools it needs

GAIA's public split is a network download, so what is traced here is the
lesson's own worked example, which is a faithful Level 3 shape: *"Visit the arXiv
listing for ReAct, find the GitHub linked in the PDF, then count the open issues
with label 'bug' and return the ratio of bugs to total issues as a decimal."*
The lesson's `gaia_level` scores it 3, and the trace below is why. The section
drawn on is **GAIA (Mialon et al., Nov 2023)**.

A GPT-4-class agent runs roughly eleven steps: search arXiv for "ReAct" and
disambiguate it from the other papers of that name (the 2022 Yao et al. one is
wanted); open the abstract page; fetch the PDF; extract text from the PDF, which
is a different capability from fetching it; scan that text for a GitHub URL,
which typically appears in a footnote rather than the abstract; resolve the URL,
which may redirect to a renamed repo; call the issues API with `state=open` and
`labels=bug`; call it again without the label filter for the denominator; handle
pagination, because the API caps a page at 100 and the naive answer is the length
of page one; divide; and format as a decimal rather than a percentage.

Six distinct tools: a web search, an HTTP fetcher, a PDF text extractor, a
link/URL resolver, an authenticated REST client that can paginate, and arithmetic
with output formatting. Note what that list is *not* — it is not six calls to one
"browse" tool. The PDF extractor and the paginating REST client are the two that
fail silently: the first returns a garbled layout on a two-column paper, the
second returns a plausible wrong number. That is the design intent behind GAIA's
92%-vs-15% gap. A human doing this makes the same eleven moves and notices the
pagination cap immediately, because a page that ends at exactly 100 items looks
wrong to a person and looks like data to an agent.

The lesson's classifier is a keyword count, and that is worth reading critically
before trusting it. `steps` counts conjunctions such as "and" and "then", so any
compound sentence scores at least 1; `tools` counts "find", so a question that
merely says "find the capital of France" earns a tool point it does not need.
The classifier estimates *surface complexity of the prompt*, not depth of the
required tool chain — which is fine as triage and misleading as a difficulty
label.

### 5 — the environment that mirrors this surface is Bash, and SOTA there is not a number

Drawing on **AgentBench (Liu et al., ICLR 2024)**: AgentBench spans eight environments in three families: code (Bash, DB, KG), games
(Alfworld, LTP) and web (WebShop, Mind2Web), all multi-turn, at roughly 4k–13k
turns per split. For the product surface this phase keeps building — an agent
that reads a repository, runs commands and edits files — the mirror is **Bash**,
with DB second wherever the agent touches a datastore. It is not WebShop or
Mind2Web, even though those look more like "an agent doing a task", because their
failure mode is navigation and ours is state mutation: a wrong click is
recoverable, a wrong `rm` is not.

What "SOTA" looks like in the Bash environment is the part worth stating
carefully, because the honest answer is that a leaderboard position understates
the problem. AgentBench's own headline finding is that long-term reasoning,
decision-making and instruction following are the blockers — and Bash is where
all three compound, since every turn's state carries into the next and there is
no reset between steps. So the number to quote for Bash is not a mean success
rate; it is the success rate *paired with* the step distribution and the
irreversible-action count, which is exactly what exercise 2 showed the mean
discards.

That maps onto the lesson's three pitfalls one for one. Single-number fixation:
"our agent scores X on Bash" says less than P50/P75/P95 steps, which exercise 2
measured spanning 5.3x. Contaminated claims: a Bash score with no statement about
whether the task pool predates the model's cutoff is unreadable, and ">94% of
SWE-bench issues predate most model cutoffs" is the reason the SWE-bench-to-
SWE-bench+ drop is roughly 50% to 35%. Benchmark-as-development-target: Bash is
unusually easy to overfit, because the environment is a fixed command
vocabulary — an agent tuned to it learns the grader's shell, not shells.

The practical conclusion is the one the lesson's "what these do not measure"
section reaches from the other side. AgentBench-Bash is the right *external*
reference for this surface, and it is a sanity check rather than a target: report
it alongside a domain eval, a cost-per-resolution figure, and a tail number, or
report none of it.
