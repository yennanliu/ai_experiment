<!-- generated:start -->
# 19-capstone-projects / 10-multi-agent-software-team

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/10-multi-agent-software-team/) · upstream spec
`phases/19-capstone-projects/10-multi-agent-software-team/docs/en.md`

```bash
uv run demo practice run 10-multi-agent-software-team --ex 1
uv run demo explain 10-multi-agent-software-team --ex 1
uv run pytest demos/phases/19-capstone-projects/10-multi-agent-software-team
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Inject an obvious bug into a diff mid-run (extra `return None` before the main body). Measure… | code | T0 | `ex01_the_lessons_reviewer_false_approves_13pct_without_reading_the_diff_and_an_unreachable_code_check_reaches_0.py` |
| 2 | Reduce to two coders (architect + coder + reviewer + tester, coder runs two subtasks sequenti… | code | T0 | `ex02_two_coders_are_1_22x_slower_at_the_same_pass_rate_and_n_coders_0_passes_most_often.py` |
| 3 | Replace the merge coordinator with a single-writer constraint (subtasks touch disjoint file s… | code | T0 | `ex03_a_single_writer_rule_recuts_183_of_256_plans_and_nothing_in_the_lesson_checks_file_overlap.py` |
| 4 | Swap reviewer from GPT-5.4 to Claude Opus 4.7. Measure false-approval rate and token cost delta. | code | T0 | `ex04_the_swap_leaves_false_approval_at_13_2pct_because_no_model_is_called_and_costs_2_24x.py` |
| 5 | Add a fifth role: documenter (Haiku 4.5). After review, it produces a changelog entry. Measur… | code | T0 | `ex05_a_documenter_reading_the_board_costs_as_much_as_the_run_for_entries_wrong_1_time_in_4.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py`. It is a stdlib scaffold: a
typed message board, stub roles for the architect, coders, reviewer and
tester, and literal token counts per message. No model is called. The
reviewer is an 85% coin, the tester a 3% flake, and a bug is a `has_bug`
label that the architect plants in 30% of plans. The solutions drive those
seams on seeded runs, wrapping `coder_implement`, `reviewer_check`,
`architect_plan` and `Board`, and read the board each run leaves behind.
Prices were read on 2026-09-29 from
https://platform.claude.com/docs/en/about-claude/pricing (Opus 4.7 $5/$25,
Haiku 4.5 $1/$5 per MTok, and the 4.7 tokenizer's ~30% more tokens) and
https://developers.openai.com/api/docs/pricing (gpt-5.4 $2.50/$15).

### 1 — the lesson's reviewer false-approves 13% without reading the diff; an unreachable-code check reaches 0%

**The lesson's reviewer false-approves 26 of 200 injected runs (13.0%). Prompt
v2 gets that to 0 and still approves all 200 clean runs.** The coders are
wrapped to emit real source. Mid-run, one subtask per run gets an extra
`return None` right after its docstring. Because the stub has no prompt,
each "prompt" is a review rule applied to the diff text:

| reviewer | false approvals (200 injected) | clean runs approved (200) |
|---|---:|---:|
| lesson's `reviewer_check` | 26 | — |
| v1: reject any added `return None` | 0 | 0 |
| v2: reject a statement after a `return` in the same block | 0 | 200 |

v1 fails because two of the four functions open with a legitimate
`if ...: return None` guard clause. **Neither the lesson's reviewer nor its
tester reads the diff.** Both branch on the label. With the `return None` in
the text and the label off, the reviewer approves 200/200 and the tests pass
193/200. **Every rejection goes to coder-A** whichever subtask was flagged,
so 130 of the 174 rejections reach a coder who did not write the bug.

### 2 — two coders are 1.22x slower at the same pass rate, and `n_coders=0` passes most often

**Wall-clock rises 1.22x, and the pass rate stays at 935/1000.** The code has
no clock, so wall-clock here is the critical path in generated tokens: the
serial messages plus the slowest coder lane.

| layout | critical path (tokens) | pass |
|---|---:|---:|
| 4 coders | 21,101 | 935/1000 |
| 2 coders, 2 subtasks each | 25,805 (1.22x) | 935/1000 |
| 1 coder, 4 in a row | 34,973 (1.66x) | 935/1000 |
| single-agent baseline | 21,051 | 703/1000 |

The pass rate cannot move, because `reviewer_check` and `tester_run` see
only `(diffs, rng)`, never which coder did the work. **The lesson's own
switch does something else.** `run_team(n_coders=2)` slices the plan to 2 of
its 4 subtasks and passes 936. `n_coders=0` codes nothing and passes 973,
the best score, because a planted bug in a subtask that was never dispatched
never reaches review. Four parallel coders only tie the single agent on
wall-clock (1.002x): the serial handoffs cost what parallelism saves.

### 3 — a single-writer rule re-cuts 183 of 256 plans, and nothing in the lesson checks file overlap

**Under single-writer the architect must re-cut 183 of 256 plans (71.5%).**
Each of the lesson's four subtasks is given one extra edit in
`src/__init__.py`, `src/config.py` or `tests/conftest.py`, or none, which
makes 4^4 plans. The architect then has two choices:

| strategy | coder-stage critical path | architect's extra work |
|---|---:|---|
| merge overlapping subtasks | 1.68x (worst 3.57x) | lanes left {1: 3, 2: 54, 3: 126, 4: 73} |
| extract contested edits into a first subtask | 1.46x | a fifth subtask in 183 plans |

**The lesson's plan is disjoint by construction, and nothing checks it.** Its
stub gives 4 lanes, so the rule costs 0 there. The "merge coordinator" is one
2,000-token message. A plan in which `cache` also edits `src/parser.py`
produces 200/200 identical `run_team` results.

### 4 — the swap leaves false approval at 13.2%, because no model is called, and makes review 2.24x dearer

**False approval stays at 41/310 bugged runs (13.2%) for either reviewer, and
the reviewer's cost goes from $0.0401 to $0.0900 per run (+$0.0499, 2.24x).**
Price accounts for 1.73x of that. The rest is Opus 4.7's tokenizer, which
turns the reviewer's 5,037 tokens per run into 6,549. The extra 1,511 tokens
are 4.3% of a 34,973-token team run. The rate cannot move because
`main.py` names no model and has no model seam: the reviewer's tokens are
the literal 1,800, or 3,300 after a rejection. The stub's stated 15%
false-approve rate measures 13.2% on these seeds.

### 5 — a documenter that reads the board costs as much as the run, for entries wrong 1 time in 4

**No, the quality does not justify the spend.** A documenter placed at the
`approved` message writes 744/1000 entries that are true of what shipped.
Reading the transcript costs 33,786 tokens ($0.0339 on Haiku 4.5), which is
0.97x the 34,973-token four-role run.

| why an entry is wrong | entries |
|---|---:|
| it documents a change that then fails tests | 65 |
| it credits the revision to `parser` when the bug was elsewhere | 198 of 269 revised runs |

The first cause is placement: the documenter runs before the tester. The
second comes from the lesson's revised DIFF_READY, which always says
`parser`. **The board holds only subtask names and line counts,** so
reading just those payloads (103 tokens, $0.00021) writes the same entry,
and a template writes it for nothing. `MsgKind` has no changelog kind among
its 9.
