<!-- generated:start -->
# 14-agent-engineering / 32-minimal-agent-workbench

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/32-minimal-agent-workbench/) · upstream spec
`phases/14-agent-engineering/32-minimal-agent-workbench/docs/en.md`

```bash
uv run demo practice run 32-minimal-agent-workbench --ex 1
uv run demo explain 32-minimal-agent-workbench --ex 1
uv run pytest demos/phases/14-agent-engineering/32-minimal-agent-workbench
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a `last_run` timestamp to `agent_state.json`. Refuse to run if the file is older than 24… | code | T0 | `ex01_adding_a_field_to_the_state_file_breaks_the_loader.py` |
| 2 | Add a `priority` field to the task board and change the puller to always pick the highest pri… | code | T0 | `ex02_the_puller_takes_the_first_todo_which_is_board_order.py` |
| 3 | Migrate `task_board.json` to JSON Lines so each task is a line and diffs are clean in version… | code | T0 | `ex03_the_array_loses_on_inserts_and_wins_on_field_level_review.py` |
| 4 | Write a `lint_workbench.py` that fails if `AGENTS.md` is over 80 lines or references a file t… | code | T0 | `ex04_the_router_points_at_a_file_write_initial_never_creates.py` |
| 5 | Decide which one of the three files would hurt the most to lose. Defend it. | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

### 1 — the timestamp is four lines; the migration is the expensive part

A `last_run` field and a 24-hour rule are straightforward: on a virtual clock,
readings at hours 1, 23 and 24 run, 25 and 200 refuse, and an operator
confirmation lets the 25-hour read through and rewrites `last_run`. Three allowed,
two refused, out of five.

What costs something is adding the field at all. `load_state` does
`AgentState(**raw)`, so the file's schema *is* the dataclass signature: a state
file carrying six keys against five fields raises `TypeError: got an unexpected
keyword argument 'last_run'`. The lesson calls this file "the system of record",
and a system of record that last week's code cannot read is a system of record for
one version. Filtering to known fields on load — one line — keeps both readers
working, and is what makes the field additive rather than a breaking change. The
same defect is in `load_board` (exercise 2), so two of the workbench's three files
have a strict positional schema and the third is Markdown.

Two design notes on the rule itself. The file stores an instant and the rule
compares it against the reader's "now", so two readers two hours apart disagree
about the same file — the staleness boundary depends on a clock the workbench does
not own. A monotonic run counter stored beside the timestamp makes the comparison
local. And there is no operator channel to confirm through: `run_one_turn` takes
two arguments, `AgentState` has five fields and none of them is an approval, so
the confirmation arrives as an argument the signature lacks and recording it needs
a sixth field. Without that field the next session cannot tell a confirmed stale
run from a fresh one, which is the thing the rule exists to make visible.

### 2 — the priority field is easy; board order was already the priority

`run_one_turn` pulls with `next((t for t in board if t.status == "todo"), None)`,
so today's priority is position in the JSON array. Over six shuffled orderings of
the same four tasks the shipped puller picks four different first tasks; a
priority puller picks `T-003` — the auth bypass — all six times. They agree on the
two orderings where the array happened to lead with the urgent task.

That is the finding worth keeping: the shipped puller is deterministic *and*
wrong, and its determinism comes from a property nobody declared. `save_board`
serialises the list as given, so appending a task changes which task runs next,
and nothing in the file or the tests pins the discipline.

Two details the exercise does not mention. `load_board` does `Task(**t)`, so
adding `priority` raises `TypeError` in the shipped loader — the identical failure
exercise 1 found in `load_state`, and the reason both migrations want a tolerant
load first. And a priority field without a tie-break only moves the ambiguity: two
tasks share priority 1, so `max(todo, key=priority)` still consults array order on
a tie, while `(-priority, id)` gives one distinct pick across every ordering.
Picking the tie-break is the actual design decision; the field is bookkeeping.

### 3 — JSON Lines wins inserts and reorders, and loses field-level review

"Clean diffs" has to be measured the way `git` counts them — LCS-aligned hunks,
not a positional comparison — or the answer comes out wrong in both directions.
Measured that way on a four-task board:

| edit | indented array | JSON Lines |
|---|---|---|
| flip one status | 2 changed lines | 2 changed lines |
| insert a task at the front | 9 | 1 |
| swap two tasks | 12 | 4 |

So JSON Lines wins inserts 9x and reorders 3x. The status flip is a tie on line
count and not on readability: the array's changed line is `"status": "done"`,
where JSON Lines re-serialises the entire task — the reviewer reads one field in
one format and six in the other to find out what moved. JSON Lines buys per-task
diffs by giving up per-field ones, and a board is edited field-by-field more often
than it is reordered. That is a real trade, not a migration everyone should do.

The size argument is the stronger one. The same four tasks are 38 lines with
`indent=2` and 4 as JSON Lines — 9.5 lines per task against 1.0. The lesson says
the board should stay under a screen and that a longer board means "a planning
problem, not a board problem"; at 9.5 lines each, that limit arrives at about five
tasks in the array format and fifty in JSON Lines. The format silently decides
when the planning problem gets declared.

The migration itself is two lines, and that is the point of the file layout: of
the module's nine functions, four name `board_path`, so swapping the serialiser
leaves `run_one_turn` untouched and the round trip returns the same four ids.

### 4 — the router points at a file `write_initial` never creates

The lint's two rules are a line ceiling and a reference check, and running it
against a freshly laid-down workbench fails on the second. `AGENTS.md` is 13 lines
— comfortably inside 80 — and references three paths, of which `docs/agent-rules.md`
does not exist. `write_initial` creates three files and that is not one of them.

The line rule cannot bind here at all: the ceiling is 67 lines away, so the only
live rule is the one the exercise phrases second. That is worth noticing when
writing linters generally — the cheap rule is the one that gets written and the
expensive one is the one that fires.

Two things follow from the missing file. First, the router is short *because* the
rules it routes to are missing: at 13 lines it uses 16.2% of its budget, and the
one file that would carry the startup rules, the scope and the definition of done
was never written. The doc's argument is that long manuals get ignored and short
routers get followed — true, and shortness bought by absence is a different
property from shortness bought by delegation. The lint is what tells the two
apart.

Second, the same check applied one level down is worse. `AGENTS.md` declares
`python3 -m pytest -x` and each `Task` carries an `acceptance` command, and
`run_one_turn` reads `acceptance` zero times before marking a task `done`. A lint
extended to "every acceptance command is reachable" flags 2 of 2 board tasks,
whose commands name `test_app.py` and `docs/api.md` — neither of which exists
either. The workbench declares a definition of done and ships nothing that could
satisfy it, which is Lesson 31's verification finding in file form.

### 5 — lose `agent_state.json` and the run stops; lose the others and it degrades

**agent_state.json is the system of record** makes the argument directly: state
carries the active task, the touched files, the assumptions, the blockers and the
next action; the agent reads it at every turn and the next session reads it
*instead of replaying chat*. State lives in a file because chat history is
unreliable — sessions die, conversations get trimmed, the file does not.

Ranking the three by what losing each actually costs:

**`agent_state.json` — unrecoverable.** It is the only one of the three that
holds information which exists nowhere else. `AGENTS.md` can be rewritten from the
conventions; `task_board.json` can be reconstructed from the issue tracker, the
PR list, or a conversation with whoever wrote it. The assumptions the last session
made, the blocker it hit at step 30, and the file it was halfway through editing
are recoverable only by redoing the work. Exercise 1's staleness rule exists
because of this: a 25-hour-old state file is dangerous precisely because it is
still the only record, so the answer is "confirm", not "discard".

There is a sharper version of the argument. The other two files are *inputs* — a
human wrote them and a human can write them again. State is an *output*, produced
by the run, and the only copy. In the lesson's primitive vocabulary it is session
persistence, and losing session persistence is what turns a resumable job into a
restart. Lesson 29 priced that: a run killed at step 7 costs 12 steps to redo
against a durable port's 5, and the ratio grows with run length.

**`task_board.json` — recoverable, and its loss is silent.** Reconstructing the
queue is tedious and possible. The real cost is that losing it is not obvious:
`run_one_turn` with an empty board sets `next_action = "no work on the board,
idle"` and reports success. An agent with no state *stops*; an agent with no board
*idles*, which looks like being finished. That is worse for a human watching a
dashboard and better for the repo.

**`AGENTS.md` — cheapest to lose, and the one that degrades everything else.**
The router is 13 lines and reconstructible in a minute. But it is the only file
that says *how to read the other two*, so losing it does not stop the run — it
makes every subsequent run slightly wrong, in a way that is hard to attribute.
Exercise 4's finding is the same shape from the other side: the router already
points at a file that does not exist, and nothing noticed.

**The caveat worth stating.** This ranking is for the three files *as the lesson
defines them*, and exercise 3 changes it. If the board carries `priority` and the
puller sorts by it, the board stops being reconstructible from a conversation —
"which of these was urgent" is a judgement someone made once. Every field added to
the board moves it toward the state file's category. The floor is three files; the
ranking is not a property of the count.
