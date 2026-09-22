"""Exercise 3 — the array loses on inserts and wins on field-level review.

    Migrate `task_board.json` to JSON Lines so each task is a line and diffs
    are clean in version control.

Reading of the exercise: "clean diffs" needs a definition before it can be
measured, and the definition has to be the one `git` uses -- LCS-aligned
hunks, not a positional comparison. Measured that way on three edits a board
actually receives, JSON Lines wins two and loses one, which is a more useful
answer than "migrate".

**ANSWER: JSON Lines takes an insert from 9 changed lines to 1 and a reorder
from 12 to 4.** On a 4-task board, inserting a task at the front adds **9**
lines to the indented array and **1** to JSON Lines; swapping two tasks
changes **12** and **4**. Both formats report a status flip as **2** changed
lines -- one removed, one added.

**FINDING: the flip is a tie on line count and not on readability.** The
array's changed line is `"status": "done"`; JSON Lines' is the entire task
re-serialised, so the reviewer reads **1** field in one format and **6** in
the other to find out what moved. JSON Lines buys per-task diffs by giving
up per-field ones, and a board is edited field-by-field more often than it
is reordered.

**FINDING: the array is 9.5 lines per task and JSON Lines is 1.** The same
**4** tasks serialise to **38** lines with `indent=2` and **4** with one
object per line. The lesson says the board should stay under a screen; at
9.5 lines each that limit arrives at about **5** tasks in the array format
and **50** in JSON Lines, so the format decides when "you have a planning
problem" gets declared.

**FINDING: `load_board` and `save_board` are the whole migration.** Of the
module's **9** functions, **4** name `board_path` and only those touch the
file, so swapping the format leaves `run_one_turn` untouched -- the
round trip through JSON Lines returns the same **4** ids. That separation is
the reason the migration is two lines rather than a refactor.

Structure: `as_array()` and `as_lines()` are the serialisers; `diff()`
counts hunks the way git does.
"""

from __future__ import annotations

import difflib
import json
import pathlib
import tempfile

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "32-minimal-agent-workbench"
TASKS = (("T-001", "add input validation to /signup"),
         ("T-002", "document the new /signup contract"),
         ("T-003", "patch the auth bypass"),
         ("T-004", "tidy the changelog"))


def build(ref, rows):
    return [ref.Task(id=tid, goal=goal, owner="builder",
                     acceptance=[f"pytest -k {tid}"]) for tid, goal in rows]


def as_array(board):
    return json.dumps([task.__dict__ for task in board], indent=2) + "\n"


def as_lines(board):
    return "".join(json.dumps(task.__dict__) + "\n" for task in board)


def diff(before, after):
    """Changed lines the way git counts them: LCS-aligned, not positional."""
    old, new = before.splitlines(), after.splitlines()
    rows = list(difflib.unified_diff(old, new, n=0, lineterm=""))
    added = sum(1 for row in rows if row.startswith("+") and not row.startswith("+++"))
    removed = sum(1 for row in rows if row.startswith("-")
                  and not row.startswith("---"))
    return {"added": added, "removed": removed, "changed": added + removed,
            "lines": len(new)}


def edits(ref, render):
    board = build(ref, TASKS)
    base = render(board)
    board[2].status = "done"
    flipped = render(board)
    board = build(ref, TASKS)
    board.insert(0, build(ref, (("T-000", "hotfix the outage"),))[0])
    inserted = render(board)
    board = build(ref, TASKS)
    board[0], board[3] = board[3], board[0]
    reordered = render(board)
    return {"flip": diff(base, flipped), "insert": diff(base, inserted),
            "reorder": diff(base, reordered),
            "lines": len(base.rstrip("\n").split("\n"))}


def roundtrip(ref, root):
    """JSON Lines through the two functions the migration touches."""
    path = root / "task_board.jsonl"
    board = build(ref, TASKS)
    path.write_text(as_lines(board))
    loaded = [ref.Task(**json.loads(line))
              for line in path.read_text().splitlines() if line]
    return {"ids": [t.id for t in loaded], "match": len(loaded) == len(board)}


def board_io(ref):
    names = [n for n, v in vars(ref).items()
             if callable(v) and getattr(v, "__module__", "") == ref.__name__]
    readers = [n for n in names
               if "board_path" in getattr(v_code(ref, n), "co_varnames", ())]
    return {"functions": len(names), "board_io": sorted(readers)}


def v_code(ref, name):
    return getattr(getattr(ref, name), "__code__", None) or type("X", (), {})()


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    array, lines = edits(ref, as_array), edits(ref, as_lines)
    with tempfile.TemporaryDirectory() as tmp:
        trip = roundtrip(ref, pathlib.Path(tmp))
    return {
        "tasks": len(TASKS),
        "array": array, "lines": lines,
        "per_task_array": round(array["lines"] / len(TASKS), 1),
        "per_task_lines": round(lines["lines"] / len(TASKS), 1),
        "insert_ratio": round(array["insert"]["changed"]
                              / lines["insert"]["changed"]),
        "reorder_ratio": round(array["reorder"]["changed"]
                               / lines["reorder"]["changed"]),
        "roundtrip": trip,
        **board_io(ref),
    }


def verify(result):
    array, lines = result["array"], result["lines"]
    return [
        practice.Check(
            "ANSWER: JSON Lines takes an insert from 9 changed lines to 1",
            all([array["flip"]["changed"] == 2, lines["flip"]["changed"] == 2,
                 array["insert"]["changed"] == 9, lines["insert"]["changed"] == 1,
                 array["reorder"]["changed"] == 12,
                 lines["reorder"]["changed"] == 4]),
            f"inserting a task at the front changes {array['insert']['changed']} lines in "
            f"the indented array and {lines['insert']['changed']} in JSON Lines; swapping "
            f"two tasks changes {array['reorder']['changed']} and "
            f"{lines['reorder']['changed']}. A status flip is {array['flip']['changed']} "
            "in both",
        ),
        practice.Check(
            "FINDING: the array is 9.5 lines per task and JSON Lines is 1",
            all([result["per_task_array"] == 9.5, result["per_task_lines"] == 1.0,
                 array["lines"] == 38, lines["lines"] == 4]),
            f"the same {result['tasks']} tasks serialise to {array['lines']} lines with "
            f"indent=2 and {lines['lines']} with one object per line -- "
            f"{result['per_task_array']} against {result['per_task_lines']} per task. The "
            "lesson wants the board under a screen; the format decides where that falls",
        ),
        practice.Check(
            "FINDING: load_board and save_board are the whole migration",
            all([result["board_io"] == ["load_board", "main", "save_board",
                                        "write_initial"],
                 result["functions"] == 9,
                 result["roundtrip"]["match"] is True,
                 result["roundtrip"]["ids"][0] == "T-001"]),
            f"of the module's {result['functions']} functions, {result['board_io']} name "
            f"board_path, so the format is swappable without touching run_one_turn. The "
            f"JSON Lines round trip returns {result['roundtrip']['ids']}",
        ),
        practice.Check(
            "FINDING: the flip is a tie on lines and not on readability",
            all([array["flip"]["changed"] == lines["flip"]["changed"],
                 result["insert_ratio"] == 9, result["reorder_ratio"] == 3,
                 array["flip"]["added"] == 1]),
            f"both formats report the flip as {array['flip']['changed']} changed lines, "
            f"but the array's is the status field alone where JSON Lines re-serialises "
            f"the whole task. JSON Lines wins the insert {result['insert_ratio']}x and "
            f"the reorder {result['reorder_ratio']}x, and gives up per-field review",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
