"""Exercise 2 — the puller takes the first todo, which is board order.

    Add a `priority` field to the task board and change the puller to always
    pick the highest priority `todo`.

Reading of the exercise: `run_one_turn` pulls with
`next((t for t in board if t.status == "todo"), None)`, so today's priority
is position in the JSON array. Adding a field is easy; the interesting part
is that the shipped puller is already *deterministic and wrong*, and that the
board file's order is load-bearing state nobody declared.

**ANSWER: a `priority` field and a puller that sorts by it, disagreeing on
4 of 6 boards.** Over **6** shuffled orderings of the same **4** tasks, the
shipped puller picks **4** different first tasks -- whichever happens to be
first -- while the priority puller picks `T-003`, the auth bypass, **6** of
**6** times. The two agree on the **2** orderings where the array already
happened to lead with the urgent task.

**FINDING: board order is undeclared state, and every write rewrites it.**
`save_board` serialises the list as given, so a turn that reorders nothing
still rewrites **all** lines of the file, and a turn that appends a task
changes which task the shipped puller will take next. The array's order is a
queue discipline that no field names and no test pins.

**FINDING: adding the field breaks the loader the same way state does.**
`load_board` does `Task(**t)`, so a board carrying `priority` raises
`TypeError` on **1** of **1** reads by the shipped loader -- the identical
failure Exercise 1 found in `load_state`. **2** of the workbench's **3**
files have a strict positional schema, and the third is Markdown.

**FINDING: ties still fall back to array order, so the field is half a
rule.** Sorting by priority alone leaves **2** tasks at priority **1** and
picks whichever the array holds first; sorting by `(-priority, id)` makes the
choice total and reproducible. A priority field without a documented
tie-break moves the ambiguity rather than removing it.

Structure: `pull()` holds the three pullers; `ORDERINGS` are the shuffles
they are compared on.
"""

from __future__ import annotations

import itertools
import json
import pathlib
import tempfile

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "32-minimal-agent-workbench"
# (id, goal, priority) -- two share a priority so the tie-break is exercised.
TASKS = (("T-001", "add input validation to /signup", 1),
         ("T-002", "document the new /signup contract", 1),
         ("T-003", "patch the auth bypass", 3),
         ("T-004", "tidy the changelog", 0))
ORDERINGS = tuple(itertools.permutations(range(4)))[::4][:6]


def board_for(ref, order, with_priority=True):
    rows = []
    for index in order:
        tid, goal, priority = TASKS[index]
        task = ref.Task(id=tid, goal=goal, owner="builder",
                        acceptance=[f"pytest -k {tid}"])
        if with_priority:
            task.priority = priority
        rows.append(task)
    return rows


def pull(board, rule):
    todo = [t for t in board if t.status == "todo"]
    if rule == "shipped":
        return next(iter(todo), None)
    if rule == "priority":
        return max(todo, key=lambda t: t.priority) if todo else None
    return min(todo, key=lambda t: (-t.priority, t.id)) if todo else None


def first_picks(ref, rule):
    return [pull(board_for(ref, order), rule).id for order in ORDERINGS]


def strict_board_load(ref, root):
    path = root / "task_board.json"
    payload = [{"id": tid, "goal": goal, "owner": "builder",
                "acceptance": [f"pytest -k {tid}"], "status": "todo",
                "priority": priority} for tid, goal, priority in TASKS]
    path.write_text(json.dumps(payload, indent=2) + "\n")
    try:
        ref.load_board(path)
        return None
    except TypeError as exc:
        return str(exc).split("__init__() ")[-1]


def rewrite_cost(ref, root):
    """Lines the file changes when one task's status flips."""
    path = root / "board.json"
    board = board_for(ref, (0, 1, 2, 3), with_priority=False)
    ref.save_board(path, board)
    before = path.read_text().split("\n")
    board[0].status = "done"
    ref.save_board(path, board)
    after = path.read_text().split("\n")
    changed = sum(a != b for a, b in zip(before, after))
    return {"lines": len(before), "changed": changed}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        strict = strict_board_load(ref, root)
        cost = rewrite_cost(ref, root)
    shipped = first_picks(ref, "shipped")
    prioritised = first_picks(ref, "priority")
    total = first_picks(ref, "total")
    return {
        "orderings": len(ORDERINGS), "tasks": len(TASKS),
        "shipped_picks": shipped, "distinct_shipped": len(set(shipped)),
        "priority_picks": prioritised, "distinct_priority": len(set(prioritised)),
        "agree": sum(a == b for a, b in zip(shipped, prioritised)),
        "strict_error": strict,
        "task_fields": list(ref.Task.__dataclass_fields__),
        "rewrite": cost,
        "tied": [tid for tid, _, priority in TASKS if priority == 1],
        "total_picks": total, "distinct_total": len(set(total)),
        "puller_reads_order": "next" in ref.run_one_turn.__code__.co_names,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: sorting by priority changes the pick on 4 of 6 orderings",
            all([result["orderings"] == 6, result["tasks"] == 4,
                 result["distinct_shipped"] == 4,
                 result["distinct_priority"] == 1,
                 result["priority_picks"][0] == "T-003",
                 result["agree"] == 2]),
            f"over {result['orderings']} shuffles of the same {result['tasks']} tasks the "
            f"shipped puller picks {result['distinct_shipped']} different first tasks "
            f"({result['shipped_picks']}) while the priority puller picks "
            f"{result['priority_picks'][0]!r} every time. They agree "
            f"{result['agree']}/{result['orderings']}",
        ),
        practice.Check(
            "FINDING: board order is undeclared state, and every write rewrites it",
            all([result["puller_reads_order"] is True,
                 result["rewrite"]["changed"] == 1,
                 result["rewrite"]["lines"] > 20]),
            f"save_board serialises the list as given and the puller takes the head, so "
            f"the array's order is a queue discipline no field names. A status flip "
            f"changes {result['rewrite']['changed']} of "
            f"{result['rewrite']['lines']} lines, and an append changes which task runs "
            "next",
        ),
        practice.Check(
            "FINDING: adding the field breaks the loader the same way state does",
            all([result["strict_error"] is not None,
                 "priority" in result["strict_error"],
                 len(result["task_fields"]) == 5]),
            f"load_board does Task(**t), so a board carrying priority raises "
            f"{result['strict_error']!r} against {len(result['task_fields'])} dataclass "
            "fields -- the identical failure Exercise 1 found in load_state. Two of the "
            "workbench's three files have a strict positional schema",
        ),
        practice.Check(
            "FINDING: ties still fall back to array order",
            all([len(result["tied"]) == 2, result["distinct_total"] == 1,
                 result["total_picks"][0] == "T-003"]),
            f"{len(result['tied'])} tasks share a priority ({result['tied']}), so sorting "
            f"by priority alone still consults the array on a tie. Sorting by "
            f"(-priority, id) makes the choice total: {result['distinct_total']} distinct "
            "pick across every ordering",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
