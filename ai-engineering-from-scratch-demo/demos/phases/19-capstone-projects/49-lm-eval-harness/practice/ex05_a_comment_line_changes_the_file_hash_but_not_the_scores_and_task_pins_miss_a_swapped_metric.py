"""Exercise 5 — pin task content with sha256.

    Pin task content with a sha256 in the leaderboard so a future reader can verify they scored the same tasks.

Reading of the exercise: each task is hashed from what the harness actually
scores, not from the file's bytes. That is the loaded `Example` records as
sorted-key compact JSON, so comment and blank lines, which
`load_task_jsonl` skips, do not count. The board written by the lesson's
`write_leaderboard` gains a `task_sha256` on each row and a `tasks_sha256`
over the sorted `name:hash` lines. `check_pins(board, task_dir)` reloads a
task directory and lists the tasks whose hash no longer matches. Four edits
are tried against a pinned toy-adapter board: none, a comment, one changed
target, and a swapped metric function.

**ANSWER: a board pinned on freshly seeded tasks verifies against them and
against the lesson's committed `outputs/tasks/`,** with 0 of 5 mismatches;
the committed files are byte-identical to `seed_fixture_tasks` output.

**FINDING: a raw-bytes hash would reject a harmless edit.** Adding a comment
and a blank line to `arithmetic.jsonl` changes its file sha256, but not the
canonical hash or the score. Changing one target, "25.0" to "25", is caught
(mismatch: arithmetic) and moves the toy's arithmetic score from 1.0 to 0.8.
The lesson's committed `leaderboard.json` has no hash field, so a reader
could not tell that change from a model regression.

**FINDING: pinning tasks does not pin the metric.** Scoring `summary` with
`exact_match` instead of `rouge_l` leaves every hash matching and moves the
summary score from 1.0 to 0.0, because the toy's first-sentence summary
keeps its full stop. The leaderboard needs the harness code's hash too.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import tempfile

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "49-lm-eval-harness"


def task_sha(examples):
    rows = [dataclasses.asdict(ex) for ex in examples]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pinned_board(ref, task_dir, path):
    tasks = ref.load_all_tasks(task_dir)
    adapter = ref.ToyAdapter()
    ref.write_leaderboard(ref.run_leaderboard(tasks, adapter), path, adapter_name=adapter.name)
    board = json.loads(path.read_text())
    for row in board["tasks"]:
        row["task_sha256"] = task_sha(tasks[row["task"]])
    lines = "".join(f"{t['task']}:{t['task_sha256']}\n" for t in sorted(board["tasks"], key=lambda t: t["task"]))
    board["tasks_sha256"] = hashlib.sha256(lines.encode()).hexdigest()
    path.write_text(json.dumps(board, indent=2) + "\n")
    return board


def check_pins(ref, board, task_dir):
    tasks = ref.load_all_tasks(task_dir)
    return [t["task"] for t in board["tasks"] if t["task"] not in tasks or task_sha(tasks[t["task"]]) != t["task_sha256"]]


def scores(board):
    return {t["task"]: round(t["score"], 3) for t in board["tasks"]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    tmp = pathlib.Path(tempfile.mkdtemp())
    seeded = tmp / "tasks"
    ref.seed_fixture_tasks(seeded)
    base = pinned_board(ref, seeded, tmp / "base.json")
    lesson = parity.lesson_dir(PHASE, LESSON) / "outputs"
    shipped = sorted(p.name for p in (lesson / "tasks").glob("*.jsonl"))
    identical = all((lesson / "tasks" / n).read_bytes() == (seeded / n).read_bytes() for n in shipped)
    arith = seeded / "arithmetic.jsonl"
    before = hashlib.sha256(arith.read_bytes()).hexdigest()
    arith.write_text("# annotated by a contributor\n\n" + arith.read_text())
    comment = (hashlib.sha256(arith.read_bytes()).hexdigest() != before, check_pins(ref, base, seeded),
               scores(pinned_board(ref, seeded, tmp / "c.json"))["arithmetic"])
    arith.write_text(arith.read_text().replace('["25.0"]', '["25"]'))
    target = (check_pins(ref, base, seeded), scores(pinned_board(ref, seeded, tmp / "t.json"))["arithmetic"])
    arith.write_text(arith.read_text().replace('["25"]', '["25.0"]'))
    ref.METRIC_FNS["rouge_l"] = ref.METRIC_FNS["exact_match"]
    metric = (check_pins(ref, base, seeded), scores(pinned_board(ref, seeded, tmp / "m.json"))["summary"])
    return {
        "self": check_pins(ref, base, seeded), "committed": check_pins(ref, base, lesson / "tasks"),
        "identical": identical, "n_shipped": len(shipped), "has_top": len(base["tasks_sha256"]) == 64,
        "comment": comment, "target": target, "metric": metric, "base": scores(base),
        "shipped_keys": sorted(json.loads((lesson / "leaderboard.json").read_text())),
    }


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: the pinned board verifies against the seeded and the committed task files",
            (r["self"], r["committed"], r["identical"], r["n_shipped"], r["has_top"]) == ([], [], True, 5, True),
            f"mismatches vs seeded {r['self']}, vs committed {r['committed']}; committed == seeded bytes: "
            f"{r['identical']} ({r['n_shipped']} files)",
        ),
        practice.Check(
            "FINDING: a comment changes the file hash but not the content; a target edit is caught",
            r["comment"] == (True, [], 1.0) and r["target"] == (["arithmetic"], 0.8)
            and "task_sha256" not in r["shipped_keys"] and "tasks_sha256" not in r["shipped_keys"],
            f"comment: (file hash changed, mismatches, score) {r['comment']}; '25.0'->'25': {r['target']}; "
            f"committed leaderboard keys {r['shipped_keys']}",
        ),
        practice.Check(
            "FINDING: task pins do not pin the metric; swapping rouge_l for exact_match passes the pins",
            r["metric"] == ([], 0.0) and r["base"]["summary"] == 1.0,
            f"(mismatches, summary score) after the swap {r['metric']}, was {r['base']['summary']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
