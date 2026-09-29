"""Exercise 3 — leaderboard diff.

    Add a leaderboard diff command: given two `leaderboard.json` files, print which tasks moved and by how much.

Reading of the exercise: `diff_boards(old, new)` joins two `leaderboard.v1`
files on task name. Each task comes back as moved (with its score delta),
unchanged, added or removed, and the overall delta and any change of adapter
are reported with it. `latency_ms` and `timestamp` differ on every run, so
they are left out of "moved". Running the file with two paths prints the
table; with none it grades itself. Three pairs are diffed: the toy adapter
against a prompt-echo adapter, the lesson's committed `leaderboard.json`
against a fresh toy run, and echo on all five tasks against echo with the
`generation` file removed.

**ANSWER: toy -> echo moves 4 of 5 tasks.** arithmetic, code-exec and
multiple-choice each move -1.0, summary moves -0.361, generation stays at
1.0, and the overall moves -0.672.

**FINDING: the committed `outputs/leaderboard.json` says `"adapter":
"arithmetic"`,** a name `main.py` never writes; a fresh run writes `toy.v1`.
Its scores match a fresh run on all 5 tasks, while `latency_ms` differs on
5 of 5. A diff that counted every changed field would call every re-run a
move, so the doc's "reproducible" holds for scores only.

**FINDING: the overall can move when no task does.** Removing
`generation.jsonl` moves none of the four remaining
tasks, yet the echo adapter's overall drops from 0.328 to 0.160, because
the overall is an unweighted mean over whichever tasks are present.
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import types

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "49-lm-eval-harness"


def diff_boards(old, new, tol=1e-9):
    a = {t["task"]: t["score"] for t in old["tasks"]}
    b = {t["task"]: t["score"] for t in new["tasks"]}
    rows = []
    for task in sorted(a.keys() | b.keys()):
        if task not in b:
            rows.append((task, "removed", None))
        elif task not in a:
            rows.append((task, "added", None))
        else:
            d = b[task] - a[task]
            rows.append((task, "moved" if abs(d) > tol else "unchanged", round(d, 3)))
    return {
        "rows": rows,
        "overall": round(new["overall_score"] - old["overall_score"], 3),
        "adapter": (old.get("adapter"), new.get("adapter")),
    }


def print_diff(d):
    print(f"adapter {d['adapter'][0]} -> {d['adapter'][1]}   overall {d['overall']:+.3f}")
    for task, status, delta in d["rows"]:
        print(f"  {task:>16}  {status:>9}  {'' if delta is None else f'{delta:+.3f}'}")


def write_board(ref, tasks, adapter, path):
    ref.write_leaderboard(ref.run_leaderboard(tasks, adapter), path, adapter_name=adapter.name)
    return json.loads(path.read_text())


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    tmp = pathlib.Path(tempfile.mkdtemp())
    ref.seed_fixture_tasks(tmp / "tasks")
    tasks = ref.load_all_tasks(tmp / "tasks")
    echo = types.SimpleNamespace(name="echo", generate=lambda ps: list(ps))
    toy = write_board(ref, tasks, ref.ToyAdapter(), tmp / "toy.json")
    eco = write_board(ref, tasks, echo, tmp / "echo.json")
    four = write_board(ref, {k: v for k, v in tasks.items() if k != "generation"}, echo, tmp / "four.json")
    shipped = json.loads((parity.lesson_dir(PHASE, LESSON) / "outputs" / "leaderboard.json").read_text())
    lat = {t["task"]: t["latency_ms"] for t in shipped["tasks"]}
    return {
        "toy_echo": diff_boards(toy, eco),
        "shipped_fresh": diff_boards(shipped, toy),
        "latency_changed": sum(lat[t["task"]] != t["latency_ms"] for t in toy["tasks"]),
        "drop": diff_boards(eco, four),
        "echo_overall": (round(eco["overall_score"], 3), round(four["overall_score"], 3)),
    }


def moved(d):
    return {t: x for t, s, x in d["rows"] if s == "moved"}


def verify(result):
    r = result
    te, sf, dr = r["toy_echo"], r["shipped_fresh"], r["drop"]
    return [
        practice.Check(
            "ANSWER: toy -> echo moves 4 of 5 tasks and the overall by -0.672",
            moved(te) == {"arithmetic": -1.0, "code-exec": -1.0, "multiple-choice": -1.0, "summary": -0.361}
            and te["overall"] == -0.672,
            f"moved {moved(te)}; overall {te['overall']:+}",
        ),
        practice.Check(
            "FINDING: the committed leaderboard names adapter 'arithmetic'; only scores reproduce",
            sf["adapter"] == ("arithmetic", "toy.v1") and moved(sf) == {} and r["latency_changed"] == 5,
            f"adapter {sf['adapter']}, tasks moved {len(moved(sf))}/5, latency_ms changed "
            f"{r['latency_changed']}/5",
        ),
        practice.Check(
            "FINDING: removing a task moves the overall while no shared task moves",
            moved(dr) == {} and ("generation", "removed", None) in dr["rows"] and r["echo_overall"] == (0.328, 0.16),
            f"shared tasks moved {len(moved(dr))}; echo overall {r['echo_overall'][0]} -> {r['echo_overall'][1]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    if len(sys.argv) == 3:
        print_diff(diff_boards(*(json.loads(pathlib.Path(p).read_text()) for p in sys.argv[1:])))
        raise SystemExit(0)
    raise SystemExit(practice.selfcheck(globals()))
