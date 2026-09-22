"""Exercise 5 — the packet names two paths and the next session needs six.

    Add a "next session prereqs" section listing exactly the artifacts the
    next session must load before acting.

Reading of the exercise: "exactly the artifacts" is a list that can be checked
against a real directory. So build it as paths plus an existence check, run it
against a workbench where some artifacts are missing, and compare with what
the packet names today.

**ANSWER: the packet names 2 paths where the next session needs 6, and
neither of the 2 is checked.** `verdict_pointer` carries
`outputs/verification/<task>.json` and `outputs/review/<task>.json`, both
f-strings that nothing verifies. The prereq list the next session actually
needs adds the state file, the scope contract, the feature board and the
feedback log -- **4** artifacts the packet never mentions.

**FINDING: on a real directory, 2 of the 6 prereqs are missing and the shipped
pointer still reads green.** Write four of the six artifacts to a temp
workbench and the prereq check reports **2** missing by name, while
`generate_handoff` emits the same two pointers it always emits. A pointer that
cannot be wrong is not a receipt.

**FINDING: the prereqs are an ordering, not a set.** The state file must load
before the feature board (the board says which feature is active, the state
says where the work stopped inside it), and both before the contract that
bounds the next diff. Listing them unordered leaves the next session to
rediscover the order -- which is the rediscovery the packet exists to stop.

**FINDING: `next_action` is prose, and the prereqs are what make it
executable.** The lesson's own demo sets it to "open PR with current diff and
request review", which names **0** artifacts and **0** commands. The prereq
list is what turns that sentence into a first minute the next session can
spend acting instead of looking.

Structure: `PREREQS` is the ordered list with a reason per artifact;
`check()` runs it against a temporary workbench.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "40-multi-session-handoff"
TASK = "T-001"

# (order, relative path, why the next session must load it before acting)
PREREQS = [
    ("agent_state.json", "where the work stopped and what the blockers are"),
    ("feature_list.json", "which feature is active; only one may be in_progress"),
    (f"outputs/scope/{TASK}.json", "the globs that bound the next diff"),
    (f"outputs/verification/{TASK}.json", "the gate's findings, including any unresolved"),
    (f"outputs/review/{TASK}.json", "the reviewer's per-dimension scores"),
    ("feedback_record.jsonl", "the full command log the packet only tails"),
]
PRESENT = [PREREQS[i][0] for i in (0, 1, 3, 4)]


def check(root, prereqs=PREREQS):
    """Each prereq with whether it is actually on disk, in load order."""
    return [{"path": path, "why": why, "present": (root / path).exists()}
            for path, why in prereqs]


def workbench():
    """A temp workbench carrying four of the six artifacts."""
    root = Path(tempfile.mkdtemp(prefix="handoff-"))
    for name in PRESENT:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}\n")
    return root


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    snapshot = ref.WorkbenchSnapshot(
        task_id=TASK,
        state={"active_task_id": None, "blockers": [],
               "next_action": "open PR with current diff and request review"},
        verdict={"passed": True, "findings": []}, review={"verdict": "pass", "total": 9},
        feedback=[{"command": "pytest", "exit_code": 0}],
        diff_summary={"touched": ["app/signup.py"]})
    markdown, payload = ref.generate_handoff(snapshot)
    root = workbench()
    rows = check(root)
    pointed = set(payload.verdict_pointer.values())
    return {
        "pointer_count": len(payload.verdict_pointer),
        "prereq_count": len(rows),
        "unnamed": [row["path"] for row in rows if row["path"] not in pointed],
        "missing": [row["path"] for row in rows if not row["present"]],
        "pointer_checked": any("exists" in line for line in markdown.splitlines()),
        "order": [row["path"] for row in rows],
        "next_action": payload.next_action,
        "action_names_artifacts": sum(row["path"] in payload.next_action for row in rows),
        "action_names_commands": sum(word in payload.next_action
                                     for word in ("pytest", "python", "git ", "uv ")),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the packet names 2 paths where the next session needs 6",
            all([result["pointer_count"] == 2, result["prereq_count"] == 6,
                 len(result["unnamed"]) == 4, result["pointer_checked"] is False]),
            f"verdict_pointer carries {result['pointer_count']} f-string paths that nothing "
            f"verifies, against {result['prereq_count']} prereqs; the "
            f"{len(result['unnamed'])} the packet never mentions are {result['unnamed']}",
        ),
        practice.Check(
            "FINDING: on a real directory 2 of the 6 prereqs are missing",
            all([result["missing"] == [f"outputs/scope/{TASK}.json", "feedback_record.jsonl"],
                 result["pointer_checked"] is False]),
            f"a workbench carrying four of the six artifacts reports {result['missing']} "
            "missing by name, while generate_handoff emits the same two pointers it always "
            "emits. A pointer that cannot be wrong is not a receipt",
        ),
        practice.Check(
            "FINDING: the prereqs are an ordering, not a set",
            all([result["order"][0] == "agent_state.json",
                 result["order"][1] == "feature_list.json",
                 result["order"][2] == f"outputs/scope/{TASK}.json"]),
            f"the load order is {result['order'][:3]} and then the verdict, review and "
            "feedback log: state before board before contract, because the board says "
            "which feature is active and the contract bounds the next diff",
        ),
        practice.Check(
            "FINDING: next_action is prose, and the prereqs make it executable",
            all([result["action_names_artifacts"] == 0,
                 result["action_names_commands"] == 0,
                 result["next_action"].startswith("open PR")]),
            f"the demo's next_action is {result['next_action']!r}: it names "
            f"{result['action_names_artifacts']} artifacts and "
            f"{result['action_names_commands']} commands, so the prereq list is what turns "
            "the sentence into a first minute spent acting instead of looking",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
