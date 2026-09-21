"""Exercise 4 — the router points at a file write_initial never creates.

    Write a `lint_workbench.py` that fails if `AGENTS.md` is over 80 lines or
    references a file that does not exist.

Reading of the exercise: the second rule is the interesting one, because the
shipped `AGENTS_MD` names `docs/agent-rules.md` and `write_initial` creates
**3** files, none of them that one. So the lint is written, run against a
freshly laid-down workbench, and it fails on the reference -- which is the
result that makes the lint worth having.

**ANSWER: the lint fails a fresh workbench on 1 of its 2 rules.** `AGENTS.md`
is **13** lines, comfortably inside the 80-line ceiling, and references **3**
paths of which **1** -- `docs/agent-rules.md` -- does not exist after
`write_initial`. The other **2**, `agent_state.json` and `task_board.json`,
are created.

**FINDING: the router is short because the rules it routes to are missing.**
The doc's argument is that "long manuals get ignored; short routers get
followed", and at **13** lines this router is **16.2%** of its budget --
because the **1** file that would carry the startup rules, the scope and the
definition of done was never written. Shortness bought by absence is not the
same property.

**FINDING: the verification command is named and never run.**
`AGENTS.md` declares `python3 -m pytest -x` and `Task.acceptance` carries a
per-task command, and `run_one_turn` marks a task `done` after reading
**0** of them. A lint that checked "every acceptance command is reachable"
would flag **2** of **2** board tasks, whose commands name `test_app.py` and
`docs/api.md` -- neither of which exists either.

**FINDING: the 80-line rule is the cheap half and it cannot bind here.**
Adding every rule the router defers to would take the file past **80** lines
only if the deeper docs were inlined, which is the failure the rule exists to
catch. On the shipped tree the ceiling is **67** lines of headroom away,
so the lint's only live rule is the reference check -- which is the one the
exercise phrases second.

Structure: `lint()` is the two rules; `lay_down()` builds a fresh workbench
in a temporary directory so the reference tree is untouched.
"""

from __future__ import annotations

import pathlib
import re
import tempfile

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "32-minimal-agent-workbench"
MAX_LINES = 80
PATH_PATTERN = re.compile(r"`([\w./-]+\.(?:md|json|py|txt))`")


def lay_down(ref, root):
    """write_initial against a temporary directory, never the reference tree."""
    paths = {name: root / name for name in
             ("AGENTS.md", "agent_state.json", "task_board.json")}
    ref.write_initial(paths["agent_state.json"], paths["task_board.json"],
                      paths["AGENTS.md"])
    return paths


def referenced(text):
    return sorted(set(PATH_PATTERN.findall(text)))


def lint(root, agents_path, max_lines=MAX_LINES):
    text = agents_path.read_text()
    lines = text.rstrip("\n").split("\n")
    paths = referenced(text)
    missing = [p for p in paths if not (root / p).exists()]
    problems = []
    if len(lines) > max_lines:
        problems.append(f"{len(lines)} lines > {max_lines}")
    for path in missing:
        problems.append(f"references {path}, which does not exist")
    return {"ok": not problems, "problems": problems, "lines": len(lines),
            "referenced": paths, "missing": missing}


def acceptance_targets(ref, board_path):
    """Files the board's acceptance commands name, and whether they exist."""
    board = ref.load_board(board_path)
    targets = []
    for task in board:
        for command in task.acceptance:
            for token in command.split():
                if "." in token and "/" not in token[:1]:
                    targets.append(token.split("::")[0])
                    break
    return targets


def turn_reads_acceptance(ref):
    return "acceptance" in ref.run_one_turn.__code__.co_names


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        paths = lay_down(ref, root)
        report = lint(root, paths["AGENTS.md"])
        targets = acceptance_targets(ref, paths["task_board.json"])
        created = sorted(p.name for p in root.iterdir())
        unreachable = [t for t in targets if not (root / t).exists()]
    return {
        "ok": report["ok"], "problems": report["problems"],
        "lines": report["lines"], "ceiling": MAX_LINES,
        "headroom": MAX_LINES - report["lines"],
        "referenced": report["referenced"], "missing": report["missing"],
        "created": created,
        "budget_used": round(100 * report["lines"] / MAX_LINES, 1),
        "acceptance_targets": targets, "unreachable": unreachable,
        "turn_reads_acceptance": turn_reads_acceptance(ref),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the lint fails a fresh workbench on 1 of its 2 rules",
            all([result["ok"] is False, result["lines"] == 13,
                 result["missing"] == ["docs/agent-rules.md"],
                 len(result["referenced"]) == 3, len(result["created"]) == 3,
                 len(result["problems"]) == 1]),
            f"AGENTS.md is {result['lines']} lines, inside the {result['ceiling']}-line "
            f"ceiling, and references {result['referenced']} of which "
            f"{result['missing']} does not exist after write_initial. The lint reports "
            f"{result['problems']}",
        ),
        practice.Check(
            "FINDING: the router is short because the rules it routes to are missing",
            all([result["budget_used"] == 16.2, result["lines"] == 13,
                 result["missing"] == ["docs/agent-rules.md"]]),
            f"at {result['lines']} lines the router uses {result['budget_used']}% of its "
            f"budget, because the one file that would carry the startup rules, the scope "
            f"and the definition of done -- {result['missing'][0]} -- was never written. "
            "Shortness bought by absence is a different property",
        ),
        practice.Check(
            "FINDING: the verification command is named and never run",
            all([result["turn_reads_acceptance"] is False,
                 len(result["acceptance_targets"]) == 2,
                 len(result["unreachable"]) == 2]),
            f"AGENTS.md declares a verification command and Task.acceptance carries "
            f"per-task ones, and run_one_turn reads acceptance "
            f"({result['turn_reads_acceptance']}). A lint checking reachability would "
            f"flag {len(result['unreachable'])} of {len(result['acceptance_targets'])} "
            f"board commands: {result['acceptance_targets']}",
        ),
        practice.Check(
            "FINDING: the 80-line rule cannot bind here",
            all([result["headroom"] == 67, result["lines"] < MAX_LINES,
                 len(result["problems"]) == 1]),
            f"the ceiling is {result['headroom']} lines away, so the only live rule is "
            "the reference check -- the one the exercise phrases second. The line rule "
            "fires when the deeper docs get inlined, which is the failure it exists for",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
