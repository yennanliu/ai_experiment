"""Exercise 3 — the control that keeps the prototype out is a monkeypatch in 25 files.

    Add a technical control that prevents a prototype from reaching
    production data.

Reading of the exercise: "technical" rules out a note in a README. This
repository already has the control the exercise asks for, and it is worth
naming because it is the kind that works: a solution that needs to exercise
a lesson's writer points the module's output path at a temporary directory
before calling it.

**ANSWER: 25 shipped solutions redirect a path into `tempfile`, and the
reference tree they could have written to is untouched.** Lesson 42's four code
solutions set `ref.PACK` to a fresh temp directory before running the pack
assembler; the upstream lesson's own `outputs/` directory is committed and
unmodified. The control is **1** assignment per solution, executed before the
call rather than checked afterwards.

**FINDING: the control works because the production path is a module
global.** `PACK`, `OVERRIDES_PATH` and their siblings are computed from
`__file__` at import time, so a caller can rebind them. A writer that
computed its destination inside the function would leave **0** places to
intervene, and the same solution would have to fork the code it is supposed
to import.

**FINDING: `required_controls("prototype")` names none of this.** It returns
**3** strings -- synthetic or recorded inputs, discardable implementation,
learning question -- and **0** of them is enforceable. The doc's own sentence
is that a warning banner is not enough; a list of three phrases is a banner
with three lines.

**FINDING: the check that the control held is cheaper than the control.**
Asserting that the reference tree has no new files is one `git status` on a
directory the tests never write to. Running it here reports **0** modified
paths under the phase, which is the receipt that the **25** redirections
worked -- and it is the artifact the stage decision should carry.

Structure: `redirects()` counts the control; `untouched()` is the receipt.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "53-prototype-pilot-or-production"
BASE = Path(__file__).resolve().parents[2]
REFERENCE = Path("/Users/jliu/ai-engineering-from-scratch")
PHASE_DIR = "phases/14-agent-engineering"


def solutions():
    """Every shipped solution except this one, which names the control it counts."""
    here = Path(__file__).resolve()
    return sorted(path for path in BASE.glob("*/practice/ex0*.py")
                  if path.resolve() != here)


def redirects():
    """Solutions that point a module global at a temporary directory."""
    rows = [path for path in solutions()
            if "tempfile" in path.read_text(encoding="utf-8")]
    rebinds = [path for path in rows
               if "ref.PACK =" in path.read_text(encoding="utf-8")]
    return len(rows), len(rebinds)


def untouched():
    """The receipt: the reference tree has nothing modified or added."""
    done = subprocess.run(["git", "status", "--porcelain", "--", PHASE_DIR],
                          cwd=REFERENCE, capture_output=True, text=True)
    return [line for line in done.stdout.splitlines() if line.strip()]


def globals_at_import(ref):
    """Module-level paths a caller can rebind, as the control requires."""
    pack = parity.load_reference(PHASE, "42-agent-workbench-capstone", "main")
    gate = parity.load_reference(PHASE, "38-verification-gates", "main")
    return {"pack": isinstance(getattr(pack, "PACK", None), Path),
            "overrides": isinstance(getattr(gate, "OVERRIDES_PATH", None), Path),
            "stage_controls": ref.required_controls("prototype")}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    total, rebinds = redirects()
    dirty = untouched()
    shapes = globals_at_import(ref)
    controls = shapes["stage_controls"]
    return {
        "solutions": len(solutions()), "redirects": total, "rebinds": rebinds,
        "dirty": dirty, "clean": not dirty,
        "pack_global": shapes["pack"], "overrides_global": shapes["overrides"],
        "controls": controls, "control_count": len(controls),
        "enforceable": sum(any(word in control for word in
                               ("path", "directory", "read-only", "token"))
                           for control in controls),
        "stage": ref.choose_stage(ref.BuildDecision(
            "Can the pack install cleanly?", False, False, 2, True, False)),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 25 solutions redirect into tempfile and the reference tree is clean",
            all([result["redirects"] == 25, result["rebinds"] == 4,
                 result["clean"] is True, result["dirty"] == []]),
            f"{result['redirects']} of {result['solutions']} shipped solutions redirect a "
            f"path into tempfile, {result['rebinds']} of them by rebinding the pack "
            f"assembler's own output global, and the reference phase reports "
            f"{len(result['dirty'])} modified paths",
        ),
        practice.Check(
            "FINDING: the control works because the production path is a module global",
            all([result["pack_global"] is True, result["overrides_global"] is True]),
            "PACK and OVERRIDES_PATH are Paths computed at import time, so a caller can "
            "rebind them; a writer computing its destination inside the function would "
            "leave nowhere to intervene and force a fork of the code being imported",
        ),
        practice.Check(
            "FINDING: required_controls('prototype') names none of this",
            all([result["control_count"] == 3, result["enforceable"] == 0,
                 result["stage"] == "prototype"]),
            f"the prototype stage draws {result['controls']} -- "
            f"{result['enforceable']} of {result['control_count']} mentioning a path, a "
            "directory or a credential -- so the list is a banner with three lines",
        ),
        practice.Check(
            "FINDING: the check that the control held is cheaper than the control",
            all([result["clean"] is True, result["redirects"] == 25]),
            f"one git status over the reference phase reports {len(result['dirty'])} "
            f"modified paths, which is the receipt that {result['redirects']} redirections "
            "worked -- and it is the artifact the stage decision should carry",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
