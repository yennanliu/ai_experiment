"""Exercise 5 — the rollback is one commit, and the receipt has to name the dependents.

    Design a rollback receipt for the bounded pilot.

Reading of the exercise: a receipt records what a rollback would undo and
what it would break. In this repository the unit of the pilot is one lesson's
practice directory, so both halves are computable from the tree.

**ANSWER: reverting one lesson takes out 9 files and breaks 7 other
solutions.** The unit is Lesson 47's practice directory -- **8** files, plus
the top-level README a lesson regenerates -- and **7** solution files in
lessons 48, 49, 51 and 52 name that directory and read its files, so a revert
that looks local takes those measurements with it. The receipt has to carry
the dependent list, not just the unit.

**FINDING: the dependency runs one way and grows with the phase.** Lessons
43 and 44 have **9** dependents each; lessons 48 through 52 have **0**
between them. Reverting an early lesson is expensive and
reverting a late one is free, which is the opposite of how the commit log
reads -- and nothing in the stage plan records it.

**FINDING: `required_controls("pilot")` names rollback and not its
receipt.** The pilot draws **5** controls including "rollback" and "audit
trail"; neither is a record of what a rollback would cost. `plan` returns
**3** keys, so the receipt lives outside the document that requires it.

**FINDING: the cheap half of the receipt is the one nobody writes.**
Recording the commit is one line; recording the dependents is one grep over
the tree, and it is the half that decides whether the rollback is a revert or
a project. Computing it for all **10** finished lessons takes one pass and
finds **21** dependency edges.

Structure: `footprint()` is what a revert removes; `dependents()` reads the
real cross-lesson references; `receipt()` assembles both.
"""

from __future__ import annotations

from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "53-prototype-pilot-or-production"
BASE = Path(__file__).resolve().parents[2]
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "demos").is_dir())
FINISHED = tuple(f"{number}-" for number in range(43, 53))
UNIT = "47-outcomes-before-output"


def lessons():
    return sorted(path.name for path in BASE.iterdir()
                  if path.is_dir() and path.name.startswith(FINISHED))


def solutions():
    """The finished pilot's own solutions: lessons 43 to 52, nothing later."""
    return [path for path in sorted(BASE.glob("*/practice/ex0*.py"))
            if path.parents[1].name.startswith(FINISHED)]


def dependents(lesson):
    """Solutions in other lessons that name this one's directory."""
    return sorted(path.parents[1].name[:2] for path in solutions()
                  if lesson in path.read_text(encoding="utf-8")
                  and path.parents[1].name != lesson)


def footprint(lesson):
    """What a revert of this lesson would take out: its directory, plus the
    top-level README that scripts/coverage.py regenerates with every lesson."""
    folder = BASE / lesson / "practice"
    own = sorted(path.relative_to(BASE).as_posix() for path in folder.rglob("*")
                 if path.is_file() and "__pycache__" not in path.parts)
    return own + ["README.md"]


def receipt(lesson):
    files = footprint(lesson)
    return {"lesson": lesson, "files": len(files),
            "own_directory": sum(lesson in path for path in files),
            "dependents": dependents(lesson)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    row = receipt(UNIT)
    edges = {lesson: len(dependents(lesson)) for lesson in lessons()}
    pilot = ref.plan(ref.BuildDecision("Would a reader trust a generated answer?",
                                       True, True, 3, True, False))
    controls = ref.required_controls("pilot")
    return {
        **row, "lessons": len(edges), "edges": sum(edges.values()),
        "resolve": sum((BASE / path).exists() or path == "README.md"
                       for path in footprint(UNIT)),
        "early": [edges[name] for name in lessons()[:2]],
        "late": sum(edges[name] for name in lessons()[5:]),
        "controls": controls, "control_count": len(controls),
        "names_rollback": "rollback" in controls,
        "names_receipt": any("receipt" in control or "cost" in control
                             for control in controls),
        "keys": sorted(pilot),
        "stage": pilot["stage"],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: reverting one lesson touches its directory and breaks 7 solutions",
            all([result["lesson"] == UNIT, result["files"] == 9,
                 result["own_directory"] == result["files"] - 1,
                 result["resolve"] == result["files"],
                 len(result["dependents"]) == 7,
                 sorted(set(result["dependents"])) == ["48", "49", "51", "52"]]),
            f"the revert takes out {result['files']} files, all but the generated README "
            f"inside {result['lesson']}, while {len(result['dependents'])} solutions in "
            f"lessons {sorted(set(result['dependents']))} read its files",
        ),
        practice.Check(
            "FINDING: the dependency runs one way and grows with the phase",
            all([result["early"] == [9, 9], result["late"] == 0,
                 result["lessons"] == 10, result["edges"] == 37]),
            f"the two earliest finished lessons have {result['early']} dependents and the "
            f"last five have {result['late']} between them; across {result['lessons']} "
            f"lessons there are {result['edges']} edges, and reverting an early lesson is "
            "expensive exactly where the log looks safest",
        ),
        practice.Check(
            "FINDING: required_controls('pilot') names rollback and not its receipt",
            all([result["control_count"] == 5, result["names_rollback"] is True,
                 result["names_receipt"] is False, len(result["keys"]) == 3,
                 result["stage"] == "pilot"]),
            f"the pilot draws {result['controls']}; rollback is named and what it would "
            f"cost is not, and plan returns {result['keys']}, so the receipt lives outside "
            "the document that requires it",
        ),
        practice.Check(
            "FINDING: the cheap half of the receipt is the one nobody writes",
            all([result["edges"] == 37, result["lessons"] == 10,
                 result["files"] == 9]),
            f"recording the commit is one line; recording the dependents is one pass over "
            f"the tree that finds {result['edges']} edges across {result['lessons']} "
            "lessons, and it is the half that decides whether a rollback is a revert or a "
            "project",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
