"""Exercise 1 — the five dimensions score a forked copy ten out of ten.

    Add a sixth dimension specific to your product domain. Defend why it is
    not absorbed by the existing five.

Reading of the exercise: "not absorbed" is a testable claim, not a rhetorical
one. Two diffs that differ only in the new dimension must score identically on
the five; if any of the five moves, the dimension is already covered. This
repository's domain rule is `DESIGN D5` -- a solution imports the lesson's own
`code/`, it never forks it -- so the sixth dimension is **reference fidelity**.

**ANSWER: a forked copy scores 10/10 on the five dimensions, the same as the
faithful diff.** Run both through `review`: identical `problem_fit`,
`scope_discipline`, `assumptions`, `verification_quality`,
`handoff_readiness`, identical total, identical verdict `pass`. The fork
duplicates **143** lines of the lesson's reference module and every dimension
is blind to it, which is the non-absorption proof the exercise asks for.

**FINDING: no scorer opens a file, so the dimension cannot be computed from
`ReviewerInputs` at all.** `diff_summary` is typed `dict[str, list[str]]` --
names, not contents -- and **0** of the **5** scorers reads anything but names,
`state`, `feedback` and `verdict`. Reference fidelity is a property of the
bytes, so adding the dimension means widening the inputs first: the same shape
Lesson 38 hit when an exemption needed a timestamp the scope report never
carried.

**FINDING: the real solutions in this repo score 2, and they do it by import.**
The **4** shipped Lesson 38 solutions all call `parity.load_reference`, hold
**0** copied reference definitions, and score 2/2. The forked variant scores 0.
The dimension separates them on a single mechanical fact, which is what keeps
it a check rather than a judgment.

**FINDING: naming a file after the goal scores better than solving it.**
`score_problem_fit` counts goal words appearing in file *names*, so the
lesson's own clean case -- `app/signup.py` plus `tests/test_signup.py` --
scores **1**, while one empty file named `signup_validation_input.py` scores
**2**. Verbosity bias with the model removed: the rubric rewards the label.

Structure: `fidelity()` is the proposed scorer; `pair()` builds the two diffs
that differ only in it.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "39-reviewer-agent"
GOAL = "add input validation to signup"
SHIPPED = Path(__file__).resolve().parent.parent.parent / "38-verification-gates" / "practice"


def fidelity(sources):
    """Proposed sixth dimension: does the solution import the lesson's code or copy it?"""
    copied = sum(len(re.findall(r"^def \w+|^class \w+", text, re.M))
                 for name, text in sources.items() if "load_reference" not in text)
    imports = sum("load_reference" in text for text in sources.values())
    if copied:
        return 0, f"{copied} definitions copied from the reference module"
    return (2, f"{imports} of {len(sources)} files import the reference") if imports else (
        1, "no reference import and no copy: nothing to compare")


def build(ref, touched):
    return ref.ReviewerInputs(
        task_id="T-001", goal=GOAL, diff_summary={"touched": touched},
        state={"active_task_id": None, "assumptions": ["email and password only"],
               "next_action": "pick next task from board"},
        feedback=[{"command": "pytest", "exit_code": 0}],
        verdict={"passed": True, "findings": []})


def pair(ref):
    """Two diffs identical to the five dimensions, opposite on reference fidelity."""
    ref_source = inspect.getsource(ref)
    touched = ["practice/ex01_signup_validation_input.py"]
    faithful = {touched[0]: "from harness import parity\nref = parity.load_reference(...)\n"}
    forked = {touched[0]: ref_source}
    reports = [ref.review(build(ref, touched)) for _ in range(2)]
    return {"scores": [[(d.name, d.score) for d in r.dimensions] for r in reports],
            "totals": [r.total for r in reports], "verdicts": [r.verdict for r in reports],
            "fidelity": [fidelity(faithful), fidelity(forked)],
            "forked_lines": len(ref_source.splitlines())}


def shipped_scores():
    """Reference fidelity over the four solutions this repo actually shipped."""
    files = sorted(SHIPPED.glob("ex0*.py"))
    return len(files), fidelity({f.name: f.read_text() for f in files})


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    sources = {fn.__name__: inspect.getsource(fn) for fn in ref.SCORERS}
    two = pair(ref)
    count, shipped = shipped_scores()
    return {
        **two, "scorers": len(ref.SCORERS),
        "read_bytes": sum("read_text" in s or "open(" in s for s in sources.values()),
        "diff_type": ref.ReviewerInputs.__annotations__["diff_summary"],
        "shipped_count": count, "shipped": shipped,
        "clean_fit": ref.review(build(ref, ["app/signup.py", "tests/test_signup.py"]
                                      )).dimensions[0].score,
        "named_fit": ref.review(build(ref, ["signup_validation_input.py"])).dimensions[0].score,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: a forked copy scores 10/10, the same as the faithful diff",
            all([result["scores"][0] == result["scores"][1],
                 result["totals"] == [10, 10], result["verdicts"] == ["pass", "pass"],
                 result["fidelity"][0][0] == 2, result["fidelity"][1][0] == 0]),
            f"both diffs score {result['totals']} and land on {result['verdicts']} with "
            f"identical dimensions, while reference fidelity reads "
            f"{result['fidelity'][0][0]} against {result['fidelity'][1][0]} -- the fork "
            f"duplicates {result['forked_lines']} lines and no dimension sees it",
        ),
        practice.Check(
            "FINDING: no scorer opens a file, so the dimension needs wider inputs",
            all([result["scorers"] == 5, result["read_bytes"] == 0,
                 result["diff_type"] == "dict[str, list[str]]"]),
            f"{result['read_bytes']} of {result['scorers']} scorers read file contents and "
            f"diff_summary is typed {result['diff_type']} -- names, not bytes. The "
            "dimension means widening ReviewerInputs first, the way Lesson 38's exemption "
            "needed a timestamp the scope report never carried",
        ),
        practice.Check(
            "FINDING: the shipped solutions score 2, by import",
            all([result["shipped_count"] == 4, result["shipped"][0] == 2,
                 "4 of 4" in result["shipped"][1]]),
            f"the {result['shipped_count']} Lesson 38 solutions score "
            f"{result['shipped'][0]}/2 -- {result['shipped'][1]} -- against 0 for the fork. "
            "One mechanical fact separates them, which keeps the dimension a check",
        ),
        practice.Check(
            "FINDING: naming a file after the goal scores better than solving it",
            all([result["clean_fit"] == 1, result["named_fit"] == 2]),
            f"the lesson's own clean case scores {result['clean_fit']}/2 on problem_fit "
            f"while one file named after the goal scores {result['named_fit']}/2 -- "
            "score_problem_fit counts goal words in file names, so the rubric rewards the "
            "label. Verbosity bias with the model taken out",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
