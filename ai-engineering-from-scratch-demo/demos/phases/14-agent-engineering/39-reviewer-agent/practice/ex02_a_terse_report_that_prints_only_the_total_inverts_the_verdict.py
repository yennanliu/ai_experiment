"""Exercise 2 — a terse report that prints only the total inverts the verdict.

    Run the reviewer with two different system prompts (terse, verbose).
    Which produces a report a human is more likely to read?

Reading of the exercise: the dimension scorers here are deterministic stubs, so
"system prompt" is the report *renderer* -- the same `ReviewReport` written two
ways. That makes the question answerable without a model: render both, measure
what each one costs a reader and what each one omits.

**ANSWER: terse, at 2 lines against 12, but only because it keeps the zero.**
The verbose rendering of one report is **12** lines and **322** characters; the
terse one is **2** lines and **74**, a **6.0x** cut. The whole of the
difference is the four passing dimensions, which carry no action. What terse
must not cut is the dimension that scored 0, because the verdict is not a
function of the total.

**FINDING: 8 out of 10 hard-fails while 7 out of 10 passes.** `review` fails
hard on `any(d.score == 0)` before it compares the total, so a run scoring
2/2/2/2/0 lands on `hard_fail` at **8** points and a run scoring 2/1/2/1/1
lands on `pass` at **7**. A terse report that prints `total` and `verdict` and
nothing else reads as a bug; it needs the zero to be legible.

**FINDING: 3 of the 5 notes are counts, so verbose buys length and not
judgment.** The notes say "keyword hits across touched files: 3", "off-scope
warnings: 0", "1 assumptions recorded" -- numbers the reader could have read
off the artifacts. **2** carry a qualitative claim, and one of those ("either
work was trivial or undocumented") is the scorer saying it cannot tell.

**FINDING: on a passing run the verbose report has 0 actionable lines.** The
clean demo renders **13** lines, **0** of which name something to do. The terse
renderer collapses the same report to **1** line. Every line a reviewer emits
on a green run is a line spent teaching the reader to skim the next one.

Structure: `terse()` and `verbose()` are the two renderers; `cases()` builds
the 8-point hard fail and the 7-point pass.
"""

from __future__ import annotations

import inspect
import re

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "39-reviewer-agent"
GOAL = "add input validation to signup"
NAMED = ["signup_validation_input.py"]


def build(ref, **kw):
    base = dict(task_id="T-001", goal=GOAL, diff_summary={"touched": NAMED},
                state={"active_task_id": None, "assumptions": ["email and password only"],
                       "next_action": "pick next task from board"},
                feedback=[{"command": "pytest", "exit_code": 0}],
                verdict={"passed": True, "findings": []})
    base.update(kw)
    return ref.ReviewerInputs(**base)


def cases(ref):
    """The clean run, an 8-point hard fail, and a 7-point pass."""
    clean = build(ref)
    eight = build(ref, state={"active_task_id": None, "assumptions": ["a"], "next_action": ""})
    seven = build(ref, state={"active_task_id": "T-001", "assumptions": ["a"],
                              "next_action": "keep going"},
                  feedback=[{"command": "pytest", "exit_code": 0},
                            {"command": "ruff", "exit_code": 1}],
                  verdict={"passed": True, "findings": [{"code": "scope.off_scope"}]})
    return [ref.review(case) for case in (clean, eight, seven)]


def verbose(report):
    """Every dimension with its note, the shape `main()` prints today."""
    lines = [f"# Review {report.task_id}", "", f"total: {report.total}/10",
             f"verdict: {report.verdict}", "", "## Dimensions", ""]
    lines += [f"- {d.name}: {d.score}/2 -- {d.note}" for d in report.dimensions]
    return "\n".join(lines + [""])


def terse(report):
    """Verdict, the total, and only the dimensions a human has to act on."""
    lines = [f"{report.task_id}: {report.verdict} ({report.total}/10)"]
    lines += [f"  {d.name} {d.score}/2 -- {d.note}" for d in report.dimensions if d.score < 2]
    return "\n".join(lines)


def shape(report):
    return {"verbose_lines": len(verbose(report).strip().splitlines()),
            "verbose_chars": len(verbose(report)),
            "terse_lines": len(terse(report).splitlines()),
            "terse_chars": len(terse(report))}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    clean, eight, seven = cases(ref)
    notes = [d.note for d in clean.dimensions]
    sizes = shape(eight)
    return {
        **sizes, "ratio": round(sizes["verbose_lines"] / sizes["terse_lines"], 1),
        "eight": (eight.total, eight.verdict), "seven": (seven.total, seven.verdict),
        "zero_first": inspect.getsource(ref.review).index("has_zero")
        < inspect.getsource(ref.review).index("total >= 7"),
        "counted_notes": sum(bool(re.search(r"\d", note)) for note in notes),
        "notes": len(notes),
        "clean_verbose": len(verbose(clean).strip().splitlines()),
        "clean_terse": len(terse(clean).splitlines()),
        "clean_actionable": sum(d.score < 2 for d in clean.dimensions),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: terse, at 2 lines against 12, because it keeps the zero",
            all([result["verbose_lines"] == 12, result["verbose_chars"] == 322,
                 result["terse_lines"] == 2, result["terse_chars"] == 74,
                 result["ratio"] == 6.0]),
            f"one report renders in {result['verbose_lines']} lines and "
            f"{result['verbose_chars']} characters verbose against "
            f"{result['terse_lines']} and {result['terse_chars']} terse, a "
            f"{result['ratio']}x cut, and the whole difference is the passing dimensions",
        ),
        practice.Check(
            "FINDING: 8 out of 10 hard-fails while 7 out of 10 passes",
            all([result["eight"] == (8, "hard_fail"), result["seven"] == (7, "pass"),
                 result["zero_first"] is True]),
            f"review tests any(score == 0) before it compares the total, so "
            f"{result['eight'][0]} points lands on {result['eight'][1]} and "
            f"{result['seven'][0]} points lands on {result['seven'][1]}. A terse report "
            "printing total and verdict alone reads as a bug",
        ),
        practice.Check(
            "FINDING: 3 of the 5 notes are counts",
            all([result["counted_notes"] == 3, result["notes"] == 5]),
            f"{result['counted_notes']} of {result['notes']} notes carry a number the "
            "reader could have read off the artifacts, so the verbose rendering buys length "
            "and not judgment -- and one of the qualitative two says the scorer cannot tell",
        ),
        practice.Check(
            "FINDING: on a passing run the verbose report has 0 actionable lines",
            all([result["clean_verbose"] == 12, result["clean_actionable"] == 0,
                 result["clean_terse"] == 1]),
            f"the clean demo renders {result['clean_verbose']} verbose lines, "
            f"{result['clean_actionable']} of which name something to do, against "
            f"{result['clean_terse']} terse line. Lines spent on a green run teach the "
            "reader to skim the next one",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
