"""Exercise 3 — the dimension the reviewer exists for is the one it cannot be sure of.

    Add a `confidence` field per dimension. Refuse to ship the report when
    confidence in the lowest dimension is below 0.6.

Reading of the exercise: confidence has to come from somewhere mechanical, or
it is a second number the scorer invents. The honest source is the branch each
scorer took: reading an exit code or a `findings` list is a fact (1.0);
inferring intent from a file name, or landing in a branch whose own note says
the scorer cannot tell, is not (0.5).

**ANSWER: the rule refuses to ship 3 of 3 runs, including the clean one.**
Score the lesson's two demo cases and the clean baseline, attach confidence by
branch, and `min(confidence)` is **0.5** every time. `problem_fit` is derived
from file names on every input the reviewer can be given, so the refusal
threshold of 0.6 is never met and the reviewer emits nothing, ever.

**FINDING: the never-confident dimension is the one the reviewer exists for.**
The lesson's own framing is that acceptance cannot ask "did this solve the
right problem" -- `problem_fit` is that question, and it is answered by
`sum(any(k in f.lower() for f in files) for k in keywords)` over **3**
goal-derived keywords and the *names* of the touched files. **1** of the **5**
dimensions is this weak, and dropping it to satisfy the threshold deletes the
reason to run a reviewer at all.

**FINDING: `score_assumptions` returns the same 1 for two opposite worlds.**
Its note reads "no assumptions recorded; either work was trivial or
undocumented" -- one score for "nothing to say" and "said nothing", which is a
scorer publishing its own confidence in prose while the dataclass has nowhere
to put it. **1** of the **5** scorers admits this in prose; the other weak
dimension, `problem_fit`, is exactly as unsure and never says so.

**FINDING: there is no channel to refuse in.** `ReviewReport` has **4** fields
and `review` emits **3** verdict literals -- `pass`, `soft_fail`, `hard_fail`
-- none of which means "no verdict". Refusing per dimension instead of per
report keeps one: drop `problem_fit` and the clean case rescores **8/8**,
still a pass, with the unanswerable question named rather than silently worth
two points.

Structure: `confidence()` reads the branch each scorer took; `runs()` scores
the clean baseline and the lesson's two demo cases.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "39-reviewer-agent"
GOAL = "add input validation to signup"
FLOOR = 0.6


def build(ref, **kw):
    base = dict(task_id="T-001", goal=GOAL,
                diff_summary={"touched": ["app/signup.py", "tests/test_signup.py"]},
                state={"active_task_id": None, "assumptions": ["email and password only"],
                       "next_action": "pick next task from board"},
                feedback=[{"command": "pytest", "exit_code": 0}],
                verdict={"passed": True, "findings": []})
    base.update(kw)
    return ref.ReviewerInputs(**base)


def runs(ref):
    """The clean baseline plus the lesson's own two demo cases."""
    wrong = build(ref, task_id="T-002", diff_summary={"touched": ["docs/api.md"]},
                  state={"active_task_id": "T-002", "assumptions": [], "next_action": ""},
                  verdict={"passed": True,
                           "findings": [{"code": "scope.off_scope", "severity": "warn"}]})
    creep = build(ref, task_id="T-003",
                  verdict={"passed": False,
                           "findings": [{"code": "scope.forbidden", "severity": "block"}]})
    return [build(ref), wrong, creep]


def confidence(dimension):
    """0.5 when the branch inferred from names or admitted it could not tell."""
    if dimension.name == "problem_fit":
        return 0.5, "inferred from file names"
    if "either" in dimension.note:
        return 0.5, "the note says the scorer cannot tell"
    return 1.0, "read from an artifact field"


def scored(ref):
    reports = [ref.review(case) for case in runs(ref)]
    rows = []
    for report in reports:
        conf = [confidence(d) for d in report.dimensions]
        rows.append({"total": report.total, "verdict": report.verdict,
                     "min": min(c for c, _ in conf),
                     "weak": [d.name for d, (c, _) in zip(report.dimensions, conf) if c < FLOOR]})
    return reports, rows


def verdict_words(ref):
    """The verdict literals `review` can emit."""
    source = inspect.getsource(ref.review)
    return sorted(word for word in ("pass", "soft_fail", "hard_fail")
                  if f'"{word}"' in source)


def weakness(ref, rows, clean):
    """Which dimensions fall under the floor, and what dropping them costs."""
    kept = [d for d in clean.dimensions if confidence(d)[0] >= FLOOR]
    return {"weak_names": sorted({name for row in rows for name in row["weak"]}),
            "admits": sum("either" in inspect.getsource(fn) for fn in ref.SCORERS),
            "kept": (sum(d.score for d in kept), 2 * len(kept))}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    reports, rows = scored(ref)
    fit_source = inspect.getsource(ref.score_problem_fit)
    return {
        **weakness(ref, rows, reports[0]),
        "runs": len(rows), "shipped": sum(row["min"] >= FLOOR for row in rows),
        "mins": [row["min"] for row in rows],
        "keywords": len([w for w in GOAL.split() if len(w) > 4]),
        "names_only": "f.lower()" in fit_source and "read_text" not in fit_source,
        "scorers": len(ref.SCORERS),
        "report_fields": len(ref.ReviewReport.__dataclass_fields__),
        "verdicts": verdict_words(ref),
        "clean_total": reports[0].total,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the rule refuses to ship 3 of 3 runs, including the clean one",
            all([result["runs"] == 3, result["shipped"] == 0,
                 result["mins"] == [0.5, 0.5, 0.5]]),
            f"across {result['runs']} runs the minimum confidence is {result['mins']} and "
            f"{result['shipped']} reports clear the {FLOOR} floor, because problem_fit is "
            "derived from file names on every input the reviewer can be given",
        ),
        practice.Check(
            "FINDING: the never-confident dimension is the one the reviewer exists for",
            all([result["weak_names"] == ["assumptions", "problem_fit"],
                 result["keywords"] == 3, result["names_only"] is True]),
            f"problem_fit answers 'did this solve the right problem' by matching "
            f"{result['keywords']} goal keywords against file names -- "
            f"{result['weak_names']} are the weak dimensions, and dropping problem_fit to "
            "clear the threshold deletes the reason to run a reviewer",
        ),
        practice.Check(
            "FINDING: score_assumptions returns the same 1 for two opposite worlds",
            all([result["admits"] == 1, result["scorers"] == 5,
                 "assumptions" in result["weak_names"]]),
            f"{result['admits']} of {result['scorers']} scorers has a branch whose note "
            "says 'either work was trivial or undocumented' -- one score for 'nothing to "
            "say' and 'said nothing', with nowhere in DimensionScore to put the doubt",
        ),
        practice.Check(
            "FINDING: there is no channel to refuse in",
            all([result["report_fields"] == 4, len(result["verdicts"]) == 3,
                 result["clean_total"] == 9, result["kept"] == (8, 8)]),
            f"ReviewReport has {result['report_fields']} fields and review emits "
            f"{len(result['verdicts'])} verdict literals, none meaning 'no verdict'. "
            f"Refusing per dimension keeps one: the clean case goes {result['clean_total']}/10 "
            f"to {result['kept'][0]}/{result['kept'][1]} with the unanswerable question named",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
