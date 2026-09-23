"""Exercise 5 — the anti-hacking clause is declared and unenforced.

    For a domain you know, write the evaluator signature you would use.
    Include (a) correctness conditions, (b) performance metric, (c) held-out
    input generation rule, (d) at least one anti-reward-hacking check.

Reading of the exercise: the domain has to be one whose evaluator already
exists, or the four clauses are wishes. The domain is this repository's own
solution gate, which scores a candidate file the way the lesson's `mse`
scores a candidate expression -- so each clause can be pointed at a line
rather than described.

**ANSWER: `evaluate(path, exercise, upstream) -> Verdict`, and three of its
four clauses already exist here.** (a) correctness is
`practice.grade_file`'s list of `Check`s, all of which must pass; (b) the
performance metric is `audit_practice`'s ceilings -- complexity **8** and a
line count whose enforced number is **150**, with **120** reported as a
target; (c) the held-out rule is the exercise text re-read from
upstream at scoring time and hashed, in `tests/test_practice.py`; (d) the
anti-reward-hacking check is that the solution must name a symbol from the
lesson's own `code/`, and it is the one that does not exist.

**FINDING: clause (d) is declared and never enforced.** `uses_reference` is a
manifest field: `harness/manifest.py` names it **4** times, and it appears
**0** times in `audit_practice.py` and **0** times in `check_deps.py`. A
solution that forked the lesson's implementation and asserted against its own
copy would pass every gate in this repository.

**FINDING: the lesson's own evaluator has one of the four.** `mse` is a
performance metric. Its correctness handling is an `except` returning
`inf` -- an error guard, not a condition a candidate must satisfy. Its
held-out inputs are a literal list of **7** floats, not a generation rule.
And it has no anti-hacking clause at all, which is what makes the
48-multiplication result checkable and most domains not.

**FINDING: the clauses hold on this lesson's own solutions.** Applied
statically to the **4** code files here, **4** define `PRACTICE_IMPL`, **4**
load the reference rather than copying it, and **4** sit inside both
ceilings. That is what clause (d) would look like if the gate ran it, and it
took one function to write.

Structure: `clause()` is the static evaluator; `gate_mentions()` counts where
each clause is implemented in this repository.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "03-alphaevolve-evolutionary-coding"

HERE = pathlib.Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "harness").is_dir())
TARGET_LINES, HARD_LINES, MAX_COMPLEXITY = 120, 150, 8
TEST_XS = 7                      # the literal held-out points run_loop ships


def branches(node):
    kinds = (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.Assert, ast.IfExp,
             ast.comprehension)
    return 1 + sum(isinstance(child, kinds) for child in ast.walk(node)) + sum(
        len(child.values) - 1 for child in ast.walk(node) if isinstance(child, ast.BoolOp))


def clause(path):
    """The four-clause evaluator, statically applied to one solution file."""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    doc = tree.body[0]
    code = len(text.splitlines()) - (doc.end_lineno - doc.lineno + 1)
    worst = max(branches(node) for node in tree.body if isinstance(node, ast.FunctionDef))
    return {
        "correct": "PRACTICE_IMPL" in text,
        "performance": code <= HARD_LINES and worst <= MAX_COMPLEXITY,
        "anti_hacking": "parity.load_reference" in text,
    }


def solutions():
    return sorted(HERE.glob("ex*.py"))


def gate_mentions(name):
    return (ROOT / name).read_text(encoding="utf-8").count("uses_reference")


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    scored = [clause(path) for path in solutions()]
    source = inspect.getsource(ref.mse)
    tests = (HERE / "tests" / "test_practice.py").read_text(encoding="utf-8")
    return {
        "files": len(scored),
        "correct": sum(row["correct"] for row in scored),
        "performance": sum(row["performance"] for row in scored),
        "anti_hacking": sum(row["anti_hacking"] for row in scored),
        "manifest_mentions": gate_mentions("harness/manifest.py"),
        "audit_mentions": gate_mentions("scripts/audit_practice.py"),
        "deps_mentions": gate_mentions("scripts/check_deps.py"),
        "holdout_rule": tests.count("parity.doc_text") >= 1 and "spec_hash" in tests,
        "mse_guard": source.count("except"),
        "mse_returns_inf": source.count('float("inf")'),
        "mse_conditions": source.count("assert"),
        "test_points": TEST_XS,
        "ceilings": [TARGET_LINES, HARD_LINES, MAX_COMPLEXITY],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: three of the four clauses already exist in this gate",
            all([result["holdout_rule"], result["ceilings"] == [120, 150, 8],
                 result["correct"] == result["files"]]),
            f"correctness is grade_file's Check list, the performance metric is the "
            f"{result['ceilings']} ceilings, the held-out rule is the upstream re-read "
            f"and spec hash in tests/test_practice.py -- and the anti-hacking clause is "
            "the missing one",
        ),
        practice.Check(
            "FINDING: clause (d) is declared and never enforced",
            all([result["manifest_mentions"] == 4, result["audit_mentions"] == 0,
                 result["deps_mentions"] == 0]),
            f"uses_reference appears {result['manifest_mentions']} times in the "
            f"manifest parser and {result['audit_mentions']} and "
            f"{result['deps_mentions']} times in the two gates, so a solution that "
            "forked the lesson's code would pass everything",
        ),
        practice.Check(
            "FINDING: the lesson's own evaluator has one of the four",
            all([result["mse_guard"] == 1, result["mse_returns_inf"] == 1,
                 result["mse_conditions"] == 0, result["test_points"] == 7]),
            f"mse is a performance metric; its correctness handling is "
            f"{result['mse_guard']} except returning inf and "
            f"{result['mse_conditions']} conditions, and its held-out inputs are "
            f"{result['test_points']} literal floats rather than a rule",
        ),
        practice.Check(
            "FINDING: the clauses hold on this lesson's own solutions",
            all([result["files"] == 4, result["correct"] == 4,
                 result["anti_hacking"] == 4, result["performance"] == 4]),
            f"of the {result['files']} code files here, {result['correct']} define "
            f"PRACTICE_IMPL, {result['anti_hacking']} load the reference rather than "
            f"copying it and {result['performance']} sit inside both ceilings",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
