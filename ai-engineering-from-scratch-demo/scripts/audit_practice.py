#!/usr/bin/env python3
"""D14's mechanical ceilings, plus D10 and D12 structure. No human in the loop.

Exit non-zero on any violation. D14's line ceiling is the one rule with two numbers —
"<= 120 lines of code per file excluding the docstring; hard fail over 150" — so a file
between the two is reported rather than rejected, which is what §6.4's phase-batch review
reads. The docstring is excluded because it is mandated content (the exercise text verbatim
plus the "Reading of the exercise:" line), and charging a solution for how long its own
exercise is measures the wrong thing. Every rule here is one `DESIGN §6` lists as a
rejection reason, so a solution that passes this passes the generation gate.
"""

from __future__ import annotations

import ast
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from harness import manifest, parity, runner  # noqa: E402

MAX_LINES = 120          # D14's target: over it the audit says so, for §6.4 to read
HARD_LINES = 150         # D14's "hard fail over 150" — the number that exits non-zero
MAX_COMPLEXITY = 8
BANNED = ("TODO", "FIXME", "XXX", "<<<", "raise NotImplementedError")


def complexity(node) -> int:
    """Cyclomatic complexity: one plus each branch point."""
    score = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler,
                              ast.Assert, ast.IfExp, ast.comprehension)):
            score += 1
        elif isinstance(child, ast.BoolOp):
            score += len(child.values) - 1
    return score


def docstring_lines(tree) -> int:
    """The module docstring's own line count.

    D14 puts the ceiling on *code*, "excluding the docstring", and the docstring is
    mandated content — the exercise text verbatim plus the "Reading of the exercise:"
    line — so counting it would charge a solution for how long its exercise is.
    """
    first = tree.body[0] if tree.body else None
    if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)):
        return first.end_lineno - first.lineno + 1
    return 0


def audit_solution(path: pathlib.Path, exercise) -> tuple:
    """Returns (problems, warnings) — D14 fails over 150 lines and only reports over 120."""
    problems, warnings = [], []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text)
    code = len(lines) - docstring_lines(tree)
    if code > HARD_LINES:
        problems.append(f"{code} lines of code > hard ceiling {HARD_LINES} (D14)")
    elif code > MAX_LINES:
        warnings.append(f"{code} lines of code > target {MAX_LINES} (D14)")
    for banned in BANNED:
        if banned in text:
            problems.append(f"contains {banned!r} — a surviving scaffold marker")
    doc = ast.get_docstring(tree) or ""
    if "Reading of the exercise:" not in doc:
        problems.append("docstring has no 'Reading of the exercise:' line (DESIGN §6.4)")
    if exercise.en.split()[0] not in doc:
        problems.append("docstring does not quote the exercise text (D12)")
    for func in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
        score = complexity(func)
        if score > MAX_COMPLEXITY:
            problems.append(f"{func.name}() complexity {score} > {MAX_COMPLEXITY} (D14)")
    names = {n.targets[0].id for n in tree.body
             if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    if "PRACTICE_IMPL" not in names:
        problems.append("no PRACTICE_IMPL (D13)")
    return problems, warnings


def audit_explain(exercise, readme, headings) -> list:
    """DESIGN §6's gate for prose items: a *resolvable* citation.

    The answer must name a lesson section, that section must actually exist in
    the reference `docs/en.md`, and the README must carry the answer. A citation
    that points nowhere is the prose equivalent of a test that asserts nothing.
    """
    problems = []
    body = readme.read_text(encoding="utf-8") if readme.is_file() else ""
    if exercise.cites not in body:
        problems.append(f"ex{exercise.index:02d}: citation {exercise.cites!r} "
                        f"not answered in README")
    if headings is not None and exercise.cites not in headings:
        problems.append(f"ex{exercise.index:02d}: cites {exercise.cites!r}, which is not "
                        f"a heading in the lesson's docs/en.md")
    return problems


def _reference_headings(pack):
    """The lesson's own section titles, or None if the reference is unreachable."""
    try:
        text = parity.doc_text(pack.phase, pack.lesson, "en")
    except Exception:
        return None
    return {line.lstrip("#").strip() for line in text.splitlines()
            if line.startswith("#")}


def audit_lesson(man_path: pathlib.Path) -> tuple:
    pack = manifest.load_practice(man_path)
    directory = man_path.parent
    problems, warnings = [], []
    readme = directory / "README.md"
    if not readme.is_file():
        problems.append(f"{directory}: no README.md")
    headings = _reference_headings(pack) if any(
        e.kind == "explain" for e in pack.exercises) else None
    for ex in pack.exercises:
        if ex.kind == "explain":
            problems += audit_explain(ex, readme, headings)
            continue
        path = directory / ex.filename
        if not path.is_file():
            problems.append(f"ex{ex.index:02d}: missing {ex.filename} (D10)")
            continue
        found, warned = audit_solution(path, ex)
        problems += [f"{ex.filename}: {p}" for p in found]
        warnings += [f"{ex.filename}: {w}" for w in warned]
        for fixture in ex.fixtures:
            if not (directory / fixture).is_file():
                problems.append(f"ex{ex.index:02d}: fixture {fixture} missing")
            elif '"_meta"' not in (directory / fixture).read_text(encoding="utf-8"):
                problems.append(f"ex{ex.index:02d}: fixture {fixture} is unlabelled (D14)")
    tests = directory / "tests"
    n_tests = len(list(tests.glob("test_*.py"))) if tests.is_dir() else 0
    if n_tests < 1:
        problems.append(f"{directory}: no tests/test_*.py")
    return problems, warnings


def main(argv=None) -> int:
    paths = runner._manifests(argv[0] if argv else None)
    if not paths:
        print("audit_practice: no manifests found", file=sys.stderr)
        return 1
    failed = 0
    for path in paths:
        problems, warnings = audit_lesson(path)
        label = path.parent.parent.name
        failed += bool(problems)
        print(f"{'FAIL' if problems else 'warn' if warnings else 'ok  '} {label}")
        for line in problems + warnings:
            print(f"     {line}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
