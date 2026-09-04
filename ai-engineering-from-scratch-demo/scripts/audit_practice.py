#!/usr/bin/env python3
"""D14's mechanical ceilings, plus D10 and D12 structure. No human in the loop.

Exit non-zero on any violation. Every rule here is one `DESIGN §6` lists as a
rejection reason, so a solution that passes this passes the generation gate.
"""

from __future__ import annotations

import ast
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from harness import manifest, parity, runner  # noqa: E402

MAX_LINES = 120
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


def audit_solution(path: pathlib.Path, exercise) -> list:
    problems = []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if len(lines) > MAX_LINES:
        problems.append(f"{len(lines)} lines > ceiling {MAX_LINES} (D14)")
    for banned in BANNED:
        if banned in text:
            problems.append(f"contains {banned!r} — a surviving scaffold marker")
    tree = ast.parse(text)
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
    return problems


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


def audit_lesson(man_path: pathlib.Path) -> list:
    pack = manifest.load_practice(man_path)
    directory = man_path.parent
    problems = []
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
        problems += [f"{ex.filename}: {p}" for p in audit_solution(path, ex)]
        for fixture in ex.fixtures:
            if not (directory / fixture).is_file():
                problems.append(f"ex{ex.index:02d}: fixture {fixture} missing")
            elif '"_meta"' not in (directory / fixture).read_text(encoding="utf-8"):
                problems.append(f"ex{ex.index:02d}: fixture {fixture} is unlabelled (D14)")
    tests = directory / "tests"
    n_tests = len(list(tests.glob("test_*.py"))) if tests.is_dir() else 0
    if n_tests < 1:
        problems.append(f"{directory}: no tests/test_*.py")
    return problems


def main(argv=None) -> int:
    paths = runner._manifests(argv[0] if argv else None)
    if not paths:
        print("audit_practice: no manifests found", file=sys.stderr)
        return 1
    failed = 0
    for path in paths:
        problems = audit_lesson(path)
        label = path.parent.parent.name
        if problems:
            failed += 1
            print(f"FAIL {label}")
            for problem in problems:
                print(f"     {problem}")
        else:
            print(f"ok   {label}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
