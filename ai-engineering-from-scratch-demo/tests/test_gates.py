"""`PLAN §3`'s exit gate: every gate demonstrated **failing** on a broken fixture.

A gate never seen to fail is not a gate — it is a function that returns 0. Each
test here breaks one rule on purpose and asserts the gate catches exactly it.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import audit_practice  # noqa: E402
import check_deps  # noqa: E402

from harness import coverage, manifest  # noqa: E402

MANIFEST = """\
lesson: broken-lesson
phase: 00-phase
source: phases/00-phase/broken-lesson/docs/en.md
exercises:
  - index: 1
    slug: thing
    kind: code
    tier: T0
    deps_group: none
    en: |
      Do the thing
    zh: |
      做那件事
    verifies: it does the thing
"""

SOLUTION = '''\
"""Exercise 1 — thing.

    Do the thing

Reading of the exercise: do it.
"""
from harness import practice


def solve():
    return 1


def verify(result):
    return [practice.Check("one", result == 1)]


PRACTICE_IMPL = {"solve": solve, "verify": verify}
'''


@pytest.fixture
def lesson(tmp_path):
    directory = tmp_path / "practice"
    (directory / "tests").mkdir(parents=True)
    (directory / "practice.yaml").write_text(MANIFEST, encoding="utf-8")
    (directory / "ex01_thing.py").write_text(SOLUTION, encoding="utf-8")
    (directory / "README.md").write_text("# broken-lesson\n", encoding="utf-8")
    (directory / "tests" / "test_practice.py").write_text("def test_x(): pass\n", encoding="utf-8")
    return directory


def test_a_correct_lesson_passes_the_audit(lesson):
    assert audit_practice.audit_lesson(lesson / "practice.yaml") == []


def test_gate_rejects_a_missing_solution_file(lesson):
    (lesson / "ex01_thing.py").unlink()
    problems = audit_practice.audit_lesson(lesson / "practice.yaml")
    assert any("missing ex01_thing.py" in p for p in problems)


def test_gate_rejects_a_surviving_scaffold_todo(lesson):
    path = lesson / "ex01_thing.py"
    path.write_text(SOLUTION + "\n# TODO: finish this\n", encoding="utf-8")
    problems = audit_practice.audit_lesson(lesson / "practice.yaml")
    assert any("TODO" in p for p in problems)


def test_gate_rejects_a_solution_over_the_line_ceiling(lesson):
    path = lesson / "ex01_thing.py"
    path.write_text(SOLUTION + "\n" + "x = 1\n" * 200, encoding="utf-8")
    problems = audit_practice.audit_lesson(lesson / "practice.yaml")
    assert any("ceiling" in p and "D14" in p for p in problems)


def test_gate_rejects_complexity_over_eight(lesson):
    body = "\n".join(f"    if x == {n}:\n        return {n}" for n in range(12))
    path = lesson / "ex01_thing.py"
    path.write_text(SOLUTION + f"\n\ndef tangled(x):\n{body}\n", encoding="utf-8")
    problems = audit_practice.audit_lesson(lesson / "practice.yaml")
    assert any("complexity" in p for p in problems)


def test_gate_rejects_a_missing_reading_line(lesson):
    path = lesson / "ex01_thing.py"
    path.write_text(SOLUTION.replace("Reading of the exercise: do it.", "Just does it."),
                    encoding="utf-8")
    problems = audit_practice.audit_lesson(lesson / "practice.yaml")
    assert any("Reading of the exercise" in p for p in problems)


def test_gate_rejects_a_missing_practice_impl(lesson):
    path = lesson / "ex01_thing.py"
    path.write_text(SOLUTION.replace("PRACTICE_IMPL = ", "NOT_THE_SHIM = "), encoding="utf-8")
    problems = audit_practice.audit_lesson(lesson / "practice.yaml")
    assert any("PRACTICE_IMPL" in p for p in problems)


def test_gate_rejects_a_missing_readme(lesson):
    (lesson / "README.md").unlink()
    assert any("README" in p for p in audit_practice.audit_lesson(lesson / "practice.yaml"))


def test_gate_rejects_an_unlabelled_fixture(lesson, tmp_path):
    (lesson / "fixtures").mkdir()
    (lesson / "fixtures" / "data.json").write_text('{"rows": []}', encoding="utf-8")
    text = MANIFEST.replace("    verifies: it does the thing\n",
                            "    verifies: it does the thing\n    fixtures:\n"
                            "      - fixtures/data.json\n")
    (lesson / "practice.yaml").write_text(text, encoding="utf-8")
    problems = audit_practice.audit_lesson(lesson / "practice.yaml")
    assert any("unlabelled" in p for p in problems)


def test_check_deps_rejects_an_undeclared_import(lesson):
    path = lesson / "ex01_thing.py"
    path.write_text("import numpy\n" + SOLUTION, encoding="utf-8")
    allowed = check_deps.STDLIB | check_deps.LOCAL | check_deps.GROUP_MODULES["none"]
    assert "numpy" in check_deps.imports(path) - allowed


def test_check_deps_allows_a_guarded_import_in_the_harness():
    """parity.try_numpy imports numpy lazily — that must stay legal."""
    path = ROOT / "harness" / "parity.py"
    assert "numpy" in check_deps.imports(path)
    assert "numpy" not in check_deps.toplevel_imports(path)


def test_explain_gate_requires_a_resolvable_citation():
    """DESIGN §6: a prose item's citation must name a real lesson section."""
    from harness import manifest as m

    exercise = m.Exercise(index=1, slug="x", kind="explain", tier="T0",
                          en="Explain something", zh="解釋", cites="The Concept")
    readme = ROOT / "tests" / "_tmp_readme.md"
    readme.write_text("answer, drawing on The Concept\n", encoding="utf-8")
    try:
        assert audit_practice.audit_explain(exercise, readme, {"The Concept"}) == []
        # cited section does not exist upstream
        problems = audit_practice.audit_explain(exercise, readme, {"Something Else"})
        assert any("not a heading" in p for p in problems)
        # answer missing from the README
        readme.write_text("no answer here\n", encoding="utf-8")
        problems = audit_practice.audit_explain(exercise, readme, {"The Concept"})
        assert any("not answered in README" in p for p in problems)
    finally:
        readme.unlink()


def test_coverage_check_flags_spec_drift(lesson, monkeypatch):
    pack = manifest.load_practice(lesson / "practice.yaml")
    upstream = "Do something completely different"
    assert coverage.spec_hash(pack.exercises[0].en) != coverage.spec_hash(upstream)
