"""Every exercise in this lesson, graded by its own PRACTICE_IMPL (D13).

The test does not restate any threshold: it runs the same solve/verify pair that
`uv run demo practice run` and `python exNN_*.py` run, so the three callers
cannot drift apart.
"""

from __future__ import annotations

import pathlib

import pytest

from harness import manifest, practice

HERE = pathlib.Path(__file__).resolve().parent.parent
PACK = manifest.load_practice(HERE / "practice.yaml")


@pytest.mark.parametrize("exercise", PACK.code_exercises, ids=lambda e: e.stem)
def test_solution_passes_its_own_checks(exercise):
    result = practice.grade_file(HERE / exercise.filename, exercise.stem)
    assert result.status == "pass", practice.report(result)
    assert result.checks, "a solution with no checks proves nothing"


@pytest.mark.parametrize("exercise", PACK.exercises, ids=lambda e: e.stem)
def test_exercise_text_matches_upstream(exercise):
    """D12/D15: the stored spec is the upstream spec, or the build is stale."""
    from harness import coverage, parity
    upstream = coverage.exercise_block(parity.doc_text(PACK.phase, PACK.lesson, "en"))
    assert coverage.spec_hash(exercise.en) == coverage.spec_hash(upstream[exercise.index - 1])


def test_every_upstream_exercise_is_covered():
    from harness import coverage, parity
    upstream = coverage.exercise_block(parity.doc_text(PACK.phase, PACK.lesson, "en"))
    assert len(PACK.exercises) == len(upstream) == 6
