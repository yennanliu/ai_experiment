"""The runner enforces the half of the contract a manifest cannot express."""

import sys
from pathlib import Path

import pytest

from harness import runner
from harness.manifest import parse

MANIFEST = """lesson: phases/01-math-foundations/02-vectors-matrices-operations
title: A demo
tier: T0
runtime_seconds: {budget}
deps_group: math
proves: it runs
"""


def make_demo(tmp_path, body, *, budget=10):
    (tmp_path / "run.py").write_text(body)
    demo = parse(MANIFEST.format(budget=budget), tmp_path / "demo.yaml")
    object.__setattr__(demo, "path", tmp_path)
    return demo


def test_a_demo_that_exits_zero_passes(tmp_path):
    result = runner.run_demo(make_demo(tmp_path, "print('ok')"), quiet=True)
    assert result.status == runner.PASS and result.ok


def test_a_demo_that_exits_non_zero_fails(tmp_path):
    demo = make_demo(tmp_path, "import sys; sys.exit(3)")
    result = runner.run_demo(demo, quiet=True)
    assert result.status == runner.FAIL
    assert "exit 3" in result.detail


def test_a_demo_that_blows_its_declared_budget_is_rejected(tmp_path):
    """DESIGN section 7: a runtime over the declared budget fails the gate."""
    demo = make_demo(tmp_path, "import time; time.sleep(1.2)", budget=1)
    result = runner.run_demo(demo, quiet=True)
    assert result.status == runner.OVER_BUDGET
    assert not result.ok
    assert "runtime_seconds: 1" in result.detail


def test_a_hanging_demo_is_killed_not_waited_on(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "GRACE_SECONDS", 1)
    demo = make_demo(tmp_path, "import time; time.sleep(30)", budget=1)
    result = runner.run_demo(demo, quiet=True)
    assert result.status == runner.OVER_BUDGET
    assert "killed after" in result.detail


def test_a_missing_entrypoint_fails_rather_than_silently_passing(tmp_path):
    demo = parse(MANIFEST.format(budget=10), tmp_path / "demo.yaml")
    object.__setattr__(demo, "path", tmp_path)
    assert runner.run_demo(demo, quiet=True).status == runner.FAIL


def test_a_demo_can_import_the_harness_without_an_install(tmp_path):
    """The runner puts the repo on PYTHONPATH so `from harness...` resolves."""
    demo = make_demo(tmp_path, "from harness import __version__; print(__version__)")
    assert runner.run_demo(demo, quiet=True).status == runner.PASS


def test_selecting_a_lesson_that_does_not_exist_says_how_to_start_it():
    with pytest.raises(SystemExit, match="demo scaffold"):
        runner.select("phases/99-nope/01-nothing")


def test_selection_filters_compose():
    assert runner.select(tier="T0", parity_only=True)
    assert runner.select(phase="07", tier="T0") == []
    assert all(d.phase_number == "11" for d in runner.select(phase="11"))
