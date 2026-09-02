"""The two rules that make 511 demos survivable: clean skips and real parity."""

from pathlib import Path

import pytest

from harness import tiers
from harness.manifest import parse
from harness.parity import ParityError, assert_close, compare, flatten, load_reference

HERE = Path("demos/x/demo.yaml")


def demo(**overrides):
    body = {
        "lesson": "phases/01-math-foundations/02-vectors-matrices-operations",
        "title": "A demo", "tier": "T0", "runtime_seconds": 10,
        "deps_group": "math", "proves": "something",
    }
    body.update(overrides)
    lines = [f"{k}: {v}" for k, v in body.items()]
    return parse("\n".join(lines), HERE)


# --------------------------------------------------------------------------
# tiers
# --------------------------------------------------------------------------


def test_a_t0_demo_runs_anywhere():
    assert tiers.check(demo()) is None


def test_a_t3_demo_skips_with_a_remedy_not_a_crash(monkeypatch):
    """DESIGN D2: a T3 demo on a Mac must explain itself, never stack-trace."""
    monkeypatch.setattr(tiers, "has_cuda", lambda: False)
    skip = tiers.check(demo(tier="T3", skip_reason="rent an A100, ~$1.50/hr"))
    assert skip is not None
    assert "CUDA" in skip.reason
    assert "$1.50/hr" in skip.remedy
    assert "SKIP" in skip.render()


def test_a_t1_demo_skips_when_torch_is_absent(monkeypatch):
    monkeypatch.setattr(tiers, "has_module", lambda name: False)
    skip = tiers.check(demo(tier="T1", deps_group="llm"))
    assert skip and "uv sync --extra llm" in skip.remedy


def test_a_t2_demo_with_no_tape_skips_instead_of_billing_a_live_call(tmp_path):
    manifest = tmp_path / "demo.yaml"
    manifest.write_text("")
    spec = demo(tier="T2", cassette="tape.json")
    object.__setattr__(spec, "path", tmp_path)
    skip = tiers.check(spec, run_mode=tiers.REPLAY)
    assert skip and "cassettes/tape.json" in skip.reason


def test_live_mode_without_a_key_skips_rather_than_erroring(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    spec = demo(tier="T2", cassette="tape.json")
    object.__setattr__(spec, "path", tmp_path)
    object.__setattr__(spec, "needs_env", ["ANTHROPIC_API_KEY"])
    (tmp_path / "cassettes").mkdir()
    (tmp_path / "cassettes" / "tape.json").write_text("{}")
    skip = tiers.check(spec, run_mode=tiers.LIVE)
    assert skip and "ANTHROPIC_API_KEY" in skip.reason


def test_demo_mode_defaults_to_replay(monkeypatch):
    monkeypatch.delenv("DEMO_MODE", raising=False)
    assert tiers.mode() == tiers.REPLAY


def test_an_unknown_demo_mode_is_refused(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "yolo")
    with pytest.raises(SystemExit):
        tiers.mode()


# --------------------------------------------------------------------------
# parity
# --------------------------------------------------------------------------


def test_flatten_handles_the_shapes_the_lessons_actually_produce():
    class LessonMatrix:            # Phase 07 style: flat row-major .data
        data = [1.0, 2.0, 3.0, 4.0]

    assert flatten(3) == [3.0]
    assert flatten([[1, 2], [3, 4]]) == [1.0, 2.0, 3.0, 4.0]
    assert flatten(LessonMatrix()) == [1.0, 2.0, 3.0, 4.0]


def test_flatten_refuses_what_it_cannot_turn_into_numbers():
    with pytest.raises(ParityError, match="cannot flatten"):
        flatten(object())


def test_assert_close_reports_the_worst_element_when_it_fails():
    with pytest.raises(ParityError) as excinfo:
        assert_close([1.0, 2.0, 3.0], [1.0, 2.0, 9.0], label="x", atol=1e-9)
    assert "worst element [2]" in str(excinfo.value)
    assert "9.0" in str(excinfo.value)


def test_a_shape_mismatch_fails_instead_of_comparing_a_prefix():
    with pytest.raises(ParityError, match="shape mismatch"):
        assert_close([1.0], [1.0, 2.0], label="x")


def test_comparing_nothing_is_an_error_not_a_pass():
    """An empty comparison would otherwise be a green test that proves nothing."""
    with pytest.raises(ParityError, match="nothing to compare"):
        assert_close([], [], label="x")


def test_rtol_defaults_to_zero_so_atol_means_what_it_says():
    """A 1e-4 relative default would silently swallow a 1e-12 claim."""
    assert not compare([1.0], [1.0001], label="x", atol=1e-12).passed
    assert compare([1.0], [1.0001], label="x", atol=1e-12, rtol=1e-3).passed


def test_load_reference_imports_the_lessons_own_code():
    module = load_reference(
        "phases/01-math-foundations/02-vectors-matrices-operations/code/matrices.py"
    )
    assert module.Matrix([[1, 2], [3, 4]]).determinant() == -2


def test_load_reference_names_the_file_it_could_not_find():
    with pytest.raises(ParityError, match="reference module not found"):
        load_reference("phases/00-setup-and-tooling/99-nope/code/main.py")
