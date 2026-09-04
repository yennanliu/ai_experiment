"""tiers, parity, coverage, cassette and the practice shim."""

from __future__ import annotations

import json

import pytest

from harness import cassette, coverage, explain, parity, practice, tiers

PHASE, LESSON = "01-math-foundations", "01-linear-algebra-intuition"


def test_t0_always_runs_and_missing_capability_carries_a_remedy():
    assert tiers.probe("T0").ok
    gpu = tiers.probe("T3")
    if not gpu.ok:
        assert gpu.remedy, "a skip without a remedy is the stack trace we banned"


def test_tier_ceiling_is_a_ceiling_not_a_match(monkeypatch):
    monkeypatch.setenv("DEMO_TIER", "T1")
    assert tiers.selected("T0") and tiers.selected("T1")
    assert not tiers.selected("T2")


def test_reference_is_found_by_ancestor_search():
    root = parity.find_reference_root()
    assert (root / "phases" / PHASE / LESSON / "docs" / "en.md").is_file()


def test_load_reference_swallows_import_time_stdout(capsys):
    """Some lesson modules run their demos at import; that must not reach the runner."""
    parity.load_reference("02-ml-fundamentals", "03-logistic-regression",
                          "logistic_regression")
    captured = capsys.readouterr()
    assert captured.out == "", f"import leaked {len(captured.out)} chars of demo output"


def test_load_reference_imports_rather_than_copies():
    module = parity.load_reference(PHASE, LESSON, "vectors")
    assert module.__file__.endswith("code/vectors.py")
    assert str(parity.find_reference_root()) in module.__file__


def test_assert_close_reports_the_measured_deviation():
    deviation = parity.assert_close([1.0, 2.0], [1.0, 2.0 + 1e-13], atol=1e-9)
    assert deviation.ok and 0 < deviation.worst < 1e-9
    with pytest.raises(AssertionError, match="worst"):
        parity.assert_close([1.0], [1.5], atol=1e-9)


def test_exercise_block_folds_wrapped_lines():
    items = coverage.exercise_block(
        "## Exercises\n\n1. first item\n   continued here\n2. second\n\n## Key Terms\n")
    assert items == ["first item continued here", "second"]


def test_exercise_block_handles_the_chinese_heading():
    assert coverage.exercise_block("## 練習\n\n1. 做這個\n") == ["做這個"]


def test_spec_hash_ignores_whitespace_but_not_words():
    assert coverage.spec_hash("a  b\nc") == coverage.spec_hash("a b c")
    assert coverage.spec_hash("a b") != coverage.spec_hash("a c")


def test_lesson_url_is_derived_from_the_mirrored_path():
    assert explain.lesson_url(PHASE, LESSON).endswith(f"/phases/{PHASE}/{LESSON}/")


def test_cassette_redacts_at_the_write_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(cassette, "CASSETTE_DIR", tmp_path)
    tape = cassette.Cassette(name="t", model="m")
    tape.entries = {"k": {"request": {"key": "sk-abcdefghijklmno"}, "response": "hi"}}
    written = json.loads(tape.save().read_text())
    assert "sk-abcdefghijklmno" not in json.dumps(written)
    assert "<redacted>" in json.dumps(written)


def test_replay_miss_names_the_record_command(tmp_path, monkeypatch):
    monkeypatch.setattr(cassette, "CASSETTE_DIR", tmp_path)
    monkeypatch.setenv("DEMO_MODE", "replay")
    with pytest.raises(cassette.CassetteMiss, match="DEMO_MODE=live"):
        cassette.Cassette(name="empty").play({"a": 1}, lambda: "never called")


def test_grade_file_reports_a_missing_practice_impl(tmp_path):
    path = tmp_path / "ex99_nothing.py"
    path.write_text("x = 1\n")
    result = practice.grade_file(path)
    assert result.status == "error" and "PRACTICE_IMPL" in result.detail


def test_grade_file_flags_a_solution_that_checks_nothing(tmp_path):
    path = tmp_path / "ex98_empty.py"
    path.write_text("PRACTICE_IMPL = {'solve': lambda: 1, 'verify': lambda r: []}\n")
    assert practice.grade_file(path).status == "error"


def test_a_failing_check_fails_the_grade(tmp_path):
    path = tmp_path / "ex97_bad.py"
    path.write_text(
        "from harness import practice\n"
        "PRACTICE_IMPL = {'solve': lambda: 1,\n"
        "                 'verify': lambda r: [practice.Check('no', False, 'nope')]}\n")
    result = practice.grade_file(path)
    assert result.status == "fail" and not result.ok
