"""Manifest schema: strict, total, and loud on anything unknown."""

from __future__ import annotations

import pytest

from harness import manifest

GOOD = """\
lesson: demo-lesson
phase: 00-phase
source: phases/00-phase/demo-lesson/docs/en.md
exercises:
  - index: 1
    slug: thing
    kind: code
    tier: T0
    en: |
      Do the thing
    zh: |
      做那件事
    verifies: it does the thing
"""


def write(tmp_path, text):
    path = tmp_path / "practice.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_a_good_manifest(tmp_path):
    pack = manifest.load_practice(write(tmp_path, GOOD))
    assert pack.lesson == "demo-lesson"
    ex = pack.exercises[0]
    assert ex.filename == "ex01_thing.py"
    assert ex.deps_group == "none"


def test_unknown_key_is_rejected(tmp_path):
    with pytest.raises(manifest.ManifestError, match="unknown key"):
        manifest.load_practice(write(tmp_path, GOOD + "    surprise: 1\n"))


def test_code_kind_requires_a_verifies_threshold(tmp_path):
    text = GOOD.replace("    verifies: it does the thing\n", "")
    with pytest.raises(manifest.ManifestError, match="requires a 'verifies'"):
        manifest.load_practice(write(tmp_path, text))


def test_explain_kind_requires_a_citation(tmp_path):
    text = GOOD.replace("kind: code", "kind: explain").replace(
        "    verifies: it does the thing\n", "")
    with pytest.raises(manifest.ManifestError, match="requires a 'cites'"):
        manifest.load_practice(write(tmp_path, text))


def test_explain_kind_needs_no_verifies(tmp_path):
    """The DESIGN §6 fix: the gate is kind-aware, so prose is not rejected."""
    text = GOOD.replace("kind: code", "kind: explain").replace(
        "    verifies: it does the thing", "    cites: The Concept")
    pack = manifest.load_practice(write(tmp_path, text))
    assert pack.exercises[0].kind == "explain"
    assert pack.code_exercises == ()


@pytest.mark.parametrize("bad,match", [
    ("kind: code", "kind: sculpture"),
    ("tier: T0", "tier: T9"),
])
def test_enumerations_are_closed(tmp_path, bad, match):
    with pytest.raises(manifest.ManifestError):
        manifest.load_practice(write(tmp_path, GOOD.replace(bad, match)))


def test_indices_must_start_at_one_and_ascend(tmp_path):
    text = GOOD.replace("index: 1", "index: 2")
    with pytest.raises(manifest.ManifestError, match="must start at 1"):
        manifest.load_practice(write(tmp_path, text))


def test_empty_exercise_text_is_rejected(tmp_path):
    text = GOOD.replace("      做那件事\n", "      \n")
    with pytest.raises(manifest.ManifestError, match="zh text is empty"):
        manifest.load_practice(write(tmp_path, text))
