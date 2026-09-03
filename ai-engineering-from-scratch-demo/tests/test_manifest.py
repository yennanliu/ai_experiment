"""The manifest parser is load-bearing: CI, deps and coverage all read it."""

from pathlib import Path

import pytest

from harness.manifest import ManifestError, load_yaml, parse

MINIMAL = """
lesson: phases/01-math-foundations/02-vectors-matrices-operations
title: A demo
tier: T0
runtime_seconds: 10
deps_group: math
proves: >
  It proves something.
"""
HERE = Path("demos/x/demo.yaml")


def without(manifest: str, field: str) -> str:
    """Drop `field:` and any indented continuation (block scalars, lists)."""
    kept, skipping = [], False
    for line in manifest.splitlines():
        if line.startswith(f"{field}:"):
            skipping = True
            continue
        if skipping and (not line.strip() or line[:1].isspace()):
            continue
        skipping = False
        kept.append(line)
    return "\n".join(kept)


def test_the_supported_yaml_subset_round_trips():
    parsed = load_yaml(
        "a: 1\n"
        "b: 2.5\n"
        "c: [x, y]\n"
        "d:\n  - p\n  - q\n"
        "e: >\n  folded\n  text\n"
        "f: |\n  kept\n  lines\n"
        "g: null\n"
        "h: true\n"
        "i: 'quoted: colon'\n"
        "j: []\n"
    )
    assert parsed == {
        "a": 1, "b": 2.5, "c": ["x", "y"], "d": ["p", "q"],
        "e": "folded text", "f": "kept\nlines", "g": None, "h": True,
        "i": "quoted: colon", "j": [],
    }


def test_comments_and_blank_lines_are_ignored():
    assert load_yaml("# lead\n\na: 1  # trailing\n") == {"a": 1}


def test_a_hash_inside_a_quoted_value_is_not_a_comment():
    assert load_yaml("a: 'issue #1'\n") == {"a": "issue #1"}


@pytest.mark.parametrize("bad", ["  indented: 1", "no colon here"])
def test_unsupported_yaml_raises_rather_than_being_misread(bad):
    """Silently misparsing a manifest is worse than refusing it."""
    with pytest.raises(ManifestError):
        load_yaml(bad)


def test_a_valid_manifest_parses():
    demo = parse(MINIMAL, HERE)
    assert demo.tier == "T0"
    assert demo.phase == "01-math-foundations"
    assert demo.phase_number == "01"
    assert demo.entrypoint == "run.py"
    assert demo.needs_env == []
    assert not demo.has_parity


@pytest.mark.parametrize("field", ["lesson", "title", "tier", "runtime_seconds",
                                   "deps_group", "proves"])
def test_every_required_field_is_actually_required(field):
    with pytest.raises(ManifestError, match="missing required field"):
        parse(without(MINIMAL, field), HERE)


@pytest.mark.parametrize(("old", "new", "message"), [
    ("tier: T0", "tier: T9", "tier must be one of"),
    ("deps_group: math", "deps_group: nope", "deps_group must be one of"),
    ("lesson: phases/01", "lesson: elsewhere/01", "must be a `phases/...` path"),
])
def test_an_out_of_range_value_is_rejected(old, new, message):
    with pytest.raises(ManifestError, match=message):
        parse(MINIMAL.replace(old, new), HERE)


def test_an_unknown_field_is_rejected():
    """A typo'd field name would otherwise be silently ignored forever."""
    with pytest.raises(ManifestError, match="unknown field"):
        parse(MINIMAL + "tierr: T1\n", HERE)


def test_a_t2_demo_without_a_cassette_is_rejected():
    """Without a tape, replay mode would fall through to a billed live call."""
    with pytest.raises(ManifestError, match="requires a `cassette`"):
        parse(MINIMAL.replace("tier: T0", "tier: T2"), HERE)


def test_a_t3_demo_without_a_skip_reason_is_rejected():
    """A T3 demo that only crashes on a laptop is not a demo."""
    with pytest.raises(ManifestError, match="requires a `skip_reason`"):
        parse(MINIMAL.replace("tier: T0", "tier: T3"), HERE)


def test_runtime_seconds_must_be_a_positive_integer():
    with pytest.raises(ManifestError, match="positive integer"):
        parse(MINIMAL.replace("runtime_seconds: 10", "runtime_seconds: 0"), HERE)
