"""The YAML subset parser — the item `PLAN §3` flagged as M0's main risk.

Tested in both directions: the subset parses correctly, and everything outside
it is a loud `YamlError` rather than a silent misparse.
"""

from __future__ import annotations

import pytest

from harness import yamlite


def test_scalars_and_types():
    got = yamlite.loads("a: 1\nb: 1.5\nc: true\nd: null\ne: text\nf: 'quoted: colon'\n")
    assert got == {"a": 1, "b": 1.5, "c": True, "d": None, "e": "text",
                   "f": "quoted: colon"}


def test_nested_mapping_and_sequence():
    got = yamlite.loads("top:\n  inner:\n    - 1\n    - 2\nother: x\n")
    assert got == {"top": {"inner": [1, 2]}, "other": "x"}


def test_sequence_of_mappings_with_block_scalar():
    got = yamlite.loads(
        "items:\n"
        "  - index: 1\n"
        "    text: |\n"
        "      line one\n"
        "      line two\n"
        "    tier: T0\n"
        "  - index: 2\n"
        "    text: folded\n"
    )
    assert got["items"][0]["text"] == "line one\nline two\n"
    assert got["items"][0]["tier"] == "T0"
    assert got["items"][1] == {"index": 2, "text": "folded"}


def test_folded_block_scalar_joins_lines():
    got = yamlite.loads("k: >\n  one\n  two\n")
    assert got["k"] == "one two"


def test_unicode_is_preserved_verbatim():
    got = yamlite.loads("zh: |\n  實作 `Vector.angle_between(other)`，回傳夾角\n")
    assert got["zh"].strip() == "實作 `Vector.angle_between(other)`，回傳夾角"


def test_comments_and_blank_lines_ignored():
    assert yamlite.loads("# lead\n\na: 1  # trailing\n") == {"a": 1}


@pytest.mark.parametrize("text,fragment", [
    ("a: [1, 2]\n", "flow collections"),
    ("a: &anchor x\n", "anchors"),
    ("a: 1\n b: 2\n", "not a multiple"),
    ("a: 1\na: 2\n", "duplicate key"),
    ("just text\n", "expected 'key: value'"),
    ("a: 'unterminated\n", "unterminated"),
    ("\ta: 1\n", "tab in indentation"),
])
def test_outside_the_subset_is_a_loud_error(text, fragment):
    with pytest.raises(yamlite.YamlError) as exc:
        yamlite.loads(text)
    assert fragment in str(exc.value)


def test_error_names_the_line():
    with pytest.raises(yamlite.YamlError) as exc:
        yamlite.loads("a: 1\nb: 2\nc: [3]\n")
    assert "line 3" in str(exc.value)
