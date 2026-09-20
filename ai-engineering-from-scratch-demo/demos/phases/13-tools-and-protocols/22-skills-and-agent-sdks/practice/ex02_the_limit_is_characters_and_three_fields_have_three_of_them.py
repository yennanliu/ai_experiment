"""Exercise 2 — the limit is characters, and three fields have three of them.

    Add boundary tests proving that a 500-character `compatibility` value
    passes and a 501-character value fails as a specification error.

Reading of the exercise: two assertions fix one boundary, so the sweep is
widened to the states either side of "present and long" -- absent, present
and empty -- because those are different codes and a test that only checks
length would call them the same failure. The unit is then checked, since
`len()` on a Python string counts characters and a 500-character value can be
more than 500 bytes on the wire.

**ANSWER: 500 passes, 501 fails with `compatibility-too-long`.** The check is
`len(compatibility) > 500`, so the boundary is inclusive and the failing
report carries exactly **1** issue with that code. A valid skill at 500
characters reports `valid=True`.

**FINDING: absent, empty and over-length are three outcomes, not two.**
Omitting `compatibility` is valid -- the whole block is guarded by
`"compatibility" in fields` -- while an empty value fails as
`compatibility-empty` and an over-length one as `compatibility-too-long`. A
test that only measures length would treat the optional field as required.

**FINDING: the three bounded fields have three different limits and state
none of them.** `name` is **64**, `description` is **1024**, `compatibility`
is **500** -- and the frontmatter carries no schema, so a skill author
discovers each by exceeding it. The boundary at each is inclusive in the same
way, which is the only thing they share.

**FINDING: the limit counts characters, so the byte length is unbounded.**
A 500-character value of `"→"` is **1500** bytes and passes, while a
501-character ASCII value is **501** bytes and fails. Anything downstream
budgeting bytes -- a header, a database column, a context window -- is
reading a number that does not bound it.

Structure: `report` builds one SKILL.md with a given frontmatter and
validates it, so every row differs only in the field under test.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "22-skills-and-agent-sdks"
NAME = "report-writer"
BODY = "Use this skill to draft a report.\n"


def skill_text(**fields):
    lines = ["---", f"name: {NAME}", "description: Draft a report when asked."]
    lines += [f"{key}: {value}" for key, value in fields.items()]
    return "\n".join(lines + ["---", "", BODY])


def report(ref, **fields):
    return ref.validate_skill_text(skill_text(**fields), NAME)


def codes(result):
    return sorted(issue.code for issue in result.issues)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    at_limit = report(ref, compatibility="c" * 500)
    over = report(ref, compatibility="c" * 501)
    absent = report(ref)
    empty = report(ref, compatibility='""')
    multibyte = report(ref, compatibility="→" * 500)
    long_name = ref.validate_skill_text(
        "\n".join(["---", f"name: {'n' * 65}", "description: d", "---", "", BODY]), "n" * 65)
    long_description = ref.validate_skill_text(
        "\n".join(["---", f"name: {NAME}", f"description: {'d' * 1025}", "---", "", BODY]),
        NAME)
    return {
        "at_limit_valid": at_limit.valid, "at_limit_codes": codes(at_limit),
        "over_valid": over.valid, "over_codes": codes(over),
        "absent_valid": absent.valid, "absent_codes": codes(absent),
        "empty_codes": codes(empty),
        "name_codes": codes(long_name), "description_codes": codes(long_description),
        "limits": {"name": 64, "description": 1024, "compatibility": 500},
        "multibyte_valid": multibyte.valid,
        "multibyte_chars": 500, "multibyte_bytes": len(("→" * 500).encode()),
        "ascii_over_bytes": len(("c" * 501).encode()),
        "core_fields": sorted(ref.CORE_FIELDS),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 500 passes and 501 fails with compatibility-too-long",
            all([result["at_limit_valid"], result["at_limit_codes"] == [],
                 not result["over_valid"],
                 result["over_codes"] == ["compatibility-too-long"]]),
            f"a 500-character value reports valid={result['at_limit_valid']} with "
            f"{result['at_limit_codes']} issues, and 501 reports "
            f"valid={result['over_valid']} with {result['over_codes']} -- exactly one issue. "
            "The check is `> 500`, so the boundary is inclusive",
        ),
        practice.Check(
            "FINDING: absent, empty and over-length are three outcomes, not two",
            all([result["absent_valid"], result["absent_codes"] == [],
                 result["empty_codes"] == ["compatibility-empty"],
                 result["over_codes"] == ["compatibility-too-long"]]),
            f"omitting the field is valid ({result['absent_valid']}), an empty value gives "
            f"{result['empty_codes']} and an over-length one {result['over_codes']} -- the "
            "whole block is guarded by `'compatibility' in fields`. A test measuring only "
            "length would treat the optional field as required",
        ),
        practice.Check(
            "FINDING: the three bounded fields have three limits and state none of them",
            all([result["name_codes"] == ["name-too-long"],
                 result["description_codes"] == ["description-too-long"],
                 result["limits"] == {"name": 64, "description": 1024,
                                      "compatibility": 500}]),
            f"the limits are {result['limits']} -- three numbers for three fields -- and the "
            f"frontmatter carries no schema, so an author discovers each by exceeding it: "
            f"{result['name_codes']}, {result['description_codes']}, "
            f"{result['over_codes']}. The inclusive boundary is all they share",
        ),
        practice.Check(
            "FINDING: the limit counts characters, so the byte length is unbounded",
            all([result["multibyte_valid"],
                 result["multibyte_bytes"] == 1500,
                 result["multibyte_bytes"] > result["ascii_over_bytes"]]),
            f"a {result['multibyte_chars']}-character value of a 3-byte codepoint is "
            f"{result['multibyte_bytes']} bytes and passes, while a 501-character ASCII "
            f"value is {result['ascii_over_bytes']} bytes and fails. Anything downstream "
            "budgeting bytes is reading a number that does not bound it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
