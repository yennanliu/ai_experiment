"""Exercise 3 — allowlisting makes it valid, and the report still names it.

    Add one runtime extension to the allowlist. Write a test proving the same
    file is still distinguishable from a portable-only skill.

Reading of the exercise: "the same file" is the constraint that makes this
interesting -- one byte sequence, two host policies, and the question is
whether anything survives the policy that lets a consumer tell the two skills
apart. It does: `validity` moves and `runtime_extensions` does not, so the
report separates "this host permits it" from "this file needs it", which are
the two different questions a portability check asks.

**ANSWER: allowlisting flips `valid` and leaves `runtime_extensions`
unchanged.** The same text reports `valid=False` with
`unsupported-runtime-field` under an empty policy and `valid=True` under
`("x-claude-model",)`, while `runtime_extensions` is `("x-claude-model",)` in
both. A portable-only skill reports `()` under either policy -- so the two
files are distinguishable by a field the policy cannot change.

**FINDING: validity is a fact about the pair, and the extension list is a
fact about the file.** Four combinations of two files and two policies give
**3** distinct validities and **2** distinct extension lists: the policy
moves one and not the other. A consumer that wants "runs here" reads `valid`;
one that wants "runs anywhere" reads `runtime_extensions` and ignores the
host it is on.

**FINDING: the allowlist is per-field, so permitting one does not permit the
next.** An issue is raised per unpermitted field: a file using both
`x-claude-model` and `x-openai-tier` reports **1** issue under a one-field
policy and **2** under an empty one, while its `runtime_extensions` is the
same pair either way. Each extension is admitted by name.

**FINDING: `core_fields` is the complement and is computed the same way.**
`set(fields) - CORE_FIELDS` produces the extensions and `set(fields) &
CORE_FIELDS` the core, from one parse -- so a field is portable exactly when
it is in a set of **6** names hard-coded in the module. There is no version
on that set, which is what would let a skill declare which contract it was
written against.

Structure: `check` validates one text under one policy, so the file and the
policy vary independently.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "22-skills-and-agent-sdks"
NAME = "report-writer"
EXTENSION = "x-claude-model"
OTHER = "x-openai-tier"


def skill_text(*extensions):
    lines = ["---", f"name: {NAME}", "description: Draft a report when asked."]
    lines += [f"{field}: value" for field in extensions]
    return "\n".join(lines + ["---", "", "Draft the report.\n"])


def check(ref, text, policy=()):
    return ref.validate_skill_text(text, NAME, allowed_runtime_extensions=policy)


def codes(report):
    return sorted(issue.code for issue in report.issues)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    extended_text, portable_text = skill_text(EXTENSION), skill_text()
    grid = {
        ("extended", "closed"): check(ref, extended_text),
        ("extended", "open"): check(ref, extended_text, (EXTENSION,)),
        ("portable", "closed"): check(ref, portable_text),
        ("portable", "open"): check(ref, portable_text, (EXTENSION,)),
    }
    both = check(ref, skill_text(EXTENSION, OTHER), (EXTENSION,))
    neither = check(ref, skill_text(EXTENSION, OTHER))
    return {
        "validity": {f"{f}/{p}": r.valid for (f, p), r in grid.items()},
        "extensions": {f"{f}/{p}": list(r.runtime_extensions) for (f, p), r in grid.items()},
        "closed_codes": codes(grid[("extended", "closed")]),
        "open_codes": codes(grid[("extended", "open")]),
        "distinct_validities": len({r.valid for r in grid.values()}),
        "distinct_extensions": len({r.runtime_extensions for r in grid.values()}),
        "both_issues": len(both.issues), "both_extensions": list(both.runtime_extensions),
        "neither_issues": len(neither.issues),
        "core": sorted(grid[("extended", "open")].core_fields),
        "core_set": sorted(ref.CORE_FIELDS), "core_size": len(ref.CORE_FIELDS),
        "version_field": [f for f in ref.CORE_FIELDS if "version" in f],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: allowlisting flips valid and leaves runtime_extensions unchanged",
            all([result["validity"]["extended/closed"] is False,
                 result["validity"]["extended/open"] is True,
                 result["extensions"]["extended/closed"] == [EXTENSION],
                 result["extensions"]["extended/open"] == [EXTENSION],
                 result["extensions"]["portable/open"] == [],
                 result["closed_codes"] == ["unsupported-runtime-field"],
                 result["open_codes"] == []]),
            f"the same text reports {result['closed_codes']} under an empty policy and "
            f"{result['open_codes']} under ({EXTENSION!r},), while runtime_extensions stays "
            f"{result['extensions']['extended/open']} in both. A portable-only skill reports "
            f"{result['extensions']['portable/open']} under either policy",
        ),
        practice.Check(
            "FINDING: validity is a fact about the pair and the extension list about the file",
            all([result["distinct_validities"] == 2,
                 result["distinct_extensions"] == 2,
                 result["validity"]["portable/closed"] is True]),
            f"four combinations of two files and two policies give "
            f"{result['distinct_validities']} validities {result['validity']} and "
            f"{result['distinct_extensions']} extension lists. A consumer asking 'runs here' "
            "reads valid; one asking 'runs anywhere' reads runtime_extensions and ignores "
            "the host it is on",
        ),
        practice.Check(
            "FINDING: the allowlist is per-field, so permitting one does not permit the next",
            all([result["both_issues"] == 1, result["neither_issues"] == 2,
                 result["both_extensions"] == sorted([EXTENSION, OTHER])]),
            f"a file using both extensions under a one-field policy reports "
            f"{result['both_issues']} issue and under an empty policy "
            f"{result['neither_issues']} -- one per unpermitted field -- while its "
            f"runtime_extensions is {result['both_extensions']} either way. Each extension "
            "is admitted by name",
        ),
        practice.Check(
            "FINDING: core_fields is the complement, computed from one hard-coded set",
            all([result["core"] == ["description", "name"],
                 result["core_size"] == 6, result["version_field"] == []]),
            f"the report's core fields are {result['core']} and the extensions their "
            f"complement, both from the {result['core_size']}-name set "
            f"{result['core_set']}. There is {len(result['version_field'])} version on that "
            "set, which is what would let a skill declare which contract it was written "
            "against",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
