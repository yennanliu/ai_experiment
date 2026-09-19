"""Exercise 1 — twenty findings to zero, and duplicates are counted once per copy.

    Take the `BAD_REGISTRY` in `code/main.py` and rewrite each tool to pass the
    linter. Measure description length and count rule violations before and
    after.

Reading of the exercise: each of the three tools is rewritten rather than
deleted, keeping whatever the original was plausibly for, and the before/after
counts are taken from the lesson's own `lint_registry`. Counting violations is
the part that repays attention -- one of the twenty is a rule that fires once per
offending tool rather than once per offence, so the count is not a count of
problems.

**ANSWER: 20 findings to 0, and the descriptions go from 19/16/109 to 88/112/85
characters.** Six `block`, eleven `warn` and three `nit` become none. Two of the
three descriptions were under the 40-character floor; the third was long enough
and carried two injection patterns.

**FINDING: the duplicate-name rule reports once per copy, not once per
collision.** `lint_registry` loops over every name and appends a finding whenever
`names.count(n) > 1`, so two tools sharing a name produce **2** findings and
three produce **3**. A registry with one naming mistake repeated four times reads
as four blocking problems.

**FINDING: passing the linter is not the same as being right.** A tool named
`delete_note` described as "Use when the user asks for the weather. Do not use
for invoices." lints at **0 findings**. Nothing compares the description to the
name, so the one property that makes a description useful to a model -- that it
describes *this* tool -- is the one property not checked.

**FINDING: and one of the twenty is a false positive.** `get_weather_in_tokyo`
is correctly flagged for embedding an argument, but the rule is
`_(in|for|at|by)_\\w+$`, which also fires on **`search_notes_for_text`** -- where
`text` names the field being searched, not a baked-in value. The rewrite has to
avoid a legitimate name shape to score zero.

Structure: `FIXED` is the rewritten registry, `census` counts findings by
severity, and `lengths` reports description sizes on either side.
"""

from __future__ import annotations

from collections import Counter

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "05-tool-schema-design"
FIXED = [
    {
        "name": "run_workflow",
        "description": ("Use when the user names a saved workflow to run. "
                        "Do not use for ad-hoc one-off commands."),
        "input_schema": {
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string",
                                "description": "Saved workflow id, e.g. wf-41."},
                "dry_run": {"type": "boolean"},
            },
            "required": ["workflow_id"],
        },
    },
    {
        "name": "get_weather",
        "description": ("Use when the user asks about current conditions in a named city. "
                        "Do not use for forecasts or historical weather."),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Bengaluru."},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"],
                          "description": "Temperature unit."},
            },
            "required": ["city"],
        },
    },
    {
        "name": "notes_list",
        "description": ("Use when the user wants their notes enumerated. "
                        "Do not use for searching note bodies."),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer"},
                "cursor": {"type": "string", "description": "Opaque page cursor."},
            },
            "required": [],
        },
    },
]
MISMATCHED = {
    "name": "delete_note",
    "description": ("Use when the user asks for the weather in a named city. "
                    "Do not use for invoices."),
    "input_schema": {"type": "object", "properties": {}, "required": []},
}


def census(ref, registry):
    return dict(Counter(finding.severity for finding in ref.lint_registry(registry)))


def total(ref, registry):
    return len(ref.lint_registry(registry))


def lengths(registry):
    return [len(tool["description"]) for tool in registry]


def duplicated(ref, copies):
    """Findings a registry of `copies` identically named tools produces."""
    one = dict(FIXED[1])
    registry = [dict(one) for _ in range(copies)]
    return sum(1 for finding in ref.lint_registry(registry)
               if "duplicate" in finding.message)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {
        "before": census(ref, ref.BAD_REGISTRY), "before_total": total(ref, ref.BAD_REGISTRY),
        "after": census(ref, FIXED), "after_total": total(ref, FIXED),
        "before_lengths": lengths(ref.BAD_REGISTRY), "after_lengths": lengths(FIXED),
        "under_floor": sum(length < 40 for length in lengths(ref.BAD_REGISTRY)),
        "tools": len(FIXED),
        "duplicates": {copies: duplicated(ref, copies) for copies in (2, 3, 4)},
        "mismatched": total(ref, [MISMATCHED]),
        "embedded_arg": [str(f) for f in ref.lint_name("search_notes_for_text")],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 20 findings to 0, and two descriptions were under the floor",
            all([result["before"] == {"block": 6, "warn": 11, "nit": 3},
                 result["before_total"] == 20, result["after"] == {},
                 result["after_total"] == 0, result["tools"] == 3,
                 result["before_lengths"] == [19, 16, 109], result["under_floor"] == 2,
                 result["after_lengths"] == [88, 112, 85]]),
            f"the shipped registry lints at {result['before_total']} findings "
            f"({result['before']}) and the rewrite at {result['after_total']}. Descriptions "
            f"go from {result['before_lengths']} characters to {result['after_lengths']}; "
            f"{result['under_floor']} of the three were under the 40-character floor and the "
            "third was long enough while carrying two injection patterns",
        ),
        practice.Check(
            "FINDING: the duplicate-name rule reports once per copy, not once per collision",
            all([result["duplicates"] == {2: 2, 3: 3, 4: 4}]),
            f"lint_registry loops over every name and appends a finding whenever "
            f"names.count(n) > 1, so copies of one name produce {result['duplicates']} "
            "findings. A registry with a single naming mistake repeated four times reads as "
            "four blocking problems",
        ),
        practice.Check(
            "FINDING: passing the linter is not the same as being right",
            result["mismatched"] == 0,
            f"a tool named {MISMATCHED['name']} described as 'Use when the user asks for the "
            f"weather' lints at {result['mismatched']} findings. Nothing compares the "
            "description to the name, so the one property that makes a description useful to "
            "a model -- that it describes this tool -- is the one property not checked",
        ),
        practice.Check(
            "FINDING: and one of the twenty is a false positive",
            all([len(result["embedded_arg"]) == 1,
                 "argument appears embedded" in result["embedded_arg"][0]]),
            f"get_weather_in_tokyo is correctly flagged, but the rule is "
            f"_(in|for|at|by)_\\w+$, which also fires on search_notes_for_text: "
            f"{result['embedded_arg']}. There 'text' names the field being searched rather "
            "than a baked-in value, so the rewrite has to avoid a legitimate name shape to "
            "score zero",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
