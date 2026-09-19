"""Exercise 5 — the lesson states five parameter rules and lints two.

    Read Composio's tool-design field guide top to bottom. Identify one rule
    not covered in this lesson and add it to the linter. Then design a schema
    that uses the forbidden construct non-essentially and refactor it.

Reading of the exercise: the field guide is a network fetch and this runs
offline, so rather than paraphrase a document it cannot open, the rule is taken
from a source that *is* checkable -- the lesson's own "Parameter design"
section, which states five rules in five bullets. Auditing the linter against
them turns "find a rule it misses" into a measurement, and it misses three.

**ANSWER: the rule added is snake_case for parameter names.** `lint_name`
applies `^[a-z][a-z0-9_]*$` to the tool name and nothing applies it to the
fields. A schema with `noteId` and `MAX` lints at **0** findings today and at
**2** with the rule, which is a one-line extension of `lint_schema`.

**FINDING: the lesson states five parameter rules and the linter implements
two.** "Describe the field" is a `nit` and "no overly flexible types" is the
missing-`type` block. **Enum every closed set** is checked only for a key
literally named `action`; **typed IDs need a pattern** is not checked at all;
and **required vs optional** only checks that the key exists, never what is in
it. Three of five.

**FINDING: the two it skips are the two that catch hallucinated arguments.**
A `note_id` with no `pattern` accepts `"note-oops"` and a closed set with no
`enum` accepts any string -- both are the shapes a model invents when it is
guessing. The rules the linter does implement govern what the *author* wrote;
the rules it skips govern what the *model* may send.

**FINDING: and the new rule would fire on nothing the lesson ships.** All
**10** parameter names across `GOOD_REGISTRY` and `BAD_REGISTRY` are already
snake_case, so the extension scores 0 on both. A rule with no failing fixture is
a rule nobody has tested, which is the same gap Exercise 4 found in the CI gate.

Structure: `lint_parameter_names` is the extension, `audit` checks each stated
rule against the linter's behaviour on a probe schema, and `CAMEL` is the schema
the new rule exists to reject.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "05-tool-schema-design"
SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")
CAMEL = {"type": "object",
         "properties": {"noteId": {"type": "string", "description": "Note id."},
                        "MAX": {"type": "integer"}},
         "required": ["noteId"]}
UNPATTERNED_ID = {"type": "object",
                  "properties": {"note_id": {"type": "string", "description": "Note id."}},
                  "required": ["note_id"]}
OPEN_SET = {"type": "object",
            "properties": {"units": {"type": "string", "description": "Unit."}},
            "required": ["units"]}


def lint_parameter_names(ref, schema, tool_name):
    """The added rule: parameter names obey the same casing as tool names."""
    return [ref.Finding("warn", f"{tool_name}.{key}", "parameter name must be snake_case")
            for key in schema.get("properties", {}) if not SNAKE.match(key)]


def audit(ref):
    """Does the linter enforce each parameter rule the lesson's docs state?"""
    return {
        "enum every closed set": any("monolithic" in f.message
                                     for f in ref.lint_schema(OPEN_SET, "t")),
        "required vs optional": any("required" in f.message
                                    for f in ref.lint_schema(
                                        {"type": "object",
                                         "properties": {"a": {"type": "string",
                                                              "description": "x"}},
                                         "required": []}, "t")),
        "typed ids need a pattern": any("pattern" in f.message
                                        for f in ref.lint_schema(UNPATTERNED_ID, "t")),
        "no overly flexible types": any("no type" in f.message
                                        for f in ref.lint_schema(
                                            {"type": "object",
                                             "properties": {"a": {}}}, "t")),
        "describe the field": any("lacks description" in f.message
                                  for f in ref.lint_schema(
                                      {"type": "object",
                                       "properties": {"a": {"type": "string"}},
                                       "required": []}, "t")),
    }


def parameter_names(registry):
    return [key for tool in registry
            for key in tool["input_schema"].get("properties", {})]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    checked = audit(ref)
    shipped = parameter_names(ref.GOOD_REGISTRY) + parameter_names(ref.BAD_REGISTRY)
    return {
        "camel_before": len(ref.lint_schema(CAMEL, "t")),
        "camel_after": len(lint_parameter_names(ref, CAMEL, "t")),
        "camel_paths": [f.path for f in lint_parameter_names(ref, CAMEL, "t")],
        "tool_name_rule": bool(ref.lint_name("noteId")),
        "audit": checked,
        "implemented": sum(checked.values()), "stated": len(checked),
        "missing": sorted(rule for rule, ok in checked.items() if not ok),
        "shipped_params": len(shipped),
        "shipped_snake": sum(bool(SNAKE.match(key)) for key in shipped),
        "new_rule_on_shipped": sum(
            len(lint_parameter_names(ref, tool["input_schema"], tool["name"]))
            for tool in ref.GOOD_REGISTRY + ref.BAD_REGISTRY),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the rule added is snake_case for parameter names",
            all([result["camel_before"] == 0, result["camel_after"] == 2,
                 result["camel_paths"] == ["t.noteId", "t.MAX"],
                 result["tool_name_rule"]]),
            f"lint_name applies ^[a-z][a-z0-9_]*$ to the tool name -- it rejects 'noteId' "
            f"there -- and nothing applies it to the fields. A schema with noteId and MAX "
            f"lints at {result['camel_before']} findings today and "
            f"{result['camel_after']} with the rule: {result['camel_paths']}",
        ),
        practice.Check(
            "FINDING: the lesson states five parameter rules and the linter implements two",
            all([result["stated"] == 5, result["implemented"] == 2,
                 result["missing"] == ["enum every closed set", "required vs optional",
                                       "typed ids need a pattern"]]),
            f"of the {result['stated']} rules the Parameter design section states, "
            f"{result['implemented']} are enforced: {result['audit']}. Enum-every-closed-set "
            "is checked only for a key literally named action, typed-ids-need-a-pattern is "
            "not checked at all, and required-vs-optional only checks that the key exists",
        ),
        practice.Check(
            "FINDING: the two it skips are the two that catch hallucinated arguments",
            all(["typed ids need a pattern" in result["missing"],
                 "enum every closed set" in result["missing"]]),
            "a note_id with no pattern accepts 'note-oops' and a closed set with no enum "
            "accepts any string -- both are the shapes a model invents when it is guessing. "
            "The rules the linter does implement govern what the author wrote; the rules it "
            "skips govern what the model may send",
        ),
        practice.Check(
            "FINDING: and the new rule would fire on nothing the lesson ships",
            all([result["shipped_params"] == 10,
                 result["shipped_snake"] == result["shipped_params"],
                 result["new_rule_on_shipped"] == 0]),
            f"all {result['shipped_params']} parameter names across GOOD_REGISTRY and "
            f"BAD_REGISTRY are already snake_case, so the extension scores "
            f"{result['new_rule_on_shipped']} on both. A rule with no failing fixture is a "
            "rule nobody has tested, which is the gap Exercise 4 found in the CI gate",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
