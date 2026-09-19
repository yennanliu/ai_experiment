"""Exercise 1 — the one quirk it names is the one the translator misses.

    Run `code/main.py` and verify that the three provider declaration JSONs all
    serialize the same underlying `Tool` object. Modify the canonical tool to
    add an enum parameter and confirm only the Gemini translator needs to
    handle the OpenAPI quirk.

Reading of the exercise: "verify" is taken literally -- the three payloads are
compared field by field rather than eyeballed -- and the second half needs no
modification, because the canonical tool already carries an enum parameter. So
the enum is exercised where it already sits, on `units`, and the claim that only
Gemini needs special handling is checked against what Gemini's translator
actually emits.

**ANSWER: they do not all serialize the same object.** `strict=True` reaches
OpenAI and **neither** of the other two: `to_anthropic` passes three fields and
`to_gemini` passes three, and `strict` is not among them. One of the canonical
`Tool`'s four fields survives **1 of 3** translations.

**ANSWER: the enum parameter is already there.** `units` is declared
`enum: ["celsius", "fahrenheit"]` in `WEATHER.input_schema`, so the exercise's
"modify the canonical tool to add" step is already done by the lesson.

**FINDING: Gemini needs the OpenAPI quirk handled and does not handle it.**
`_gemini_schema` uppercases `type` only `if isinstance(v, str)`. `units` is
declared `"type": ["string", "null"]` -- a **list** -- so it falls through
untouched and Gemini receives lowercase `["string", "null"]` beside its
uppercased siblings `OBJECT` and `STRING`. OpenAPI 3.0, which Gemini's schema
dialect follows, has no union types at all: the correct emission is
`type: "STRING"` with `nullable: true`. The exercise says only Gemini needs to
handle the quirk, and it is right; the translator is what is wrong.

**FINDING: `additionalProperties` is the field Gemini drops on purpose.** It is
stripped by name, so Gemini sees a schema that permits undeclared keys while
OpenAI and Anthropic see one that forbids them. That is a correct translation of
an untranslatable field, and it is the only place the three declarations are
*meant* to disagree -- which makes it the control for the `strict` loss above.

Structure: `declarations` emits all three payloads, `type_values` collects every
`type` in a schema tree so the uppercase sweep can be checked, and
`carries_strict` looks for the field in each payload.
"""

from __future__ import annotations

import json

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "02-function-calling-deep-dive"


def declarations(ref, tool):
    return {"openai": ref.to_openai(tool), "anthropic": ref.to_anthropic(tool),
            "gemini": ref.to_gemini(tool)}


def type_values(node):
    """Every value a `type` key takes anywhere in a schema tree."""
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "type":
                found.append(value)
            else:
                found.extend(type_values(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(type_values(item))
    return found


def carries_strict(payloads):
    return {"openai": "strict" in payloads["openai"]["function"],
            "anthropic": "strict" in payloads["anthropic"],
            "gemini": "strict" in payloads["gemini"]["functionDeclarations"][0]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    tool = ref.WEATHER
    payloads = declarations(ref, tool)
    gemini_params = payloads["gemini"]["functionDeclarations"][0]["parameters"]
    types = type_values(gemini_params)
    strict = carries_strict(payloads)
    return {
        "strict_by_provider": strict, "strict_survives": sum(strict.values()),
        "canonical_fields": sorted(tool.__dataclass_fields__),
        "anthropic_keys": sorted(payloads["anthropic"]),
        "units_enum": tool.input_schema["properties"]["units"]["enum"],
        "units_type": tool.input_schema["properties"]["units"]["type"],
        "gemini_types": types,
        "lowercase_types": [value for value in types if not isinstance(value, str)],
        "uppercased": [value for value in types if isinstance(value, str)],
        "gemini_units_type": gemini_params["properties"]["units"]["type"],
        "gemini_units_enum": gemini_params["properties"]["units"]["enum"],
        "canonical_additional": tool.input_schema["additionalProperties"],
        "gemini_has_additional": "additionalProperties" in json.dumps(payloads["gemini"]),
        "openai_has_additional":
            "additionalProperties" in json.dumps(payloads["openai"]),
        "anthropic_has_additional":
            "additionalProperties" in json.dumps(payloads["anthropic"]),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: they do not all serialize the same object -- strict survives 1 of 3",
            all([result["strict_by_provider"] ==
                 {"openai": True, "anthropic": False, "gemini": False},
                 result["strict_survives"] == 1,
                 "strict" in result["canonical_fields"],
                 result["anthropic_keys"] == ["description", "input_schema", "name"]]),
            f"the canonical Tool carries {result['canonical_fields']}, and strict reaches "
            f"{result['strict_by_provider']} -- {result['strict_survives']} of 3. "
            f"to_anthropic passes {result['anthropic_keys']} and to_gemini passes name, "
            "description and parameters, so the field OpenAI uses to guarantee schema "
            "compliance is dropped by both other translators without comment",
        ),
        practice.Check(
            "ANSWER: the enum parameter is already there",
            all([result["units_enum"] == ["celsius", "fahrenheit"],
                 result["gemini_units_enum"] == ["celsius", "fahrenheit"]]),
            f"units is declared enum {result['units_enum']} in WEATHER.input_schema, so the "
            f"exercise's 'modify the canonical tool to add an enum parameter' is already done "
            f"by the lesson, and Gemini carries the enum through unchanged as "
            f"{result['gemini_units_enum']}",
        ),
        practice.Check(
            "FINDING: Gemini needs the OpenAPI quirk handled and does not handle it",
            all([result["units_type"] == ["string", "null"],
                 result["gemini_units_type"] == ["string", "null"],
                 result["uppercased"] == ["OBJECT", "STRING"],
                 result["lowercase_types"] == [["string", "null"]]]),
            f"_gemini_schema uppercases type only if isinstance(v, str). units is declared "
            f"{result['units_type']} -- a list -- so it falls through untouched: Gemini "
            f"receives {result['gemini_units_type']} beside its uppercased siblings "
            f"{result['uppercased']}. OpenAPI 3.0, the dialect Gemini follows, has no union "
            "types; the correct emission is type STRING with nullable true. The exercise is "
            "right that only Gemini needs the quirk handled -- the translator is what is wrong",
        ),
        practice.Check(
            "FINDING: additionalProperties is the field Gemini drops on purpose",
            all([result["canonical_additional"] is False,
                 not result["gemini_has_additional"],
                 result["openai_has_additional"], result["anthropic_has_additional"]]),
            f"the canonical schema sets additionalProperties {result['canonical_additional']}; "
            f"_gemini_schema strips it by name, so Gemini sees a schema permitting undeclared "
            f"keys while OpenAI and Anthropic see one forbidding them. That is a correct "
            "translation of an untranslatable field, and the only place the three are meant "
            "to disagree -- which makes it the control for the strict loss above",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
