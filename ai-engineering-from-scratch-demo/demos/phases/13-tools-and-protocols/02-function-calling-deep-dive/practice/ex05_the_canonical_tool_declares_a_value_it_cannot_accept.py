"""Exercise 5 — the canonical tool declares a value it cannot accept.

    Write a test vector: a tool call whose arguments violate the declared
    schema. Run it through each provider's validator (the stdlib one in Lesson
    01 will do as a proxy) and record which errors fire. Document which provider
    you would use in production for strictness.

Reading of the exercise: the test vector is written against the lesson's own
canonical schema rather than an invented one, and is run through Lesson 01's
`validate` as the exercise permits. Writing a vector that violates the schema
means first reading the schema, and reading it turns up a value that satisfies
the declared type and violates the declared enum -- so the most interesting
vector is one the lesson wrote itself.

**ANSWER: four vectors fire four errors, and the extra field fires none.**
Missing `units` gives `missing required field 'units'`; `city: 42` gives
`expected string, got int`; `units: "kelvin"` gives the enum error; and the
undeclared `evil` key gives **`[]`**, because Lesson 01's validator has no
`additionalProperties` support even though the canonical schema sets it to
`false`.

**FINDING: `units: null` is declared valid by its type and rejected by its
enum.** `"type": ["string", "null"]` permits null, `enum: ["celsius",
"fahrenheit"]` does not contain it, and `required` lists `units`. So there is no
value of `units` that exercises the null branch: the schema declares a
nullability it then forbids. The validator reports
`value None not in enum [...]` -- correctly, against a schema that contradicts
itself.

**FINDING: the union type is never type-checked at all.** `validate` dispatches
on `schema.get("type")` with `== "string"` and `== "number"`, so a *list* of
types matches neither branch and falls through. Removing the enum as a control
and re-running, `units` accepts **7, None, "celsius" and {"a": 1}** -- 4 of 4,
all clean. The only thing rejecting a bad `units` today is the enum; the type is
decorative.

**ANSWER: OpenAI, for strictness -- and the reason is the field the other two
drop.** `strict: true` makes the provider enforce the schema by constrained
decoding, so a violating call is not generated rather than caught afterwards.
Exercise 1 measured what happens to that field in translation: it survives **1
of 3**. Anthropic and Gemini validate server-side and return an error, which
still costs a turn; strictness you enforce before generation is the only kind
that costs nothing.

Structure: `VECTORS` is the test vector table, `fire` runs one through Lesson
01's validator, and `without_enum` is the control that isolates the type check
from the enum check.
"""

from __future__ import annotations

import copy

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "02-function-calling-deep-dive"
PROXY = "01-the-tool-interface"
VECTORS = {
    "missing required units": {"city": "Bengaluru"},
    "city is an int": {"city": 42, "units": "celsius"},
    "units outside the enum": {"city": "Bengaluru", "units": "kelvin"},
    "units is null": {"city": "Bengaluru", "units": None},
    "undeclared extra key": {"city": "Bengaluru", "units": "celsius", "evil": 1},
    "well formed": {"city": "Bengaluru", "units": "celsius"},
}
CONTROL_VALUES = (7, None, "celsius", {"a": 1})


def fire(validator, schema, args):
    return validator(schema, args)


def without_enum(schema):
    """The same schema with units' enum removed, isolating the type check."""
    stripped = copy.deepcopy(schema)
    del stripped["properties"]["units"]["enum"]
    return stripped


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    proxy = parity.load_reference(PHASE, PROXY, "main")
    schema = ref.WEATHER.input_schema
    fired = {label: fire(proxy.validate, schema, args) for label, args in VECTORS.items()}
    control = {repr(value): fire(proxy.validate, without_enum(schema),
                                 {"city": "B", "units": value})
               for value in CONTROL_VALUES}
    return {
        "fired": fired,
        "errors_raised": sum(1 for label, errs in fired.items()
                             if errs and label != "well formed"),
        "extra_key_clean": fired["undeclared extra key"] == [],
        "well_formed_clean": fired["well formed"] == [],
        "units_type": schema["properties"]["units"]["type"],
        "units_enum": schema["properties"]["units"]["enum"],
        "null_in_enum": None in schema["properties"]["units"]["enum"],
        "units_required": "units" in schema["required"],
        "additional_declared": schema["additionalProperties"],
        "control": control,
        "control_clean": sum(errs == [] for errs in control.values()),
        "control_size": len(CONTROL_VALUES),
        "strict_flag": ref.WEATHER.strict,
    }


def verify(result):
    fired = result["fired"]
    return [
        practice.Check(
            "ANSWER: four vectors fire four errors, and the extra field fires none",
            all([fired["missing required units"] == ["missing required field 'units'"],
                 fired["city is an int"] == ["expected string, got int"],
                 fired["units outside the enum"] ==
                 ["value 'kelvin' not in enum ['celsius', 'fahrenheit']"],
                 result["errors_raised"] == 4, result["extra_key_clean"],
                 result["well_formed_clean"],
                 result["additional_declared"] is False]),
            f"{result['errors_raised']} of the five violating vectors raise: {fired}. The "
            f"undeclared key raises nothing although the canonical schema sets "
            f"additionalProperties to {result['additional_declared']} -- Lesson 01's "
            "validator has no support for the field",
        ),
        practice.Check(
            "FINDING: units: null is declared valid by its type and rejected by its enum",
            all([result["units_type"] == ["string", "null"],
                 not result["null_in_enum"], result["units_required"],
                 fired["units is null"] ==
                 ["value None not in enum ['celsius', 'fahrenheit']"]]),
            f"units is declared type {result['units_type']}, which permits null, while its "
            f"enum {result['units_enum']} does not contain it and required lists units. No "
            f"value of units exercises the null branch: the schema declares a nullability it "
            f"then forbids, and the validator reports {fired['units is null']} -- correctly, "
            "against a schema that contradicts itself",
        ),
        practice.Check(
            "FINDING: the union type is never type-checked at all",
            all([result["control_clean"] == result["control_size"],
                 result["control_clean"] == 4]),
            f"validate dispatches on schema.get('type') with == 'string' and == 'number', so "
            f"a list of types matches neither branch and falls through. Removing the enum as "
            f"a control, units accepts {list(result['control'])} -- "
            f"{result['control_clean']} of {result['control_size']}, all clean. The only "
            "thing rejecting a bad units today is the enum; the type is decorative",
        ),
        practice.Check(
            "ANSWER: OpenAI for strictness, because of the field the other two drop",
            all([result["strict_flag"] is True]),
            f"the canonical tool sets strict={result['strict_flag']}, which makes the "
            "provider enforce the schema by constrained decoding, so a violating call is not "
            "generated rather than caught afterwards. Exercise 1 measured what translation "
            "does to that field: it survives 1 of 3. Anthropic and Gemini validate "
            "server-side and return an error, which still costs a turn -- strictness enforced "
            "before generation is the only kind that costs nothing",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
