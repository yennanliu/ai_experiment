"""Exercise 2 — coercion uses Python's number parser, not the wire's.

    Implement argument coercion for int-as-string and float-as-string. Where
    does coercion start to hide real bugs?

Reading of the exercise: `_coerce` already does both -- `int(value)` and
`float(value)` on a `str` branch -- so the implementation half is shipped and
the question is the whole exercise. It is answered by probing which strings
the coercion accepts and comparing that set against the strings a provider
could actually have emitted, which is decided by `json.loads`.

**ANSWER: both coercions are already there, and they are asymmetric.**
`"5"` becomes `5`, `"5.5"` becomes `5.5`, and `"5.0"` is a valid `number`
and an invalid `integer` -- the same token, accepted or rejected by the
schema's type rather than by its shape.

**FINDING: `"nan"` passes a bounded range.** With
`{"type": "number", "minimum": 0, "maximum": 100}`, the string `nan` coerces
to `float('nan')` and clears both bounds, because every comparison with NaN
is false. It is the one value in the probe set that satisfies a range it is
not in, and `"inf"` -- which is *not* silently accepted -- gets a clean error.

**FINDING: the coercion is Python's number parser, not JSON's.** Of the
**7** strings the validator accepts, only **3** are JSON number tokens.
`"5_0"` becomes **50** on Python's underscore separators, `"  7  "` becomes
**7** on whitespace stripping, `"٥"` becomes **5** because `int()` reads
Unicode decimal digits, and `"nan"` becomes NaN. No provider's JSON could
have produced any of them as a number.

**FINDING: the executor never sees what the model sent.** `validate` returns
`out[name] = coerced` and `dispatch` calls `executor(**validated)`, so a tool
handed `"5_0"` receives `50`, the error list is empty, and nothing anywhere
records the original token. Coercion starts hiding bugs exactly here -- not
when it converts, but when it converts and discards.

Structure: `probe()` runs one value through the lesson's own `validate`;
`json_number()` is the independent oracle for what the wire could carry.
"""

from __future__ import annotations

import json
import math

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "06-tool-use-and-function-calling"
NUMBER = {"type": "object", "required": ["x"],
          "properties": {"x": {"type": "number", "minimum": 0, "maximum": 100}}}
INTEGER = {"type": "object", "required": ["n"],
           "properties": {"n": {"type": "integer", "minimum": 1, "maximum": 10}}}
PROBES = ("5", "5.5", "5.0", "nan", "inf", "5_0", "  7  ", "٥", "0x10")


ECHO = {"type": "object", "required": ["n"], "properties": {"n": {"type": "integer"}}}


def _reject(name):
    raise ValueError(f"{name} is not a JSON number")


def json_number(text):
    """Could a provider have sent this as a JSON number? NaN and Infinity could not."""
    try:
        value = json.loads(text, parse_constant=_reject)
    except ValueError:
        return False
    return (text == text.strip() and isinstance(value, (int, float))
            and not isinstance(value, bool))


def probe(ref, schema, key, value):
    out, errors = ref.validate({key: value}, schema)
    return {"accepted": not errors, "value": out.get(key), "errors": errors}


def registry(ref):
    tools = ref.ToolRegistry()
    tools.register(ref.ToolDef(name="echo", description="Return the argument as seen.",
                               input_schema=ECHO, executor=lambda n: f"{n!r}"))
    return tools


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    numbers = {text: probe(ref, NUMBER, "x", text) for text in PROBES}
    accepted = sorted(text for text, row in numbers.items() if row["accepted"])
    seen = registry(ref).dispatch(ref.ToolCall("u1", "echo", {"n": "5_0"}))
    return {
        "as_int": probe(ref, INTEGER, "n", "5"),
        "as_number": probe(ref, NUMBER, "x", "5.5"),
        "float_only": (probe(ref, NUMBER, "x", "5.0")["accepted"],
                       probe(ref, INTEGER, "n", "5.0")["accepted"]),
        "nan_accepted": numbers["nan"]["accepted"],
        "nan_is_nan": math.isnan(numbers["nan"]["value"] or 0.0),
        "inf_errors": numbers["inf"]["errors"],
        "accepted": accepted,
        "wire_legal": sorted(text for text in accepted if json_number(text)),
        "underscore": numbers["5_0"]["value"], "padded": numbers["  7  "]["value"],
        "unicode": numbers["٥"]["value"],
        "executor_saw": seen.content, "executor_ok": seen.ok,
        "raw_recorded": "5_0" in seen.content,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: both coercions are shipped, and they disagree on '5.0'",
            all([result["as_int"] == {"accepted": True, "value": 5, "errors": []},
                 result["as_number"]["value"] == 5.5,
                 result["float_only"] == (True, False)]),
            f"'5' coerces to {result['as_int']['value']} under integer and '5.5' to "
            f"{result['as_number']['value']} under number, so the exercise's "
            f"implementation half is already shipped. '5.0' is accepted as a number and "
            f"rejected as an integer {result['float_only']} -- the same token, judged by "
            "the schema's type rather than by its shape",
        ),
        practice.Check(
            "FINDING: 'nan' passes a bounded range that excludes it",
            all([result["nan_accepted"] is True, result["nan_is_nan"] is True,
                 len(result["inf_errors"]) == 1,
                 "maximum" in result["inf_errors"][0]]),
            f"against minimum 0 and maximum 100, 'nan' is accepted "
            f"({result['nan_accepted']}) because every comparison with NaN is false, "
            f"while 'inf' gets a clean {result['inf_errors'][0]!r}. One value in the "
            "probe set satisfies a range it is not in, and it is the one JSON cannot even "
            "express",
        ),
        practice.Check(
            "FINDING: the parser is Python's, not the wire's",
            all([len(result["accepted"]) == 7, "nan" in result["accepted"],
                 result["wire_legal"] == ["5", "5.0", "5.5"],
                 result["underscore"] == 50.0, result["padded"] == 7.0,
                 result["unicode"] == 5.0]),
            f"the validator accepts {len(result['accepted'])} of the probe strings and "
            f"only {len(result['wire_legal'])} of them are JSON number literals: "
            f"'5_0' becomes {result['underscore']}, '  7  ' becomes "
            f"{result['padded']} and a Unicode digit becomes {result['unicode']}. No "
            "provider's JSON could have sent any of those three as a number",
        ),
        practice.Check(
            "FINDING: the executor never sees what the model sent",
            all([result["executor_ok"] is True, result["executor_saw"] == "50",
                 result["raw_recorded"] is False]),
            f"a tool handed '5_0' is called with {result['executor_saw']} and the "
            f"dispatch reports ok={result['executor_ok']} with the original token "
            f"recorded nowhere ({result['raw_recorded']}). Coercion starts hiding bugs "
            "not when it converts but when it converts and discards",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
