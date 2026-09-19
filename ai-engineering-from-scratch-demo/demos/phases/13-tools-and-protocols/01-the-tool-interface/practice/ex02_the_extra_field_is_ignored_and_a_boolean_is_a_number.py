"""Exercise 2 — the extra field is ignored, and a boolean is a number.

    Break the schema validator. Pass a call whose `arguments` object is missing
    a required field, and confirm the host rejects it before execution. Then
    pass a call with an extra unknown field. Decide: should the host reject or
    ignore? Justify your choice with a safety argument.

Reading of the exercise: both calls are put through the lesson's own `validate`
rather than reasoned about, and the "decide" half is answered with the argument
the measurement supports. Trying to break the validator on the two inputs the
exercise names also turns up a third that breaks it harder and is not asked
about, so it is reported.

**ANSWER: missing required is rejected; the extra field is silently ignored.**
`{"a": 1}` returns `["missing required field 'b'"]` and never reaches the
executor. `{"a": 1, "b": 2, "evil": "x"}` returns **`[]`** -- `validate` walks
the *declared* properties and never enumerates the supplied keys, so anything
undeclared passes through untouched.

**ANSWER: the host should reject, because ignoring is not the same as
discarding.** The validator ignoring a key does not remove it: the same dict is
handed to `executor(call["arguments"])`, so an undeclared key is still there for
the executor to read. Ignoring is only safe if every executor also ignores it,
which is a property of code nobody validated. Rejecting is the one choice whose
safety does not depend on the tool's implementation -- and JSON Schema spells it
`additionalProperties: false`, which this validator does not implement.

**FINDING: a boolean validates as a number.** `{"a": true, "b": false}` returns
**`[]`** and `tool_add` executes to **`{"sum": 1}`**. The check is
`isinstance(value, (int, float))`, and in Python `isinstance(True, int)` is
`True` -- so the type gate the exercise calls a safety boundary admits a value
JSON Schema excludes, and the tool does arithmetic on it. That is a worse break
than either the exercise proposes, and it passes the same "confirm the host
rejects it" test the exercise is built around.

Structure: `probe` runs one arguments dict through `validate` and, when it
passes, through the executor too; `CASES` are the two the exercise names plus
the boolean; and `enumerates_keys` checks whether `validate` ever looks at the
supplied keys.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "01-the-tool-interface"
MISSING = {"a": 1}
EXTRA = {"a": 1, "b": 2, "evil": "x"}
BOOLEAN = {"a": True, "b": False}
WELL_FORMED = {"a": 7, "b": 35}


def tools(ref):
    return {tool.name: tool for tool in ref.REGISTRY}


def probe(ref, args, name="add"):
    """Validate one arguments dict, and execute it if the validator let it pass."""
    tool = tools(ref)[name]
    errors = ref.validate(tool.input_schema, args)
    return {"errors": errors,
            "executed": tool.executor(args) if not errors else None}


def enumerates_keys(ref):
    """Does validate ever iterate the keys it was given, rather than the declared ones?"""
    source = inspect.getsource(ref.validate)
    return "for key in value" in source or "value.keys()" in source


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    missing, extra = probe(ref, MISSING), probe(ref, EXTRA)
    boolean, good = probe(ref, BOOLEAN), probe(ref, WELL_FORMED)
    return {
        "missing_errors": missing["errors"], "missing_blocked": missing["executed"] is None,
        "extra_errors": extra["errors"], "extra_executed": extra["executed"],
        "extra_key_survives": "evil" in EXTRA,
        "boolean_errors": boolean["errors"], "boolean_executed": boolean["executed"],
        "baseline": good["executed"],
        "enumerates_keys": enumerates_keys(ref),
        "supports_additional_properties":
            "additionalProperties" in inspect.getsource(ref.validate),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: missing required is rejected; the extra field is silently ignored",
            all([result["missing_errors"] == ["missing required field 'b'"],
                 result["missing_blocked"], result["extra_errors"] == [],
                 result["extra_executed"] == {"sum": 3},
                 not result["enumerates_keys"]]),
            f"{MISSING} returns {result['missing_errors']} and never reaches the executor. "
            f"{EXTRA} returns {result['extra_errors']} and executes to "
            f"{result['extra_executed']} -- validate walks the declared properties and never "
            "enumerates the supplied keys, so anything undeclared passes through untouched",
        ),
        practice.Check(
            "ANSWER: the host should reject, because ignoring is not discarding",
            all([result["extra_errors"] == [], result["extra_key_survives"],
                 not result["supports_additional_properties"]]),
            "the validator ignoring a key does not remove it: the same dict is handed to "
            "executor(call['arguments']), so an undeclared key is still there for the "
            "executor to read. Ignoring is safe only if every executor also ignores it, "
            "which is a property of code nobody validated; rejecting is the one choice whose "
            "safety does not depend on the tool's implementation. JSON Schema spells it "
            "additionalProperties: false, which this validator does not implement",
        ),
        practice.Check(
            "FINDING: a boolean validates as a number and the tool adds it",
            all([result["boolean_errors"] == [],
                 result["boolean_executed"] == {"sum": 1},
                 result["baseline"] == {"sum": 42}]),
            f"{BOOLEAN} returns {result['boolean_errors']} and tool_add executes to "
            f"{result['boolean_executed']}. The check is isinstance(value, (int, float)) and "
            f"in Python isinstance(True, int) is True, so the type gate admits a value JSON "
            f"Schema excludes and the tool does arithmetic on it -- a worse break than either "
            f"the exercise proposes, passing the same test it is built around "
            f"(the well-formed call gives {result['baseline']})",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
