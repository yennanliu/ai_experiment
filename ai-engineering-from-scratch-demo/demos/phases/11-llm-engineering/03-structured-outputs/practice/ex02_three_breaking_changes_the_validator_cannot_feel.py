"""Exercise 2 — the diff is right about three changes the validator cannot feel.

    Build a "schema diff" tool that compares two schemas and identifies
    breaking changes (removed required fields, changed types) versus
    non-breaking changes (added optional fields, relaxed constraints). This is
    essential for versioning your extraction schemas in production.

Reading of the exercise: "breaking" is read in the producer direction -- a
document accepted by v1 that v2 rejects -- because that is the direction a
versioned extraction schema breaks in, and it is the only direction with a
mechanical witness. Every labelled change therefore ships a witness document,
and the diff's verdict is then checked against what the lesson's own
`validate_schema` actually does to it.

**ANSWER: the diff classifies 10 of 10 labelled changes correctly**, from the
schema pair alone: added/removed `required`, changed `type`, raised `minimum`,
lowered `maximum`, added/removed `enum`, added optional property.

**FINDING: on 2 of the 10, the diff is right and the validator does not agree.**
For those two the witness document gets the identical verdict from v1 and v2,
so a production deploy of the new schema changes nothing it was supposed to
change -- and a rollback would change nothing either.

**MECHANISM: the constraints live inside the type branches.** `minimum` and
`maximum` are read only where `schema_type == "number"`, and `enum` only where
it is `"string"`. So a range on an `integer` field and an enum on an `integer`
field are decoration: `{"type": "integer", "minimum": 0}` accepts -5 and
`{"type": "integer", "enum": [1, 2]}` accepts 99.

**FINDING: narrowing `number` to `integer` loosens the schema.** The witness -5
is rejected by `{"type": "number", "minimum": 0}` and accepted by
`{"type": "integer", "minimum": 0}`, because the range check does not exist in
the integer branch. A change the diff calls breaking is a relaxation.

**CONTROL: hoisting the three checks out of the type branches costs five lines**
and takes agreement from 8 of 10 to 10 of 10, with no other verdict moving.

Structure: `diff` is the tool, `CHANGES` the labelled pairs with witnesses, and
`hoisted` the control validator.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "03-structured-outputs"
LOW, HIGH = float("-inf"), float("inf")
NUM, INT, STR = dict(type="number", minimum=0), dict(type="integer", minimum=0), dict(type="string")
NUM10, INT10 = dict(type="number", minimum=10), dict(type="integer", minimum=10)
ENUM_S, ENUM_I = dict(type="string", enum=["a", "b"]), dict(type="integer", enum=[1, 2])
MAX5, MAX10, PLAIN_I = dict(type="number", maximum=5), dict(type="number", maximum=10), \
    dict(type="integer")


def obj(properties, required):
    return {"type": "object", "properties": properties, "required": required}


def one(key, spec):
    return obj({key: spec}, [key])


# (label, v1, v2, witness, expected verdict)
CHANGES = [
    ("added required sku", one("name", STR), obj({"name": STR, "sku": STR},
     ["name", "sku"]), {"name": "a"}, "breaking"),
    ("removed required price", obj({"name": STR, "price": NUM}, ["name", "price"]),
     obj({"name": STR, "price": NUM}, ["name"]), {"name": "a"}, "non-breaking"),
    ("price string -> number", one("price", STR), one("price", NUM), {"price": "x"}, "breaking"),
    ("added optional rating", one("name", STR), obj({"name": STR, "rating": NUM},
     ["name"]), {"name": "a"}, "non-breaking"),
    ("minimum 0 -> 10, number", one("price", NUM), one("price", NUM10), {"price": 5}, "breaking"),
    ("minimum 0 -> 10, integer", one("qty", INT), one("qty", INT10), {"qty": 5}, "breaking"),
    ("enum added, string", one("tier", STR), one("tier", ENUM_S), {"tier": "z"}, "breaking"),
    ("enum added, integer", one("lvl", PLAIN_I), one("lvl", ENUM_I), {"lvl": 99}, "breaking"),
    ("maximum 5 -> 10", one("score", MAX5), one("score", MAX10), {"score": 8}, "non-breaking"),
    ("number -> integer", one("qty", NUM), one("qty", INT), {"qty": 3.5}, "breaking"),
]


def field_verdict(before, after):
    """Breaking iff v2 can reject a value v1 accepted -- an enum added or cut down,
    a floor raised, a ceiling lowered, or the type changed at all."""
    old, new = before.get("enum"), after.get("enum")
    return "breaking" if any([
        before.get("type") != after.get("type"),
        after.get("minimum", LOW) > before.get("minimum", LOW),
        after.get("maximum", HIGH) < before.get("maximum", HIGH),
        bool(new) and (not old or set(new) < set(old))]) else "non-breaking"


def diff(v1, v2):
    """The tool: one verdict for the schema pair, breaking if any part is."""
    if set(v2["required"]) - set(v1["required"]):
        return "breaking"
    shared = set(v1["properties"]) & set(v2["properties"])
    verdicts = {field_verdict(v1["properties"][k], v2["properties"][k]) for k in shared}
    return "breaking" if "breaking" in verdicts else "non-breaking"


def extra_errors(value, field):
    """The two checks the lesson keeps inside its `number` and `string` branches."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return []
    low, allowed = field.get("minimum"), field.get("enum")
    return ([": below minimum"] if low is not None and value < low else []) + (
        [": not in enum"] if allowed and value not in allowed else [])


def hoisted(ref, data, schema):
    """The control: the lesson's validator, plus those two checks for any type."""
    extra = [extra_errors(data.get(k), f) for k, f in schema["properties"].items()]
    return list(ref.validate_schema(data, schema)) + sum(extra, [])


def observed(check, v1, v2, witness):
    return "breaking" if check(witness, v1) == [] and check(witness, v2) else "non-breaking"


def probes(ref):
    return {"int_min": ref.validate_schema({"qty": -5}, one("qty", INT)),
            "int_enum": ref.validate_schema({"lvl": 99}, one("lvl", ENUM_I)),
            "narrow_v1": ref.validate_schema({"qty": -5}, one("qty", NUM)),
            "narrow_v2": ref.validate_schema({"qty": -5}, one("qty", INT))}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shipped = [observed(ref.validate_schema, a, b, w) for _, a, b, w, _ in CHANGES]
    control = [observed(lambda d, s: hoisted(ref, d, s), a, b, w) for _, a, b, w, _ in CHANGES]
    classified = [diff(a, b) for _, a, b, _, _ in CHANGES]
    expected = [want for *_, want in CHANGES]
    return {"classified": classified, "expected": expected, "changes": len(CHANGES),
            **agreement(classified, expected, shipped, control), **probes(ref)}


def agreement(classified, expected, shipped, control):
    return {"correct": sum(map(str.__eq__, classified, expected)),
            "agree": sum(map(str.__eq__, classified, shipped)),
            "control_agree": sum(map(str.__eq__, classified, control)),
            "silent": [c[0] for c, g, w in zip(CHANGES, shipped, classified) if g != w]}


def verify(result):
    return [
        practice.Check(
            "ANSWER: the diff classifies 10 of 10 labelled changes correctly",
            all([result["correct"] == result["changes"],
                 result["classified"] == result["expected"]]),
            f"{result['correct']} of {result['changes']} from the schema pair alone. "
            "Breaking is read in the producer direction -- a document v1 accepted that "
            "v2 rejects -- the only direction with a mechanical witness",
        ),
        practice.Check(
            "FINDING: on 2 of the 10 the diff is right and the validator does not agree",
            all([result["agree"] == 8, len(result["silent"]) == 2]),
            f"the witness gets the identical verdict from v1 and v2 on {result['silent']}. "
            f"Deploying those versions changes nothing they were meant to change, and "
            f"rolling them back changes nothing -- agreement {result['agree']} of "
            f"{result['changes']}",
        ),
        practice.Check(
            "MECHANISM: the constraints live inside the type branches",
            all([result["int_min"] == [], result["int_enum"] == []]),
            "`minimum` and `maximum` are read only where schema_type == 'number' and "
            "`enum` only where it is 'string', so {'type': 'integer', 'minimum': 0} accepts "
            "-5 and {'type': 'integer', 'enum': [1, 2]} accepts 99: decoration, on an int",
        ),
        practice.Check(
            "FINDING: narrowing number to integer loosens the schema",
            all([result["narrow_v1"] != [], result["narrow_v2"] == []]),
            f"-5 is rejected by {{'type': 'number', 'minimum': 0}} ({result['narrow_v1']}) "
            f"and accepted by {{'type': 'integer', 'minimum': 0}}, because the range check "
            "does not exist in the integer branch. A breaking change that relaxes",
        ),
        practice.Check(
            "CONTROL: hoisting range and enum out of the branches restores agreement",
            result["control_agree"] == result["changes"],
            f"checking `minimum` and `enum` on any numeric value, outside the type "
            f"dispatch, takes agreement {result['agree']} -> {result['control_agree']} of "
            f"{result['changes']}, no other verdict moving. The diff was never the weak half",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
