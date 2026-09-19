"""Exercise 2 — an unsupported keyword accepts everything instead of failing.

    Extend the validator to support `oneOf` with a discriminator. The common
    case: `line_item` is either a product or a service, tagged by `kind`.
    Strict mode has subtle rules here; check OpenAI's structured outputs guide.

Reading of the exercise: the extension is written, and the behaviour it replaces
is measured first -- because a validator that does not know a keyword has two
options, reject the schema or ignore it, and which one this validator picked
decides how bad the gap is. It ignores it. The discriminated form is then built
and run against both branches and a mismatch.

**ANSWER: `oneOf` plus a `kind` discriminator, dispatching on the tag before
validating.** A product carries `sku` and `unit_usd`, a service carries `hours`
and `rate_usd`, and both carry `kind`. The extension reads the tag, picks the
one branch it names, and reports errors against that branch alone -- so a
mistyped `hours` on a service is one error at `$.hours`, not six spread across
two branches.

**FINDING: the lesson's validator ignores `oneOf` entirely, which is worse than
rejecting it.** `validate({"oneOf": [...]}, 5)` returns **no errors**. There is
no `type` key, so every branch of the dispatch is skipped and the function falls
through to the shared constraint checks, which find nothing to check. A schema
built on `oneOf` therefore validates **every input**, and the failure is silent:
an unsupported keyword reads as an unconstrained one.

**FINDING: dispatching on the tag is what makes the errors usable.** Validating
a bad service against both branches without the discriminator produces **6**
errors, **5** of them about the product branch the document never claimed to be
-- a missing `sku`, a missing `unit_usd`, two properties "not allowed", and a
tag outside `product`'s enum. With the tag it is **1**, the one that is actually
wrong. The discriminator is not an optimisation; it is what stops an error
message from describing a shape the author never intended.

**FINDING: the strict-mode subtlety is that the tag has to be an `enum` of one.**
A discriminated union is only decidable if each branch pins its own tag to a
single value, and `"kind": {"type": "string"}` does not -- both branches would
accept any tag and the union stops being one. Pinning it with
`{"enum": ["product"]}` makes the branch selectable by a grammar, which is the
property constrained decoding needs.

Structure: `PRODUCT` and `SERVICE` are the two branches, `discriminated` is the
schema the exercise asks for, `validate_one_of` is the extension, and `naive`
is the same check without the tag, to price it.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "04-structured-output"
TAG = "kind"
PRODUCT = {
    "type": "object",
    "properties": {TAG: {"type": "string", "enum": ["product"]},
                   "sku": {"type": "string", "pattern": "^[A-Z0-9-]+$"},
                   "unit_usd": {"type": "number", "minimum": 0}},
    "required": [TAG, "sku", "unit_usd"], "additionalProperties": False,
}
SERVICE = {
    "type": "object",
    "properties": {TAG: {"type": "string", "enum": ["service"]},
                   "hours": {"type": "number", "minimum": 0},
                   "rate_usd": {"type": "number", "minimum": 0}},
    "required": [TAG, "hours", "rate_usd"], "additionalProperties": False,
}
DISCRIMINATED = {"oneOf": [PRODUCT, SERVICE], "discriminator": {"propertyName": TAG}}
GOOD_PRODUCT = {TAG: "product", "sku": "ABC-1", "unit_usd": 10.0}
GOOD_SERVICE = {TAG: "service", "hours": 2.0, "rate_usd": 150.0}
BAD_SERVICE = {TAG: "service", "hours": "two", "rate_usd": 150.0}
UNTAGGED = {TAG: "rental", "hours": 2.0, "rate_usd": 150.0}


def branch_for(schema, value, tag=TAG):
    """The one branch whose tag enum names this document's tag."""
    for candidate in schema["oneOf"]:
        allowed = candidate["properties"][tag].get("enum", [])
        if isinstance(value, dict) and value.get(tag) in allowed:
            return candidate
    return None


def validate_one_of(ref, schema, value, tag=TAG):
    """The extension: dispatch on the discriminator, then defer to the lesson."""
    chosen = branch_for(schema, value, tag)
    if chosen is None:
        named = [b["properties"][tag]["enum"][0] for b in schema["oneOf"]]
        return [f"$.{tag}: {value.get(tag)!r} matches none of {named}"]
    return [str(error) for error in ref.validate(chosen, value)]


def naive(ref, schema, value):
    """Every branch tried, which is what oneOf means without a discriminator."""
    return [str(error) for candidate in schema["oneOf"]
            for error in ref.validate(candidate, value)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ignored = [str(error) for error in ref.validate(DISCRIMINATED, 5)]
    tags = [branch["properties"][TAG] for branch in DISCRIMINATED["oneOf"]]
    return {
        "ignored_errors": ignored, "ignores_one_of": ignored == [],
        "good_product": validate_one_of(ref, DISCRIMINATED, GOOD_PRODUCT),
        "good_service": validate_one_of(ref, DISCRIMINATED, GOOD_SERVICE),
        "bad_service": validate_one_of(ref, DISCRIMINATED, BAD_SERVICE),
        "bad_service_naive": naive(ref, DISCRIMINATED, BAD_SERVICE),
        "wrong_branch": len(naive(ref, DISCRIMINATED, BAD_SERVICE))
                        - len(validate_one_of(ref, DISCRIMINATED, BAD_SERVICE)),
        "untagged": validate_one_of(ref, DISCRIMINATED, UNTAGGED),
        "tags_are_singleton_enums": all(len(tag.get("enum", [])) == 1 for tag in tags),
        "tag_values": [tag["enum"][0] for tag in tags],
        "branches": len(DISCRIMINATED["oneOf"]),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: oneOf plus a kind discriminator, dispatching before validating",
            all([result["branches"] == 2, result["tag_values"] == ["product", "service"],
                 result["good_product"] == [], result["good_service"] == [],
                 result["bad_service"] == ["$.hours: expected number, got str"]]),
            f"{result['branches']} branches tagged {result['tag_values']}. Both well-formed "
            f"documents validate clean, and a service whose hours is a string gives "
            f"{result['bad_service']} -- one error, against the branch the tag named",
        ),
        practice.Check(
            "FINDING: the lesson's validator ignores oneOf, which is worse than rejecting it",
            all([result["ignores_one_of"], result["ignored_errors"] == []]),
            f"validate on a oneOf schema against the integer 5 returns "
            f"{result['ignored_errors']}. There is no type key, so every branch of the "
            "dispatch is skipped and the shared constraint checks find nothing to check. A "
            "schema built on oneOf validates every input, and the failure is silent: an "
            "unsupported keyword reads as an unconstrained one",
        ),
        practice.Check(
            "FINDING: dispatching on the tag is what makes the errors usable",
            all([len(result["bad_service_naive"]) == 6, len(result["bad_service"]) == 1,
                 result["wrong_branch"] == 5,
                 result["untagged"] ==
                 ["$.kind: 'rental' matches none of ['product', 'service']"]]),
            f"checking the same bad service against both branches gives "
            f"{len(result['bad_service_naive'])} errors, {result['wrong_branch']} of them "
            f"about the product branch the document never claimed to be -- missing sku and "
            f"unit_usd, two properties 'not allowed', and a tag outside product's enum. With "
            f"the tag it is {len(result['bad_service'])}, the one that is actually wrong. An "
            f"unknown tag is then its own error: {result['untagged']}",
        ),
        practice.Check(
            "FINDING: the strict-mode subtlety is that the tag must be an enum of one",
            all([result["tags_are_singleton_enums"],
                 len(result["tag_values"]) == result["branches"],
                 len(set(result["tag_values"])) == result["branches"]]),
            f"each branch pins its own tag to a single value ({result['tag_values']}, all "
            "distinct). A union is only decidable if the tag selects exactly one branch; "
            "'kind': {'type': 'string'} would let both branches accept any tag and the union "
            "would stop being one. Pinning it makes the branch selectable by a grammar, "
            "which is the property constrained decoding needs",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
