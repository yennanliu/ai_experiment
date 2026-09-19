"""Exercise 5 — the forbidden construct is the optional property.

    Read OpenAI's structured outputs guide top to bottom. Identify the one
    construct it explicitly forbids in strict mode that plain JSON Schema
    allows. Then design a schema that uses the forbidden construct
    non-essentially and refactor it to be strict-compatible.

Reading of the exercise: the construct is named, a schema is built that uses it
for no reason, and the refactor is run against both a document that exercises
the optional field and one that omits it -- because "strict-compatible" is only
a real claim if the refactored schema still accepts and rejects the same
documents. The lesson's own Invoice schema is then checked against the same
rule, since it is the schema the exercise is set beside.

**ANSWER: the optional property -- a field in `properties` and absent from
`required`.** Plain JSON Schema treats absence as permitted; strict mode
requires every declared property to be listed in `required`, because a grammar
that may or may not emit a key cannot be built from the schema alone. The
refactor is mechanical: add the key to `required` and widen its type to include
`null`, so "absent" becomes "present and null".

**FINDING: the two forms accept different documents, and that is the point.**
Against `{"customer": "Acme"}` -- no `po_number` -- the loose schema reports
**0** errors and the strict one reports **1**, `missing required field`. Against
`{"customer": "Acme", "po_number": null}` the loose schema reports **1**, a type
error, and the strict one **0**. The refactor does not preserve the accepted
set; it moves optionality out of the schema's shape and into its values, which
is what makes it expressible as a grammar.

**FINDING: the lesson's own Invoice schema is already strict-compatible on this
axis.** All **4** top-level properties are in `required`, and all **3** of the
line-item properties are. `additionalProperties` is `False` at both levels.
Whatever else the schema does, it does not use the forbidden construct -- so the
exercise has to supply its own example, which is why it says "non-essentially".

**FINDING: the refactor is correct as a schema and unenforceable by this
validator.** Widening `po_number` to `["null", "string"]` makes its type a
*list*, and `validate` dispatches with `== "string"` and `== "number"` -- so a
list matches no branch and falls through. **0 of 3** nonsense values (`42`,
`[]`, `{"a": 1}`) are caught. The clean result on `po_number: null` above is
therefore not evidence the refactor works; it is evidence the validator stopped
looking, which is the same hole Lesson 13.02 finds on its own union-typed field.
Strict mode is a constraint on *schemas*, and every validator here checks
documents.

Structure: `LOOSE` uses the forbidden construct, `strict_form` performs the
refactor, and `errors` runs a document through the lesson's own validator
against either.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "04-structured-output"
LOOSE = {
    "type": "object",
    "properties": {"customer": {"type": "string", "minLength": 1},
                   "po_number": {"type": "string"}},
    "required": ["customer"],           # po_number declared, not required
    "additionalProperties": False,
}
WITHOUT = {"customer": "Acme"}
WITH_NULL = {"customer": "Acme", "po_number": None}
WITH_VALUE = {"customer": "Acme", "po_number": "PO-42"}
NONSENSE = ({"customer": "Acme", "po_number": 42},
            {"customer": "Acme", "po_number": []},
            {"customer": "Acme", "po_number": {"a": 1}})


def strict_form(schema):
    """Every declared property required; the optional ones widened to allow null."""
    optional = [key for key in schema["properties"] if key not in schema["required"]]
    properties = {}
    for key, sub in schema["properties"].items():
        if key in optional:
            declared = sub["type"]
            widened = declared if isinstance(declared, list) else [declared]
            properties[key] = {**sub, "type": sorted({*widened, "null"})}
        else:
            properties[key] = dict(sub)
    return {**schema, "properties": properties,
            "required": sorted(schema["properties"])}, optional


def errors(ref, schema, payload):
    return [str(error) for error in ref.validate(schema, payload)]


def all_required(schema):
    return sorted(schema["properties"]) == sorted(schema["required"])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    strict, optional = strict_form(LOOSE)
    invoice = ref.INVOICE_SCHEMA
    item = invoice["properties"]["line_items"]["items"]
    return {
        "optional": optional,
        "loose_required": LOOSE["required"], "strict_required": strict["required"],
        "loose_is_strict": all_required(LOOSE), "strict_is_strict": all_required(strict),
        "widened_type": strict["properties"]["po_number"]["type"],
        "loose_without": errors(ref, LOOSE, WITHOUT),
        "strict_without": errors(ref, strict, WITHOUT),
        "loose_with_null": errors(ref, LOOSE, WITH_NULL),
        "strict_with_null": errors(ref, strict, WITH_NULL),
        "strict_with_value": errors(ref, strict, WITH_VALUE),
        "strict_nonsense": [errors(ref, strict, payload) for payload in NONSENSE],
        "nonsense_caught": sum(bool(errors(ref, strict, payload)) for payload in NONSENSE),
        "nonsense_tried": len(NONSENSE),
        "invoice_props": len(invoice["properties"]),
        "invoice_all_required": all_required(invoice),
        "item_props": len(item["properties"]), "item_all_required": all_required(item),
        "invoice_closed": invoice["additionalProperties"] is False,
        "item_closed": item["additionalProperties"] is False,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the optional property -- declared but absent from required",
            all([result["optional"] == ["po_number"],
                 result["loose_required"] == ["customer"],
                 result["strict_required"] == ["customer", "po_number"],
                 not result["loose_is_strict"], result["strict_is_strict"],
                 result["widened_type"] == ["null", "string"]]),
            f"plain JSON Schema treats absence as permitted; strict mode requires every "
            f"declared property to be listed, because a grammar that may or may not emit a "
            f"key cannot be built from the schema alone. The refactor moves "
            f"{result['optional']} into required and widens its type to "
            f"{result['widened_type']}, so absent becomes present-and-null",
        ),
        practice.Check(
            "FINDING: the two forms accept different documents, and that is the point",
            all([result["loose_without"] == [],
                 result["strict_without"] == ["$.po_number: missing required field"],
                 len(result["loose_with_null"]) == 1,
                 result["strict_with_null"] == [], result["strict_with_value"] == []]),
            f"against a document with no po_number the loose schema reports "
            f"{len(result['loose_without'])} errors and the strict one "
            f"{len(result['strict_without'])}, {result['strict_without']}. Against one where "
            f"po_number is null the loose schema reports {result['loose_with_null']} and the "
            f"strict one {len(result['strict_with_null'])}. The refactor does not preserve "
            "the accepted set; it moves optionality out of the shape and into the values, "
            "which is what makes it expressible as a grammar",
        ),
        practice.Check(
            "FINDING: the lesson's own Invoice schema is already strict-compatible here",
            all([result["invoice_props"] == 4, result["invoice_all_required"],
                 result["item_props"] == 3, result["item_all_required"],
                 result["invoice_closed"], result["item_closed"]]),
            f"all {result['invoice_props']} top-level properties are in required and all "
            f"{result['item_props']} line-item properties are, with additionalProperties "
            "False at both levels. Whatever else the schema does, it does not use the "
            "forbidden construct -- which is why the exercise has to supply its own example "
            "and says 'non-essentially'",
        ),
        practice.Check(
            "FINDING: the refactor is correct as a schema and unenforceable by this validator",
            all([result["nonsense_caught"] == 0, result["nonsense_tried"] == 3,
                 result["strict_nonsense"] == [[], [], []],
                 result["widened_type"] == ["null", "string"]]),
            f"widening po_number to {result['widened_type']} makes it a *list* of types, and "
            f"validate dispatches with == 'string' and == 'number', so a list matches no "
            f"branch and falls through. {result['nonsense_caught']} of "
            f"{result['nonsense_tried']} nonsense values -- 42, [], {{'a': 1}} -- are caught: "
            f"{result['strict_nonsense']}. So strict_with_null passing above is not evidence "
            "the refactor works, it is evidence this validator stopped looking -- the same "
            "hole Lesson 13.02 finds on its own union-typed field",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
