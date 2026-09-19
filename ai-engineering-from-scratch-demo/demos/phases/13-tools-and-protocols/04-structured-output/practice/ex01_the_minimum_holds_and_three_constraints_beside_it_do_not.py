"""Exercise 1 — the minimum holds, and three constraints beside it do not.

    Run `code/main.py`. Add a fourth test case whose `total_usd` is a negative
    number. Confirm the validator rejects it with the `minimum` constraint path.

Reading of the exercise: the case is added and the rejection confirmed, and
since confirming it means reading how `minimum` reaches a number at all, the
sibling constraints on the same schema are put through the same test. The
`minimum` branch is sound. Three of the things around it are not, and an invoice
is a good object to notice that on, because it has an invariant that no JSON
Schema can carry.

**ANSWER: it is rejected, at `$.total_usd`, with `below minimum 0`.**
`total_usd: -5.0` produces exactly one error and the path names the field. The
`number` branch returns early only on a type mismatch, so the shared `minimum`
check below it is reached for every well-typed number.

**FINDING: `pattern` lets a SKU end in a newline.** `re.match` with `^[A-Z0-9-]+$`
accepts `"ABC-1\\n"`, because Python's `$` matches before a trailing newline.
The validator reports **no errors** on a line item whose SKU carries one. It is
`re.fullmatch` that means what the anchors look like they mean.

**FINDING: `line_items` has no `minItems`, so an invoice with nothing on it is
valid.** `{"line_items": [], "total_usd": 0.0}` passes clean. The lesson's own
parse-error fixture is built on that shape, so the emptiness is deliberate
somewhere and unconstrained here.

**FINDING: and `total_usd` is never checked against the line items.**
`total_usd: 999.0` against a single 10.00 item passes with **no errors**. The
one invariant that makes an invoice an invoice -- that the total is the sum --
is not expressible in JSON Schema at all, so strict mode can guarantee this
document's shape and never its arithmetic.

Structure: `errors` runs one payload through the lesson's own `validate`,
`invoice` builds a well-formed one to perturb, and `restated` recomputes the
total from the line items the schema cannot relate them to.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "04-structured-output"
ITEM = {"sku": "ABC-1", "qty": 1, "unit_usd": 10.0}


def invoice(**overrides):
    base = {"customer": "Acme", "line_items": [dict(ITEM)],
            "total_usd": 10.0, "currency": "USD"}
    base.update(overrides)
    return base


def errors(ref, payload):
    return [str(error) for error in ref.validate(ref.INVOICE_SCHEMA, payload)]


def restated(payload):
    """The total the line items imply, which the schema has no way to demand."""
    return round(sum(item["qty"] * item["unit_usd"] for item in payload["line_items"]), 2)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    negative = errors(ref, invoice(total_usd=-5.0))
    newline_sku = errors(ref, invoice(line_items=[{**ITEM, "sku": "ABC-1\n"}]))
    empty = errors(ref, invoice(line_items=[], total_usd=0.0))
    wrong_total = invoice(total_usd=999.0)
    return {
        "negative": negative, "negative_count": len(negative),
        "negative_path": negative[0].split(":")[0] if negative else None,
        "newline_sku": newline_sku,
        "sku_pattern": ref.INVOICE_SCHEMA["properties"]["line_items"]["items"]
                          ["properties"]["sku"]["pattern"],
        "empty_items": empty,
        "has_min_items": "minItems" in ref.INVOICE_SCHEMA["properties"]["line_items"],
        "wrong_total_errors": errors(ref, wrong_total),
        "stated_total": wrong_total["total_usd"], "implied_total": restated(wrong_total),
        "totals_disagree": wrong_total["total_usd"] != restated(wrong_total),
        "schema_keywords": sorted({key for prop in ref.INVOICE_SCHEMA["properties"].values()
                                   for key in prop if key != "type"}),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: rejected at $.total_usd with 'below minimum 0'",
            all([result["negative"] == ["$.total_usd: below minimum 0"],
                 result["negative_count"] == 1,
                 result["negative_path"] == "$.total_usd"]),
            f"total_usd: -5.0 produces exactly {result['negative_count']} error, "
            f"{result['negative']}. The number branch returns early only on a type mismatch, "
            "so the shared minimum check below it is reached for every well-typed number",
        ),
        practice.Check(
            "FINDING: pattern lets a SKU end in a newline",
            all([result["newline_sku"] == [], result["sku_pattern"] == "^[A-Z0-9-]+$"]),
            f"the SKU pattern is {result['sku_pattern']!r} and a line item whose sku is "
            f"'ABC-1\\n' validates with {result['newline_sku']} errors. validate uses "
            "re.match, and Python's $ matches before a trailing newline -- it is re.fullmatch "
            "that means what the anchors look like they mean",
        ),
        practice.Check(
            "FINDING: line_items has no minItems, so an empty invoice is valid",
            all([result["empty_items"] == [], not result["has_min_items"]]),
            f"an invoice with no line items and a zero total passes clean: "
            f"{result['empty_items']}. minItems is absent from the array schema, and the "
            "lesson's own parse-error fixture is built on that shape",
        ),
        practice.Check(
            "FINDING: total_usd is never checked against the line items",
            all([result["wrong_total_errors"] == [], result["stated_total"] == 999.0,
                 result["implied_total"] == 10.0, result["totals_disagree"]]),
            f"a stated total of {result['stated_total']} against line items implying "
            f"{result['implied_total']} passes with {result['wrong_total_errors']} errors. "
            f"The schema's keywords are {result['schema_keywords']} -- all of them describe "
            "one field in isolation. The invariant that makes an invoice an invoice is not "
            "expressible in JSON Schema, so strict mode can guarantee shape and never "
            "arithmetic",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
