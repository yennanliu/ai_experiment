"""Exercise 5 — a naive cross-check escalates five million fields a day.

    Design a regulatory-safe hybrid: OCR pipeline as primary, VLM as secondary
    cross-check. How do you resolve disagreement?

Reading of the exercise: the resolution rule is designed first and then priced at
the volume Exercise 1 names, because the obvious policy -- escalate every
disagreement -- turns out to be unaffordable by three orders of magnitude, and
that is what forces the design. The field list is the lesson's own
`donut_schema("invoice")`, and the error rates are stated assumptions.

**ANSWER: agree-and-accept, disagree-and-escalate, and never resolve by
confidence.** The two systems' confidences are not comparable -- an OCR
character posterior and a VLM token log-probability are different quantities on
different scales -- so a policy that picks the more confident one is a coin flip
wearing a number. Disagreement is a *signal to stop*, not an input to a decision.

**FINDING: the cross-check cuts undetected error 200x.** At a stated 2% OCR and
5% VLM per-field error, independent, with 10 plausible wrong values: primary-only
leaves **2.000%** of fields silently wrong, and agree-and-accept leaves
**0.010%** -- the case where both err *and* land on the same wrong value.

**FINDING: and it escalates 6.9% of fields, which is 4.8 million a day.** The
disagreement rate is `p1(1-p2) + p2(1-p1)` plus the both-wrong-and-differing
case = **6.9%**, and 10M invoices at **7** extractable fields each is 70M fields
-- **4,823,000** escalations a day. A reviewer at 10 seconds a field is
**13,397** hours of work per day.

**ANSWER: so the design is a field-weighted gate.** Escalate only the fields
whose value carries the risk -- `total` and `invoice_number`, **2** of the 7 --
and accept the primary silently on the rest. That is **1,378,000** escalations,
still large, so the second gate is an amount threshold: cross-check every field
on invoices above a limit and only the two key fields below it. The policy is not
"how do we resolve disagreement" but "which disagreements can we afford to see".

Structure: `schema_fields` reads the lesson's own Donut schema, `rates` is the
stated error model, `disagreement` and `silent_error` are its two outcomes, and
`escalations` prices a policy at Exercise 1's volume.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "22-document-diagram-understanding"
OCR_ERROR, VLM_ERROR = 0.02, 0.05
PLAUSIBLE_WRONG = 10
PAGES_PER_DAY = 10_000_000
CRITICAL = ("total", "invoice_number")
SECONDS_PER_REVIEW = 10


def schema_fields(ref, task="invoice"):
    """Leaf fields of the lesson's own Donut schema, one line item assumed."""
    schema = ref.donut_schema(task)
    fields = []
    for name, value in schema.items():
        if isinstance(value, list):
            fields.extend(f"{name}.{key}" for key in value[0])
        else:
            fields.append(name)
    return fields


def disagreement(p1=OCR_ERROR, p2=VLM_ERROR, plausible=PLAUSIBLE_WRONG):
    """One wrong, or both wrong on different values."""
    both_wrong = p1 * p2
    return p1 * (1 - p2) + p2 * (1 - p1) + both_wrong * (1 - 1 / plausible)


def silent_error(p1=OCR_ERROR, p2=VLM_ERROR, plausible=PLAUSIBLE_WRONG):
    """Both wrong and agreeing -- the only case the cross-check cannot see."""
    return p1 * p2 / plausible


def escalations(rate, fields, pages=PAGES_PER_DAY):
    return round(rate * fields * pages)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    fields = schema_fields(ref)
    rate = disagreement()
    critical = [name for name in fields if name in CRITICAL]
    all_fields = escalations(rate, len(fields))
    return {
        "fields": fields, "count": len(fields),
        "disagreement_pct": round(rate * 100, 1),
        "silent_pct": round(silent_error() * 100, 3),
        "primary_only_pct": round(OCR_ERROR * 100, 3),
        "improvement": round(OCR_ERROR / silent_error()),
        "escalations": all_fields,
        "review_hours": round(all_fields * SECONDS_PER_REVIEW / 3600),
        "critical": critical, "critical_count": len(critical),
        "gated": escalations(rate, len(critical)),
        "gated_reduction": round((1 - len(critical) / len(fields)) * 100),
        "confidences_comparable": False,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: agree-and-accept, disagree-and-escalate, never resolve by confidence",
            all([result["count"] == 7, not result["confidences_comparable"],
                 "total" in result["fields"]]),
            f"the lesson's own invoice schema has {result['count']} leaf fields "
            f"({result['fields']}). An OCR character posterior and a VLM token "
            "log-probability are different quantities on different scales, so a policy that "
            "picks the more confident one is a coin flip wearing a number -- disagreement is "
            "a signal to stop, not an input to a decision",
        ),
        practice.Check(
            "FINDING: the cross-check cuts undetected error 200x",
            all([result["primary_only_pct"] == 2.0, result["silent_pct"] == 0.01,
                 result["improvement"] == 200]),
            f"at a stated {OCR_ERROR:.0%} OCR and {VLM_ERROR:.0%} VLM per-field error, "
            f"independent, with {PLAUSIBLE_WRONG} plausible wrong values: primary-only leaves "
            f"{result['primary_only_pct']}% of fields silently wrong and agree-and-accept "
            f"leaves {result['silent_pct']}% -- {result['improvement']}x better, and the "
            "residue is exactly the case where both err and agree",
        ),
        practice.Check(
            "FINDING: and it escalates 6.9% of fields, 4.8 million a day",
            all([result["disagreement_pct"] == 6.9,
                 result["escalations"] == 4_823_000,
                 result["review_hours"] == 13_397]),
            f"the disagreement rate is p1(1-p2) + p2(1-p1) plus the both-wrong-and-differing "
            f"case = {result['disagreement_pct']}%, and {PAGES_PER_DAY:,} invoices at "
            f"{result['count']} fields each is {result['count'] * PAGES_PER_DAY / 1e6:.0f}M "
            f"fields -- {result['escalations']:,} escalations a day, "
            f"{result['review_hours']:,} reviewer-hours at {SECONDS_PER_REVIEW} seconds each",
        ),
        practice.Check(
            "ANSWER: so the design is a field-weighted gate",
            all([result["critical_count"] == 2, result["gated"] == 1_378_000,
                 result["gated_reduction"] == 71]),
            f"escalating only {result['critical']} -- {result['critical_count']} of "
            f"{result['count']} fields -- is {result['gated']:,} a day, "
            f"{result['gated_reduction']}% fewer. Still large, so the second gate is an "
            "amount threshold: cross-check every field above a limit and only the key two "
            "below it. The question is not how to resolve disagreement but which "
            "disagreements can be afforded",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
