"""Exercise 4 — the lesson's "interest + opt-out = lawful" rule clears health and minors' posts that need consent.

    The 2025 DPA alignment accepts legitimate interest for public-content
    training. Construct a scenario in which legitimate interest would not
    suffice and identify the legal basis a provider would need instead.

Reading of the exercise: the lesson condenses the 2025 decisions into one
rule, printed by `code/main.py` as "legitimate interest + opt-out = lawful"
and stated in docs/en.md as "Consent is not required". That rule is encoded
as written and run against five abstract scenarios. It is compared with a
GDPR reading that adds the limits the 2025 decisions did not lift. Article
6(1)(f) interest is weighed against the data subject "in particular where the
data subject is a child". Article 9 forbids processing special-category data
(health, religion, sexuality ...) whatever the Article 6 basis, unless an
Article 9(2) exception applies. Non-public content falls outside what the
Irish DPC accepted.

**ANSWER: scenario B. Public posts in a patient-support forum reveal each
author's diagnosis. Legitimate interest does not suffice, and the provider
needs explicit consent under Article 9(2)(a) on top of an Article 6 basis.**
Scenario D (public posts by users under 18) needs consent too, given by the
parent for information-society services (Article 8). Of the five scenarios,
the lesson's rule and the GDPR reading disagree on 2, B and D. Both times
the lesson's rule says lawful. The rule reads only "public" and "opt-out",
and those two inputs cannot see a child or a diagnosis.

**FINDING: the reference's disclosure cannot tell scenario B from scenario
A.** Filled for the harmless adult posts (A) and the diagnosis posts (B),
the two summaries differ only in item 4's description. Both answer item 7
(personal information) "Y", and `flag_followups` returns the same two lines
for both, CPRA and TDM opt-out. AB 2013 item 7
is binary, and CPRA's separate "sensitive personal information" category
has no field.

**FINDING: the lesson dates the Brazilian suspension two ways.** Its
heading says "Brazilian ANPD (June 2024)"; its opening summary says "2 July
2024".

Structure: `lesson_rule()` and `gdpr_basis()` are the two readings;
`summary()` fills the reference's 12 fields for a scenario.
"""

from __future__ import annotations

import contextlib
import io

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "27-data-provenance-training-governance"
# (label, public, first_party, opt_out, special_category, minors)
SCENARIOS = {
    "A adult public posts, opt-out offered": (True, True, True, False, False),
    "B patient-forum posts naming a diagnosis": (True, True, True, True, False),
    "C private messages": (False, True, True, False, False),
    "D public posts by users under 18": (True, True, True, False, True),
    "E adult public posts, no opt-out": (True, True, False, False, False),
}


def lesson_rule(public, first_party, opt_out, special, minors):
    """'legitimate interest + opt-out = lawful' for public content, as the lesson states it."""
    return public and opt_out


def gdpr_basis(public, first_party, opt_out, special, minors):
    """The basis the processing needs; 'legitimate interest' means Art 6(1)(f) suffices."""
    if special:
        return "explicit consent, Art 9(2)(a)"
    if minors:
        return "parental consent, Art 8"
    if not public:
        return "consent, Art 6(1)(a)"
    return "legitimate interest" if opt_out and first_party else "consent, Art 6(1)(a)"


def summary(ref, scenario):
    special = SCENARIOS[scenario][3]
    values = ["first-party public posts on [PLATFORM-A]", "general assistant pretraining",
              "about 10M posts", "free text; unlabeled", "Y (user-authored posts)", "N", "Y", "N",
              "deduplicated to reduce memorisation", "2024-01 to 2024-12; not ongoing", "2025-02", "N"]
    values[3] += "; health disclosures" if special else ""
    return dict(zip(ref.AB_2013_FIELDS, values))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ref.main()
    doc = parity.doc_text(PHASE, LESSON)
    table = {s: (lesson_rule(*v), gdpr_basis(*v)) for s, v in SCENARIOS.items()}
    a, b = (summary(ref, s) for s in list(SCENARIOS)[:2])
    return {
        "rule_printed": "legitimate interest + opt-out = lawful" in out.getvalue(),
        "no_consent": "Consent is not required" in doc,
        "table": table,
        "disagree": [s[0] for s, (ok, basis) in table.items() if ok != (basis == "legitimate interest")],
        "disagree_lawful": all(ok for s, (ok, basis) in table.items() if ok != (basis == "legitimate interest")),
        "flags": (ref.flag_followups(a), ref.flag_followups(b)),
        "summaries_differ": [i for i, f in enumerate(ref.AB_2013_FIELDS, 1) if a[f] != b[f]],
        "anpd": ("Brazilian ANPD (June 2024)" in doc, "Brazilian ANPD (2 July 2024)" in doc),
    }


def verify(result):
    table, (fa, fb) = result["table"], result["flags"]
    b = table["B patient-forum posts naming a diagnosis"]
    return [
        practice.Check(
            "ANSWER: scenario B needs explicit consent under Art 9(2)(a); the rules disagree on B and D",
            all([result["rule_printed"], result["no_consent"], result["disagree"] == ["B", "D"],
                 b == (True, "explicit consent, Art 9(2)(a)"), result["disagree_lawful"]]),
            f"(lesson rule says lawful, basis needed) per scenario: {table}",
        ),
        practice.Check(
            "FINDING: the reference's disclosure cannot tell scenario B from scenario A",
            all([fa == fb, len(fa) == 2, "CPRA" in fa[0], result["summaries_differ"] == [4]]),
            f"summaries differ only in item(s) {result['summaries_differ']}; flags for both: {fa}",
        ),
        practice.Check(
            "FINDING: the lesson dates the Brazilian suspension two ways",
            result["anpd"] == (True, True),
            "docs/en.md has both 'Brazilian ANPD (June 2024)' and 'Brazilian ANPD (2 July 2024)'",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
