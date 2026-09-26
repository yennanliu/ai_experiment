"""Exercise 4 — not by the lesson's own map, which requires it for 2 of 7 profiles and credits it with 0 of 10 controls.

    Argue whether ISO 42001 is "necessary in 2026" for a mid-market AI vendor.

Reading of the exercise: "necessary" is read as "a deal is lost without it",
so the argument is built from the evidence the lesson ships: PROFILE_MAP,
which says when a framework is required, and CONTROL_MAP, which says what a
framework's controls are reused for. A mid-market vendor has no profile of its
own, so each profile is read as a place such a vendor might sell.

**ANSWER: no, not as a certificate; necessary only where a buyer's
questionnaire names it.** The lesson's own profiles require ISO 42001 in 2 of
7 cases, ("US", "B2B SaaS") and ("Global", "enterprise"). It is absent from both
healthcare profiles, the fintech profile, the Colorado profile and both EU
profiles; of the 3 profiles that carry the EU AI Act, only the global one
lists it. A mid-market vendor selling
into US B2B SaaS meets it on the map. One selling to EU customers does not.
The honest 2026 answer is to hold ISO 27001 (5 of 10 map rows), document the
AI-specific practices 42001 asks for, and certify when a contract requires it.

**FINDING: the map gives ISO 42001 zero reuse.** No CONTROL_MAP row cites it,
while ISO 27001 appears in 5 of 10. The lesson says 42001 "pairs with ISO
27001", but in its own map certifying buys no control credit beyond the
certificate. The "impact assessment" row cites Colorado and EU AI Act Art. 27,
not 42001, although secondary sources list an AI-system impact assessment
among 42001's Annex A objectives (A.2-A.10, 38 controls). I could not open
iso.org to check that list at the source.

**FINDING: the pairing claim is structural, and the map cannot show it.**
ISO 42001 uses the same Harmonized Structure (clauses 4-10) as ISO 27001, per
the same secondary sources, and that is why a 27001 holder's extra effort is
small. All 5 of CONTROL_MAP's ISO 27001 citations are Annex A controls and 0
are clauses, so this, the strongest argument for adopting 42001, has no place
in the map.

**FINDING: the regulatory push for 2026 got weaker.** The EU AI Act's Annex III
high-risk deadline, the one date that could make an AI management system
urgent, moved from 2 August 2026 to 2 December 2027 (see exercise 2).

Structure: `requiring()` and `rows_citing()` read the two dicts.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "26-compliance-frameworks"


def requiring(profile_map, framework):
    return [k for k, v in profile_map.items() if framework in v]


def rows_citing(control_map, framework):
    return [c for c, cites in control_map.items() if any(framework in x for x in cites)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    pm, cm = ref.PROFILE_MAP, ref.CONTROL_MAP
    doc = parity.doc_text(PHASE, LESSON)
    return {
        "requiring": requiring(pm, "ISO 42001"),
        "profiles": len(pm),
        "ai_act": requiring(pm, "EU AI Act"),
        "rows_42001": rows_citing(cm, "ISO 42001"),
        "rows_27001": rows_citing(cm, "ISO 27001"),
        "rows": len(cm),
        "cites_27001": [x for v in cm.values() for x in v if x.startswith("ISO 27001")],
        "impact": cm["impact assessment"],
        "pairs": "pairs with ISO 27001" in doc and "### ISO 42001 — emerging" in doc,
        "mid_market": [k for k in pm if "mid" in k[1].lower()],
    }


def verify(result):
    req, ai_act = result["requiring"], result["ai_act"]
    return [
        practice.Check(
            "ANSWER: no, not as a certificate; necessary only where a buyer's "
            "questionnaire names it",
            (req, result["profiles"], result["mid_market"])
            == ([("US", "B2B SaaS"), ("Global", "enterprise")], 7, []),
            f"ISO 42001 is required by {len(req)} of {result['profiles']} profiles {req}; "
            f"of the {len(ai_act)} profiles with the EU AI Act, "
            f"{sum(p in req for p in ai_act)} require it; no mid-market profile exists",
        ),
        practice.Check(
            "FINDING: the map gives ISO 42001 zero reuse",
            (result["rows_42001"], len(result["rows_27001"]), result["pairs"])
            == ([], 5, True)
            and not any("42001" in x for x in result["impact"]),
            f"rows citing ISO 42001: {len(result['rows_42001'])} of {result['rows']}; "
            f"ISO 27001: {len(result['rows_27001'])}; impact assessment cites "
            f"{result['impact']}",
        ),
        practice.Check(
            "FINDING: the pairing claim is structural, and the map cannot show it",
            [x[:12] for x in result["cites_27001"]] == ["ISO 27001 A."] * 5,
            f"ISO 27001 citations {result['cites_27001']} are all Annex A; the shared "
            "clauses 4-10 that make 42001 cheap for a 27001 holder have no row",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
