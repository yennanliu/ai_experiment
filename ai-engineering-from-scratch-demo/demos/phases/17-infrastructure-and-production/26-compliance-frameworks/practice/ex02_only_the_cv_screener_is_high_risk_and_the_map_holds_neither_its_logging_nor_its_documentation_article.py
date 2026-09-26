"""Exercise 2 — only the CV screener is high-risk, and the map holds neither its logging nor its documentation article.

    Classify three hypothetical LLM products under EU AI Act risk tiers. What
    changes at high-risk?

Reading of the exercise: the three products are a customer-support chatbot, a
CV-screening ranker sold to recruiters, and an internal code-review assistant.
They are classified with the lesson's own rule: high-risk "kicks in for
employment, credit, education, law enforcement, migration, essential
services", limited-risk means transparency, and anything else is minimal.
"What changes" is measured two ways: against the lesson's CONTROL_MAP, and
against the fine that `main()` prints.

**ANSWER: chatbot limited, CV screener high, code reviewer minimal.** The
chatbot talks to people, so it owes the Art. 50 disclosure that they are
talking to an AI. The screener is employment (Annex III point 4). The
reviewer is internal tooling with no listed domain. At high risk the
obligations go from one disclosure to the Chapter III Section 2 set
(Art. 9-15: risk management, data governance, technical documentation,
record-keeping, transparency to deployers, human oversight, accuracy and
robustness), plus Art. 43 conformity assessment. In CONTROL_MAP the EU AI Act
rows go from 0 for both lower tiers to 3.

**FINDING: the map holds neither article the lesson names for high-risk.**
The lesson's high-risk line is "conformity assessment, documentation,
logging". The map has Art. 43, but no row cites Art. 11 (documentation) or
Art. 12 (record-keeping), and of Art. 9-15 it cites only Art. 10.

**FINDING: `main()` prints the prohibited-practice fine next to the high-risk
date.** Its closing note is "high-risk enforcement August 2, 2026" followed by
"Fines up to €35M or 7%". That is the Art. 99(3) ceiling for prohibited
practices; high-risk breaches are Art. 99(4), €15M or 3%. The lesson's
"whichever higher applies" is also only half the rule. Art. 99(6) makes it
whichever is *lower* for SMEs, so a screener vendor with €40M turnover and
under 250 staff faces up to €1.2M, not €15M.

**FINDING: both dates the lesson drills have moved.** Two secondary sources
(Gibson Dunn, 27 May 2026; Usercentrics) report that the Digital Omnibus on AI
entered into force on 27 July 2026 and defers Annex III high-risk obligations
from 2 August 2026 to 2 December 2027, and Annex I products to 2 August 2028.
I did not check this against the Official Journal. Colorado SB 26-189, signed
14 May 2026, moves that law to 1 January 2027 and drops the impact assessment
that CONTROL_MAP cites SB24-205 for (McDermott). The screener is still
high-risk; its deadline is December 2027.

Structure: `tier()` is the lesson's rule over a product's domain and
audience; `obligations()` reads CONTROL_MAP per tier.
"""

from __future__ import annotations

import contextlib
import io
import re

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "26-compliance-frameworks"
DOMAINS = (
    "employment, credit, education, law enforcement, migration, essential services"
)
PRODUCTS = {
    "support chatbot": {"domain": "customer support", "talks_to_people": True},
    "CV screener": {"domain": "employment", "talks_to_people": False},
    "code reviewer": {"domain": "internal tooling", "talks_to_people": False},
}
SECTION_2 = set(range(9, 16))
LIMITED = {50}
HIGH = SECTION_2 | {26, 27, 43, 49, 72}  # Ch. III obligations incl. deployers


def tier(product):
    if product["domain"] in DOMAINS.split(", "):
        return "high"
    return "limited" if product["talks_to_people"] else "minimal"


def cited(cites):
    return {int(a) for x in cites for a in re.findall(r"EU AI Act Art\. (\d+)", x)}


def obligations(control_map, level):
    """CONTROL_MAP rows citing an EU AI Act article that binds this tier."""
    owed = {"minimal": set(), "limited": LIMITED, "high": LIMITED | HIGH}[level]
    return [c for c, cites in control_map.items() if cited(cites) & owed]


def fine(turnover, sme, cap=15_000_000, share=0.03):
    pick = min if sme else max
    return pick(cap, share * turnover)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    doc = parity.doc_text(PHASE, LESSON)
    articles = cited(x for v in ref.CONTROL_MAP.values() for x in v)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ref.main()
    tiers = {name: tier(p) for name, p in PRODUCTS.items()}
    return {
        "tiers": tiers,
        "rows": {t: len(obligations(ref.CONTROL_MAP, t)) for t in tiers.values()},
        "domains_in_doc": DOMAINS + "." in doc,
        "named": "conformity assessment, documentation, logging" in doc,
        "articles": sorted(articles),
        "section2": sorted(articles & SECTION_2),
        "note": out.getvalue().strip().splitlines()[-2:],
        "whichever": "whichever higher applies" in doc,
        "fines": (fine(40e6, sme=False), fine(40e6, sme=True)),
        "dates": ("August 2, 2026: high-risk" in doc, "June 30, 2026" in doc),
        "colorado": ref.CONTROL_MAP["impact assessment"][0],
    }


def verify(result):
    arts, note = result["articles"], result["note"]
    expected = {
        "support chatbot": "limited",
        "CV screener": "high",
        "code reviewer": "minimal",
    }
    return [
        practice.Check(
            "ANSWER: chatbot limited, CV screener high, code reviewer minimal",
            (result["tiers"], result["rows"], result["domains_in_doc"])
            == (expected, {"limited": 0, "high": 3, "minimal": 0}, True),
            f"tiers {result['tiers']}; EU AI Act rows in CONTROL_MAP per tier "
            f"{result['rows']}",
        ),
        practice.Check(
            "FINDING: the map holds neither article the lesson names for high-risk",
            (result["named"], 43 in arts, {11, 12} & set(arts), result["section2"])
            == (True, True, set(), [10]),
            f"map cites Art. {arts}; of Art. 9-15 only {result['section2']}; "
            "no Art. 11 documentation or Art. 12 record-keeping",
        ),
        practice.Check(
            "FINDING: main() prints the prohibited-practice fine next to the high-risk date",
            ("high-risk" in note[0], "€35M or 7%" in note[1], result["whichever"])
            == (True, True, True)
            and result["fines"] == (15_000_000, 1_200_000),
            f"closing note {note}; Art. 99(4) is €15M / 3%, and for an SME at "
            f"€40M turnover Art. 99(6) takes the lower: €{result['fines'][1]:,.0f} "
            f"against €{result['fines'][0]:,.0f}",
        ),
        practice.Check(
            "FINDING: both dates the lesson drills have moved",
            result["dates"] == (True, True)
            and result["colorado"] == "Colorado AI Act SB24-205",
            "the lesson teaches August 2, 2026 and June 30, 2026; per secondary sources "
            "the Digital Omnibus moves Annex III to 2 December 2027, and Colorado SB 26-189 "
            f"moves to 1 January 2027 and drops the impact assessment the map cites to "
            f"{result['colorado']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
