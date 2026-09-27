"""Exercise 2 — EU hosting does not trigger the AI Act, so Korea binds 3 of 4 questions and the skill file misroutes 18 of 64 placements.

    A deployment is made by a US company, runs on EU infrastructure, and
    serves Korean users. Which three jurisdictions' rules apply, and which
    rule binds on each substantive question?

Reading of the exercise: the company deploys its own AI system; it has no EU
establishment and no EU users, and it does not place a model on the EU
market. Each jurisdiction's scope rule is coded from its text: EU AI Act
Art 2(1) (placing on the Union market, a deployer established in the Union,
or output used in the Union), Korean Framework Act Art 4(1) (acts abroad that
affect the Korean market or users), and the US, which has no federal AI
statute; CAISI is voluntary and state laws key on residents. The questions
are the skill file's three (transparency, risk assessment, copyright) plus
the local representative. The lesson's rule picks the strictest regime that
applies.

**ANSWER: only one of the three reaches it with an AI statute.** Korea's Act
applies (Korean users). The EU AI Act does not: hosting is not a scope
trigger in Art 2(1). The US contributes only voluntary CAISI standards.
Binding rules: transparency, Korea Art 31 (notify users, label generative
output); risk assessment, Korea Art 34(1) risk-management plan if high-impact
and Art 32 above the compute threshold; local representative, Korea Art 36;
copyright, no AI statute. The Korean Act never mentions copyright, and EU
Art 53(1)(c) does not reach this deployment.

**FINDING: the skill file routes every question to the EU.** Its trigger is
"touches EU users or infrastructure". For this deployment that binds the EU
rule on 4 of 4 questions, where Art 2(1) binds it on 0. Over all 64
(provider, infrastructure, users) placements across EU / US / UK / KR, the
two scope tests disagree on 18: 9 where only the servers are in the EU, and 9
where an EU company serves no EU users. The skill trigger misses those 9.

**FINDING: the lesson's penalty cap is the middle tier, and its date is a year
late.** The page and main.py put "penalties up to 15M EUR / 3%" at
2 Aug 2026. That is Art 99(4). Art 99(3) fines prohibited practices up to 35M
EUR or 7% of turnover, 2.33x the cap the lesson quotes. Chapter XII
(penalties) has applied since 2 Aug 2025 under Art 113(b), 365 days before
the date the lesson gives. Only Art 101's GPAI fines wait for 2026.
"Strictest" is also not the EU here: Korea's only fine is at most 30M won
(Art 43), and it is the only regime that applies.

Structure: `SCOPE` holds each regime's scope test and `SKILL_EU` the skill
file's; `binding()` picks the rule per question; `lesson_claims()` reads the
page, the skill file and TIMELINE.
"""

from __future__ import annotations

import datetime
import itertools
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "24-regulatory-frameworks-eu-us-uk-korea"
PLACES = ("EU", "US", "UK", "KR")
DEPLOYMENT = {"provider": "US", "infra": "EU", "users": "KR"}
SCOPE = {  # binding AI statutes only; the US has none federally and no US users here
    "EU": lambda d: "EU" in (d["provider"], d["users"]),       # AI Act Art 2(1)(a)-(c)
    "KR": lambda d: "KR" in (d["provider"], d["users"]),       # Framework Act Art 4(1)
    "US": lambda d: d["users"] == "US",                        # state laws key on residents
}
RULES = {  # question -> {regime: article}; strictest first, as the lesson orders them
    "transparency": {"EU": "AI Act Art 50", "KR": "Framework Act Art 31"},
    "risk assessment": {"EU": "AI Act Art 9 / Art 55", "KR": "Framework Act Art 34(1), Art 32"},
    "copyright": {"EU": "AI Act Art 53(1)(c)"},                # Korean Act: 0 mentions of copyright
    "local representative": {"EU": "AI Act Art 22 / Art 54", "KR": "Framework Act Art 36"},
}
EU_FINES = [(35e6, 0.07, "Art 99(3)"), (15e6, 0.03, "Art 99(4)"), (7.5e6, 0.01, "Art 99(5)")]
KR_FINE_WON, PENALTIES_APPLY = 30e6, datetime.date(2025, 8, 2)  # Art 43(1); Art 113(b)


def binding(applies):
    return {q: next((rules[j] for j in rules if applies.get(j)), None) for q, rules in RULES.items()}


def lesson_claims(ref):
    doc, skill = parity.doc_text(PHASE, LESSON), (
        parity.lesson_dir(PHASE, LESSON) / "outputs" / "skill-regulatory-map.md").read_text()
    date = next(d for d, e in ref.TIMELINE if "penalties" in e)
    cap = re.search(r"penalties up to (\d+)M EUR / (\d+)%", doc)
    return {"triggers": re.search(r"touches EU (\w+) or (\w+)", skill).groups(),
            "questions": re.search(r"\(([^)]*risk assessment[^)]*)\)", skill).group(1).split(", "),
            "strictest": "strictest, which in 2026 is typically the EU AI Act" in doc,
            "cap": (float(cap.group(1)) * 1e6, int(cap.group(2)) / 100),
            "date": datetime.date.fromisoformat(date)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    claims = lesson_claims(ref)
    fields = [{"users": "users", "infrastructure": "infra"}[t] for t in claims["triggers"]]

    def skill_eu(d):
        return any(d[f] == "EU" for f in fields)

    grid = [dict(zip(("provider", "infra", "users"), p)) for p in itertools.product(PLACES, repeat=3)]
    split = [d for d in grid if skill_eu(d) != SCOPE["EU"](d)]
    applies = {j: f(DEPLOYMENT) for j, f in SCOPE.items()}
    top = max(EU_FINES)
    return {"applies": applies, "binding": binding(applies),
            "skill_binding": binding({**applies, "EU": skill_eu(DEPLOYMENT)}),
            "disagree": len(split), "grid": len(grid),
            "infra_only": sum(d["infra"] == "EU" for d in split),
            "claims": claims, "top": top, "ratio": round(top[1] / claims["cap"][1], 2),
            "days_late": (claims["date"] - PENALTIES_APPLY).days}


def verify(result):
    r, c = result, result["claims"]
    kr = [q for q, rule in r["binding"].items() if "Framework" in str(rule)]
    skill_eu = [q for q, rule in r["skill_binding"].items() if "AI Act" in str(rule)]
    return [
        practice.Check(
            "ANSWER: only Korea reaches it with an AI statute; copyright binds under none",
            (r["applies"], r["binding"]["copyright"], kr) == ({"EU": False, "KR": True, "US": False}, None,
                                                              ["transparency", "risk assessment",
                                                               "local representative"]),
            f"applies {r['applies']}; binding rule per question {r['binding']}",
        ),
        practice.Check(
            "FINDING: the skill file routes every question to the EU",
            (c["triggers"], len(skill_eu), c["strictest"], c["questions"], r["disagree"], r["grid"],
             r["infra_only"]) == (("users", "infrastructure"), 4, True,
                                  ["transparency", "risk assessment", "copyright"], 18, 64, 9),
            f"skill trigger EU {c['triggers']} binds the EU on {skill_eu}; scope tests "
            f"disagree on {r['disagree']} of {r['grid']} placements, {r['infra_only']} of them "
            "EU infrastructure only",
        ),
        practice.Check(
            "FINDING: the lesson's penalty cap is the middle tier, and its date is a year late",
            (c["cap"], r["ratio"], r["days_late"]) == (EU_FINES[1][:2], 2.33, 365),
            f"lesson {c['cap']} at {c['date']} vs top tier {r['top']}, {r['ratio']}x; Chapter XII "
            f"applies from {PENALTIES_APPLY}, {r['days_late']} days earlier; Korea max "
            f"{KR_FINE_WON:,.0f} won",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
