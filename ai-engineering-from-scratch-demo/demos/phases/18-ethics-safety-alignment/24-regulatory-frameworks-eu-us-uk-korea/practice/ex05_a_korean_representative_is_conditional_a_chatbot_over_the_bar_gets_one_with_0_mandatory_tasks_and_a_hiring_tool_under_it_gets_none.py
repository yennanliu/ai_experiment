"""Exercise 5 — a Korean representative is conditional: a chatbot over the bar gets one with 0 mandatory tasks, and a hiring tool under it gets none.

    Korea's AI Framework Act requires local representatives for foreign
    providers. Describe the operational implications for a Bay Area company
    serving Korean users.

Reading of the exercise: the implications are derived from the Act's text
(CSET translation of the Framework Act as promulgated 21 January 2025, read
2026-09-27): who must appoint (Art 36(1)), what the representative does
(Art 36(1) 1-3), who is liable (Art 36(3)) and the fine (Art 43(1)). They are
then applied to three Bay Area archetypes. The thresholds are the
Enforcement Decree's as reported at entry into force: KRW 1 trillion total
revenue, KRW 10 billion AI-service revenue, or 1 million average daily Korean
users. Safety duties start at 10^26 FLOP of training compute.

**ANSWER: the obligation is a trigger check, a written appointment reported
to MSIT, and liability kept at home.** A company that crosses any threshold
appoints a representative with a Korean address and reports it to MSIT.
Whatever the representative does wrong under Art 36(1) is held against the
company (Art 36(3)). Failing to appoint costs at most KRW 30 million
(Art 43(1)), 0.3% of the smallest revenue that triggers the duty, after a
grace period of at least a year. The representative's three statutory tasks
are compute-safety reporting (Art 32(2)), a high-impact confirmation request
(Art 33(1), which the company *may* make), and support for high-impact
duties (Art 34(1)). The company's own Art 31 notice and labelling duties are
not on that list.

**FINDING: the representative and the duties it carries are triggered
separately.** A consumer chatbot over the user bar, on a licensed model below
10^26 FLOP, must appoint a representative with 0 of 3 mandatory tasks. A
high-impact hiring tool under every bar owes all Art 34 duties with no
representative. Only the frontier-model API is both caught and has a live
task (1 of 3).

**FINDING: the lesson states two conditional provisions as unconditional.**
The page says the Act "mandates local representatives for foreign AI
companies". Art 36(1) binds only operators that meet thresholds set by
Presidential Decree. The page says "Article 12 establishes an AISI". Art 12
says the Minister "may operate" one. TIMELINE cannot give the effective day:
it has "2026-01-00", where the Act took effect on 2026-01-22, one year after
promulgation, and the grace period runs to at least 2027-01-22.

Structure: `THRESHOLDS` and `TASKS` encode Arts 36 and 32-34; `assess()`
applies them to one archetype; `lesson_claims()` reads the page and TIMELINE.
"""

from __future__ import annotations

import datetime
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "24-regulatory-frameworks-eu-us-uk-korea"
THRESHOLDS = {"total_won": 1e12, "ai_won": 1e10, "kr_daily_users": 1e6}   # Enforcement Decree, any one
COMPUTE_FLOP, FINE_WON = 1e26, 30e6                                       # Art 32 per Decree; Art 43(1)
# Art 36(1): the representative's tasks -> (source duty, mandatory?, who it reaches)
TASKS = {"submit Art 32(2) safety results": ("frontier", True),
         "request Art 33(1) high-impact confirmation": ("any", False),        # "may request"
         "support Art 34(1) high-impact measures": ("high_impact", True)}
ART12, ART36 = "may operate an AI Safety Research Institute", "which meet certain user and revenue thresholds"
PROMULGATED = datetime.date(2025, 1, 21)                                  # Law No. 20676
ARCHETYPES = {  # Bay Area companies serving Korean users: (total KRW, AI KRW, KR daily users, FLOP, high-impact)
    "consumer chatbot on a licensed model": (2e10, 2e10, 1.5e6, 0.0, False),
    "hiring-screening SaaS": (5e9, 5e9, 2e4, 0.0, True),
    "frontier-model API": (2e12, 1.5e12, 3e5, 3e26, False),
}


def assess(total, ai, users, flop, high_impact):
    needs = any(v >= THRESHOLDS[k] for k, v in zip(THRESHOLDS, (total, ai, users)))
    reach = {"frontier": flop >= COMPUTE_FLOP, "high_impact": high_impact, "any": True}
    live = [t for t, (who, mandatory) in TASKS.items() if mandatory and reach[who]]
    return {"representative": needs, "mandatory_tasks": len(live) if needs else 0,
            "art34_duties": high_impact}


def lesson_claims(ref):
    section = parity.doc_text(PHASE, LESSON).split("### Korean AI Framework Act")[1].split("\n### ")[0]
    return {"rep": re.search(r"- (Local representatives for [^.]+)\.", section).group(1),
            "art12": re.search(r"Article 12 (\w+) an AISI", section).group(1),
            "effective": next(d for d, e in ref.TIMELINE if e == "Korean AI Framework Act effective")}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    effective = PROMULGATED.replace(year=PROMULGATED.year + 1) + datetime.timedelta(days=1)
    return {"archetypes": {n: assess(*a) for n, a in ARCHETYPES.items()}, "claims": lesson_claims(ref),
            "effective": effective.isoformat(), "grace_end": effective.replace(year=effective.year + 1).isoformat(),
            "fine_share": round(FINE_WON / THRESHOLDS["ai_won"], 6), "tasks": len(TASKS),
            "optional": [t for t, (_, m) in TASKS.items() if not m]}


def verify(result):
    a, c = result["archetypes"], result["claims"]
    chat, hire, api = (a[n] for n in ARCHETYPES)
    return [
        practice.Check(
            "ANSWER: appoint if any threshold is crossed; the fine is 0.3% of the smallest trigger",
            (result["tasks"], result["fine_share"], result["optional"])
            == (3, 0.003, ["request Art 33(1) high-impact confirmation"]),
            f"{result['tasks']} statutory tasks, optional {result['optional']}; max fine "
            f"{FINE_WON:,.0f} won = {result['fine_share']:.1%} of the {THRESHOLDS['ai_won']:,.0f} won trigger",
        ),
        practice.Check(
            "FINDING: the representative and the duties it carries are triggered separately",
            (chat["representative"], chat["mandatory_tasks"], hire["representative"],
             hire["art34_duties"], api["representative"], api["mandatory_tasks"])
            == (True, 0, False, True, True, 1),
            f"per archetype {a}",
        ),
        practice.Check(
            "FINDING: the lesson states two conditional provisions as unconditional",
            (c["rep"], c["art12"], c["effective"], result["effective"], result["grace_end"])
            == ("Local representatives for foreign AI companies operating in Korea", "establishes",
                "2026-01-00", "2026-01-22", "2027-01-22"),
            f"page {c['rep']!r} vs Art 36(1) {ART36!r}; page 'Article 12 {c['art12']}' vs "
            f"{ART12!r}; TIMELINE {c['effective']} vs {result['effective']}, grace to "
            f"{result['grace_end']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
