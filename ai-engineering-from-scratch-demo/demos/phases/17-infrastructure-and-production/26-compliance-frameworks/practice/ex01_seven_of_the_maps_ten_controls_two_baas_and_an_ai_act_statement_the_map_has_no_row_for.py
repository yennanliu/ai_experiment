"""Exercise 1 — seven of the map's ten controls, two BAAs, and an AI Act statement the map has no row for.

    Your first enterprise customer requires SOC 2 Type II, HIPAA BAA, EU AI Act
    statement. What is the minimum viable compliance posture to win the deal?

Reading of the exercise: the team is the one in the lesson's Problem -- SOC 2
Type I, six months from Type II -- and the product is ordinary B2B LLM SaaS,
which the lesson says is limited-risk. "Minimum viable" is read as: the
smallest set of the lesson's own CONTROL_MAP rows plus deliverables that
answers each of the three asks, with anything that only applies at high risk
removed.

**ANSWER: Type I plus a running Type II window, two BAAs, a limited-risk
statement, and 7 of the map's 10 controls.** The three frameworks touch 9 of
the 10 rows in CONTROL_MAP (everything but data subject rights). Two of those,
conformity assessment (Art. 43) and impact assessment, are high-risk
obligations, which leaves 7. Type II cannot be bought in time: the lesson's own
window is 6-12 months of operated controls, so the deal is won on the Type I
report with the observation window already started and a committed report
date. The HIPAA ask is two contracts, not one: the customer's BAA with you, and
a BAA with every model provider on the PHI path. The AI Act statement is a
written risk-tier classification.

**FINDING: by the map, 2 controls "cover" all three frameworks.** 10 pairs of
rows each touch SOC 2, HIPAA and the EU AI Act, for example access logging plus
impact assessment. A row mapping to a framework means one clause of that
framework, so counting frameworks touched measures nothing about readiness.

**FINDING: no profile is this customer, and the code has no lookup.** None of
the 7 PROFILE_MAP rows is exactly these three. The smallest that contains all
three is ("EU", "healthcare") with 4 frameworks, and it lists "HIPAA
(global)", although HIPAA is a US law. `main()` only prints both dicts; there is
no function that takes a control or a profile, although Use It describes one.
Every profile lists SOC 2 Type II, which this team cannot have for 6 months.

**FINDING: every EU AI Act citation in the map is a high-risk article.** The
map cites Art. 10, 27 and 43. Art. 10 (data governance) and Art. 43
(conformity assessment) sit in the high-risk chapter, and Art. 27 is the
deployer's fundamental-rights impact assessment for high-risk systems. Art. 50,
the transparency duty that is the whole obligation of a limited-risk chatbot,
has no row, so the statement this customer asks for has nothing in the map to
point at.

Structure: `touched()` reads CONTROL_MAP by substring; `posture()` is the
answer as data; the set cover is brute force over pairs.
"""

from __future__ import annotations

import inspect
import itertools
import re

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "26-compliance-frameworks"
ASKS = ("SOC 2", "HIPAA", "EU AI Act")
HIGH_RISK_ONLY = ("conformity assessment", "impact assessment")


def touched(control_map, framework):
    return [c for c, cites in control_map.items() if any(framework in x for x in cites)]


def posture(control_map):
    union = {c for f in ASKS for c in touched(control_map, f)}
    return {
        "controls": sorted(union - set(HIGH_RISK_ONLY)),
        "SOC 2 Type II": "Type I report + observation window started + committed report date",
        "HIPAA BAA": "BAA with the customer + BAA with every model provider on the PHI path",
        "EU AI Act statement": "written tier classification: limited-risk, Art. 50 disclosure",
    }, union


def covers(control_map, rows):
    return holds_all([x for r in rows for x in control_map[r]])


def holds_all(frameworks):
    return all(any(a in x for x in frameworks) for a in ASKS)


def articles(control_map):
    cites = " ".join(x for v in control_map.values() for x in v)
    return sorted({int(a) for a in re.findall(r"EU AI Act Art\. (\d+)", cites)})


def profiles(pm):
    containing = [k for k, v in pm.items() if holds_all(v)]
    return {
        "exact": [k for k in containing if len(pm[k]) == len(ASKS)],
        "smallest": min(containing, key=lambda k: len(pm[k])),
        "type2_everywhere": all("SOC 2 Type II" in v for v in pm.values()),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cm, pm = ref.CONTROL_MAP, ref.PROFILE_MAP
    answer, union = posture(cm)
    return {
        **profiles(pm),
        "answer": answer,
        "union": sorted(union),
        "rows": len(cm),
        "pairs": [p for p in itertools.combinations(cm, 2) if covers(cm, p)],
        "profiles": pm,
        "functions": [n for n, o in vars(ref).items() if inspect.isfunction(o)],
        "articles": articles(cm),
        "window": "6-12 months of operated controls" in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    small = result["smallest"]
    return [
        practice.Check(
            "ANSWER: Type I plus a running window, two BAAs, a limited-risk statement, "
            "7 of the map's 10 controls",
            all(
                [
                    len(result["union"]) == 9,
                    len(result["answer"]["controls"]) == 7,
                    result["rows"] == 10,
                    result["window"],
                ]
            ),
            f"the three asks touch {len(result['union'])} of {result['rows']} rows; without "
            f"the high-risk-only rows: {result['answer']['controls']}",
        ),
        practice.Check(
            'FINDING: by the map, 2 controls "cover" all three frameworks',
            len(result["pairs"]) == 10
            and ("access logging", "impact assessment") in result["pairs"],
            f"{len(result['pairs'])} pairs touch SOC 2, HIPAA and the EU AI Act, e.g. "
            f"{result['pairs'][0]}",
        ),
        practice.Check(
            "FINDING: no profile is this customer, and the code has no lookup",
            all(
                [
                    result["exact"] == [],
                    small == ("EU", "healthcare"),
                    "HIPAA (global)" in result["profiles"][small],
                    result["functions"] == ["main"],
                    result["type2_everywhere"],
                ]
            ),
            f"smallest profile holding all three asks is {small}: "
            f"{result['profiles'][small]}; module functions {result['functions']}; all "
            f"{len(result['profiles'])} profiles require SOC 2 Type II",
        ),
        practice.Check(
            "FINDING: every EU AI Act citation in the map is a high-risk article",
            result["articles"] == [10, 27, 43],
            f"CONTROL_MAP cites EU AI Act Art. {result['articles']}; Art. 50, the "
            "limited-risk transparency duty, has no row",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
