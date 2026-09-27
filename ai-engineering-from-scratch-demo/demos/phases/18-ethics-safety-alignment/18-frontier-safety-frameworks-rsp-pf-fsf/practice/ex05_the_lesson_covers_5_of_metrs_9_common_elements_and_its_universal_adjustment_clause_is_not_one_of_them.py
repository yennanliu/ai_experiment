"""Exercise 5 — the lesson covers 5 of METR's 9 common elements, and its universal adjustment clause is not one of them.

    Read METR's "Common Elements of Frontier AI Safety Policies" (2025). Name
    the three strongest cross-lab convergences and the two largest
    divergences.

Reading of the exercise: METR's report (metr.org/common-elements, December
2025 version, read 2026-09-27) names nine elements across 12 companies'
policies. It analyses RSP v2.2, PF v2 and FSF v3.0. The answer names
convergences and divergences among the lesson's three labs. It then measures
how much of METR's frame the lesson carries, by matching each element's
keywords against the lesson's Concept section and against main.py's five
comparison axes.

**ANSWER, convergences:** (1) capability thresholds in CBRN and AI R&D, (2)
model-weight security that escalates with the threshold (RSP's ASL-3
Security Standard, FSF's security levels 2-4), and (3) deployment
mitigations signed off after governance review of a written risk argument
(Risk Reports to the Board, Safeguards Reports to PF's Safety Advisory
Group, FSF's safety case to a governance function). All three are among the
5 elements the lesson's Concept section mentions.

**ANSWER, divergences:**
1. **The threshold construct itself.** The page's ASL ladder has 5 rungs and
   main.py's PF ladder 4 (PF v2 has 2). FSF lists 4 per-domain CCLs and adds
   harmful manipulation and misalignment. RSP v3.0 drops control-list ASLs
   for future levels in favour of the argument required.
2. **The competitor contingency's direction.** RSP v3.0 only ratchets up, PF
   v2 may reduce requirements under conditions, and FSF v3.0 weighs a peer's
   weaker model as a risk factor (exercise 2).

**FINDING: the lesson omits METR's halting commitments.** Its Concept section
matches none of halting deployment, halting development, full capability
elicitation or updating policies, the elements that say what a lab does
when a threshold is crossed.

**FINDING: "three tiers of frontier capability" matches none of the three.**
The page and main.py's takeaway both claim it, but the rung counts the
lesson itself gives are 5 (ASL), 4 (PF) and 4 domains of one level each
(FSF).

**FINDING: the reference's "universal" convergence is not a METR element.**
main.py's takeaway calls competitor-adjustment clauses universal. Its
`adjustment_clause` axis matches none of METR's nine elements, and four of
its five axes match the single element Capability Thresholds.

Structure: `KEYWORDS` maps each METR element to a regex; `coverage()`
applies it to a text; `rungs()` counts the tiers the lesson gives.
"""

from __future__ import annotations

import contextlib
import io
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "18-frontier-safety-frameworks-rsp-pf-fsf"
# METR's nine common elements, in its order (December 2025 version), with the words that signal each.
KEYWORDS = {
    "Capability Thresholds": r"threshold|CCL|Capability Level",
    "Model Weight Security": r"security|weights?\b",
    "Model Deployment Mitigations": r"safeguard|mitigation",
    "Conditions for Halting Deployment Plans": r"halt|pause|withh[oe]ld|stop deploy",
    "Conditions for Halting Development Plans": r"halt|pause|stop (training|development|scaling)",
    "Full Capability Elicitation During Evaluations": r"elicit",
    "Timing and Frequency of Evaluations": r"quarterly|cadence|frequen|before scaling",
    "Accountability": r"Board|Advisory Group|externally reviewed|oversee",
    "Updating Policies Over Time": r"updat|revis",
}
CONVERGENT = ("Capability Thresholds", "Model Weight Security", "Model Deployment Mitigations")


def coverage(text):
    return [name for name, rx in KEYWORDS.items() if re.search(rx, text, re.I)]


def rungs(doc, ref):
    labs = {lab["name"].split()[1]: lab for lab in ref.LABS}
    deepmind = doc.split("### DeepMind")[1].split("\n### ")[0]
    return {
        "RSP": len(re.findall(r"^- ASL-", doc, re.M)),
        "PF": len(labs["PF"]["tier_structure"].split(" per ")[0].split(" / ")),
        "FSF": len(re.findall(r"^- \w+ ", deepmind.split("v2.0")[0], re.M)),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    doc = parity.doc_text(PHASE, LESSON)
    concept = doc.split("## The Concept")[1].split("## Use It")[0]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ref.main()
    axes = [k for k in ref.LABS[0] if k != "name"]
    return {
        "concept": coverage(concept),
        "axes": {a: coverage(a + " " + " ".join(lab[a] for lab in ref.LABS)) for a in axes},
        "rungs": rungs(doc, ref),
        "three_tiers": ["three tiers of frontier capability" in " ".join(t.split())
                        for t in (doc, out.getvalue())],
        "universal": "competitor-adjustment clauses universal" in " ".join(out.getvalue().split()),
    }


def verify(result):
    r = result
    missing = [k for k in KEYWORDS if k not in r["concept"]]
    single = [a for a, hits in r["axes"].items() if "Capability Thresholds" in hits]
    return [
        practice.Check(
            "ANSWER: thresholds, weight security, reviewed deployment mitigations converge",
            (len(r["concept"]), r["concept"][:3]) == (5, list(CONVERGENT)),
            f"the Concept section matches {len(r['concept'])} of 9 METR elements: {r['concept']}",
        ),
        practice.Check(
            "FINDING: the lesson omits METR's halting commitments",
            missing == ["Conditions for Halting Deployment Plans", "Conditions for Halting Development Plans",
                        "Full Capability Elicitation During Evaluations", "Updating Policies Over Time"],
            f"unmatched: {missing}",
        ),
        practice.Check(
            "FINDING: 'three tiers of frontier capability' matches none of the three",
            (r["three_tiers"], r["rungs"]) == ([True, True], {"RSP": 5, "PF": 4, "FSF": 4}),
            f"claimed in page and main.py: {r['three_tiers']}; rungs the lesson gives: {r['rungs']}",
        ),
        practice.Check(
            "FINDING: the reference's 'universal' convergence is not a METR element",
            (r["universal"], r["axes"]["adjustment_clause"], len(single)) == (True, [], 4),
            f"{len(single)} of 5 axes match Capability Thresholds; axis -> METR elements {r['axes']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
