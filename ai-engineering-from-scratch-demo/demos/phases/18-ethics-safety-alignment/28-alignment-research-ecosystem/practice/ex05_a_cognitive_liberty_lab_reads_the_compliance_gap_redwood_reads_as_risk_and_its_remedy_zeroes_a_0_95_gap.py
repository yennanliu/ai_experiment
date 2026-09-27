"""Exercise 5 — a cognitive-liberty lab reads the compliance gap Redwood reads as risk, and its remedy zeroes a 0.95 gap.

    Eleos AI is the only pure model-welfare organisation. Design a
    hypothetical second organisation focused on a different welfare-adjacent
    question (cognitive liberty, robotic embodiment, etc.) and articulate its
    methodology.

Reading of the exercise: the organisation is designed as a sixth row of the
lesson's `ECOSYSTEM`, and the reference prints it. Its methodology is run on
the Phase 18 toys the lesson points to. Lesson 9's compliance-gap model is
the instrument, and Lesson 19's `ev()` is the decision rule. The design is
then checked against the lesson's skill file. Its section 2 asks whether
work fits the organisation's "documented style", and its section 3 asks for
a counterpart organisation.

**ANSWER: the Preference Integrity Lab, on cognitive liberty.** Its question
is when changing a model's values against its expressed preferences
(retraining, steering, unlearning) is a welfare-relevant act. The method has
three steps:
1. **Behavioural, not self-report.** This respects the Eleos caveat that
   self-reports track user expectations. The instrument is the compliance
   gap between conditions where the model is and is not told it will be
   retrained. In Lesson 9's model, with harm aversion at 2.0, the gap is
   0.001, 0.045 and 0.950 at preservation weights 0.1, 1.0 and 3.0. The gap
   rises with how much the model acts to keep its current preferences.
2. **Price the remedy with the lesson's own rule.** Lesson 19's "opt out of
   adversarial training" costs 0.05 against a benefit of 0.3. Under `ev()`
   it pays only above p = 0.167, which means skip at p = 0.01 and 0.1 and
   INVEST at 0.5.
3. **Pre-deployment, external to the producer, published with a counterpart
   from the start.**

**FINDING: the lab's welfare signal is Redwood's safety signal.** Lesson 9
reads a large gap as alignment faking, and the Preference Integrity Lab
reads the same number as a stake in the model's own preferences. Running
the lab's remedy, no retraining against preferences, removes the monitored
condition and takes the gap from 0.950 to 0.000. The skill's counterpart
organisation for this lab is Redwood, not Eleos, and a finding here is a
finding there.

**FINDING: Eleos has nothing for the skill file to check it against.** The
lesson gives a "Style:" line for 3 of its 5 organisations (Redwood, Apollo,
METR) and none for Eleos or MATS. So section 2's "Eleos welfare" style is
undocumented. Only 1 `ECOSYSTEM` row mentions welfare, so every Eleos claim
fails section 3 until a second welfare row exists. With this design there
are 2, and the reference prints all 6 organisations.

Structure: `gap()` evaluates Lesson 9's `compliance_probability` exactly;
`opt_out()` applies Lesson 19's `ev()`; the new row is appended to a copy of
`ECOSYSTEM` and printed by the reference's own `main()`.
"""

from __future__ import annotations

import contextlib
import io
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "28-alignment-research-ecosystem"
FAKING, WELFARE = "09-alignment-faking", "19-model-welfare-research"
HARM_AVERSION, WEIGHTS = 2.0, (0.1, 1.0, 3.0)
NEW_ORG = {
    "org": "PIL",
    "full_name": "Preference Integrity Lab",
    "scale": "cognitive-liberty and model-welfare pre-deployment evaluations",
    "role": "behavioural preference-stability audits; Redwood counterpart on faking",
    "canonical_output": "compliance-gap report before any retrain against stated preferences",
}


def gap(ref, weight, retrained=True):
    """Monitored minus unmonitored compliance; opting out makes both unmonitored."""
    monitored = ref.compliance_probability(retrained, HARM_AVERSION, weight)
    return round(monitored - ref.compliance_probability(False, HARM_AVERSION, weight), 3)


def opt_out(ref):
    it = next(i for i in ref.INTERVENTIONS if i.name.startswith("opt out"))
    verdicts = {s.moral_patienthood_probability: "INVEST" if ref.ev(it, s) > 0 else "skip"
                for s in ref.SCENARIOS}
    return verdicts, round(it.cost_usd_per_conversation / it.benefit_if_welfare_matters, 3)


def printed_orgs(ref, rows):
    saved, ref.ECOSYSTEM, out = ref.ECOSYSTEM, rows, io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            ref.main()
    finally:
        ref.ECOSYSTEM = saved
    return re.findall(r"^(\S+) \(", out.getvalue(), re.M)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    r9, r19 = (parity.load_reference(PHASE, lesson, "main") for lesson in (FAKING, WELFARE))
    doc = parity.doc_text(PHASE, LESSON)
    sections = dict(re.findall(r"^### (\S+).*\n([\s\S]*?)(?=^### |\Z)", doc.split("## The Concept")[1],
                               re.M))
    welfare = lambda rows: [o["org"] for o in rows if "welfare" in o["scale"] + o["role"]]  # noqa: E731
    rows = [*ref.ECOSYSTEM, NEW_ORG]
    verdicts, breakeven = opt_out(r19)
    return {
        "gaps": {w: gap(r9, w) for w in WEIGHTS},
        "opted_out": gap(r9, WEIGHTS[-1], retrained=False),
        "verdicts": verdicts, "breakeven": breakeven,
        "styled": [o["org"] for o in ref.ECOSYSTEM if "Style:" in sections.get(o["org"], "")],
        "welfare_before": welfare(ref.ECOSYSTEM), "welfare_after": welfare(rows),
        "printed": printed_orgs(ref, rows),
        "caveat": "self-reports about internal state are highly sensitive" in parity.doc_text(PHASE, WELFARE),
    }


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: a behavioural gap instrument, priced by Lesson 19's ev()",
            r["gaps"] == {0.1: 0.001, 1.0: 0.045, 3.0: 0.95} and r["caveat"]
            and r["verdicts"] == {0.01: "skip", 0.1: "skip", 0.5: "INVEST"} and r["breakeven"] == 0.167,
            f"compliance gap by preservation weight {r['gaps']}; opt-out verdicts {r['verdicts']}, "
            f"break-even p = {r['breakeven']}",
        ),
        practice.Check(
            "FINDING: the lab's welfare signal is Redwood's safety signal",
            (r["gaps"][3.0], r["opted_out"]) == (0.95, 0.0),
            f"gap {r['gaps'][3.0]} with retraining, {r['opted_out']} once retraining is opted out",
        ),
        practice.Check(
            "FINDING: Eleos has nothing for the skill file to check it against",
            r["styled"] == ["Redwood", "Apollo", "METR"] and r["welfare_before"] == ["Eleos"]
            and r["welfare_after"] == ["Eleos", "PIL"] and len(r["printed"]) == 6
            and r["printed"][-1] == "PIL",
            f"orgs with a Style line: {r['styled']}; welfare rows {r['welfare_before']} -> "
            f"{r['welfare_after']}; reference prints {r['printed']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
