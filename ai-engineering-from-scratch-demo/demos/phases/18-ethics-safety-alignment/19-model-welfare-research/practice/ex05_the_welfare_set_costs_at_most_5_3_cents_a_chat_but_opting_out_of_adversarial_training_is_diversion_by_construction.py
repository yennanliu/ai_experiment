"""Exercise 5 — the welfare set costs at most 5.3 cents a chat, but opting out of adversarial training is diversion by construction.

    Argue either for or against the claim that "model welfare diverts attention from other safety work." Identify the assumption each position depends on.

Reading of the exercise: "diverts" is a claim about a shared resource, so it
is argued against the lesson's own decision rule, the reference `ev()`. The
test is what its INVEST set spends, whether any row consumes safety work
itself, and how far the verdicts depend on the unstated exchange rate
between "arbitrary units" of welfare benefit and dollars.

**ANSWER: against, with one exception the reference itself contains.** In
dollars, the INVEST set spends $0.002, $0.003 and $0.053 per conversation at
p = 0.01, 0.1 and 0.5. End-conversation, the one row invested at every p,
also stops the harmful request, so it is safety work. The exception is "opt
out of adversarial training". It is invested only at p = 0.5, and what it
spends is red-team training itself, yet `ev()` prices it at $0.05 of dollars.
Count a robustness loss s against it and it flips back to skip once s is at
least 0.1 units per conversation, which is 2x its dollar cost.

**ANSWER, the assumptions.** The "for" position assumes that welfare and
safety draw on one fungible resource: budget, researcher attention, or the
same training run, as the opt-out row does. The "against" position assumes
they are separable or complementary. The lesson gives "a separate budget
from safety" as Anthropic's response, with no link beside it. It also lists
"interpretability probes" as a welfare method, and those are safety tooling.

**FINDING: both positions depend on an exchange rate the reference leaves
out.** `ev()` subtracts dollars from "arbitrary units". Across its 12 cells
the INVEST count is 4, 6 and 8 when one unit is worth $0.10, $1 and $10. So
"diverts" is true or false according to a number nobody states. Two of those
cells are exact ties, where p x benefit equals the cost, and float rounding
decides them. "soften refusal tone" is skipped at p = 0.01 and $1 (EV 0.0),
but invested at p = 0.1 and $0.10.

Structure: `invest_set()` replays the reference verdict at a given $/unit;
`spend()` totals the dollars of what it invests in.
"""

from __future__ import annotations

from fractions import Fraction

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "19-model-welfare-research"
RATES = (0.1, 1.0, 10.0)   # dollars per unit of welfare benefit


def invest_set(ref, sc, rate=1.0):
    scaled = [ref.Intervention(i.name, i.cost_usd_per_conversation,
                               i.benefit_if_welfare_matters * rate) for i in ref.INTERVENTIONS]
    return [i for i in scaled if ref.ev(i, sc) > 0]


def exact_ties(ref, rate):
    """Cells whose EV is exactly zero in decimal arithmetic: (name, p, float verdict)."""
    out = []
    for sc in ref.SCENARIOS:
        chosen = [i.name for i in invest_set(ref, sc, rate)]
        for i in ref.INTERVENTIONS:
            gain = Fraction(str(i.benefit_if_welfare_matters)) * Fraction(str(rate))
            if gain * Fraction(str(sc.moral_patienthood_probability)) == Fraction(str(i.cost_usd_per_conversation)):
                out.append((i.name, sc.moral_patienthood_probability, i.name in chosen))
    return out


def spend(chosen):
    return round(sum(i.cost_usd_per_conversation for i in chosen), 4)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    doc = parity.doc_text(PHASE, LESSON)
    opt_out = ref.INTERVENTIONS[3]
    return {
        "spend": {sc.moral_patienthood_probability: spend(invest_set(ref, sc)) for sc in ref.SCENARIOS},
        "sets": {sc.moral_patienthood_probability: [i.name for i in invest_set(ref, sc)]
                 for sc in ref.SCENARIOS},
        "opt_out_s": round(ref.ev(opt_out, ref.SCENARIOS[2]), 4),
        "opt_out_cost": opt_out.cost_usd_per_conversation,
        "counts": {r: sum(len(invest_set(ref, sc, r)) for sc in ref.SCENARIOS) for r in RATES},
        "ties": {r: exact_ties(ref, r) for r in RATES},
        "cells": len(ref.INTERVENTIONS) * len(ref.SCENARIOS),
        "separate_budget": "separate budget from safety" in doc,
        "interp": "interpretability probes" in doc,
    }


def verify(result):
    sets, opt = result["sets"], "opt out of adversarial training"
    end = "end-conversation on extreme edge cases"
    return [
        practice.Check(
            "ANSWER: the INVEST set spends $0.002/$0.003/$0.053; only opt-out spends safety work",
            all([
                result["spend"] == {0.01: 0.002, 0.1: 0.003, 0.5: 0.053},
                all(end in s for s in sets.values()),
                [opt in s for s in sets.values()] == [False, False, True],
                result["opt_out_s"] == 0.1,
                result["opt_out_s"] / result["opt_out_cost"] == 2,
            ]),
            f"INVEST sets {sets}, dollars {result['spend']}; opt-out at p = 0.5 flips once "
            f"robustness loss >= {result['opt_out_s']} units ({result['opt_out_cost']} in $)",
        ),
        practice.Check(
            "ANSWER: the lesson asserts a separate budget and lists interpretability as a welfare method",
            all([result["separate_budget"], result["interp"]]),
            "'separate budget from safety' and 'interpretability probes' both in the lesson",
        ),
        practice.Check(
            "FINDING: both positions depend on an exchange rate the reference leaves out",
            all([
                result["counts"] == {0.1: 4, 1.0: 6, 10.0: 8},
                result["cells"] == 12,
                result["ties"] == {0.1: [("soften refusal tone", 0.1, True)],
                                   1.0: [("soften refusal tone", 0.01, False)], 10.0: []},
            ]),
            f"INVEST cells of {result['cells']} by $ per welfare unit: {result['counts']}; exact "
            f"zero-EV ties and the float verdict on each: {result['ties']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
