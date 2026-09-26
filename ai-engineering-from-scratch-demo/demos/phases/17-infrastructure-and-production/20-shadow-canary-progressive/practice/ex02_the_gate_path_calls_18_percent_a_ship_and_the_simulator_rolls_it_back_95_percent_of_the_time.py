"""Exercise 2 — the gate path calls +18% a ship, and the simulator rolls it back 95% of the time.

    Your new model has 3% accuracy gain offline but cost/request is +18%. Is it
    a ship? Depends on the policy — write both paths.

Reading of the exercise: "3% accuracy gain" is read as 3 percentage points
and "+18%" as `Regression(cost_mult=1.18)` in the lesson's simulator. The two
policies are the lesson's own -- a model ships if it clears every gate -- and
a unit-economics policy that prices the accuracy against the cost. Both are
run, not just stated.

**ANSWER: the gate path says ship and the reference rolls it back; the value
path ships iff a correct answer is worth at least $0.12.** Gate path: +18% is
under the 1.2 cost gate, so by the policy it is a ship -- but on the shipped
seeds the reference halts it at the 10% stage on `cost_per_req`. Value path:
ship when V x 0.03 >= 0.18 x $0.02, i.e. V >= $0.12 per correct answer, 6x
the baseline cost of a request. It holds for any baseline accuracy, because
both sides are per request. Cost per correct answer still rises: at 80%
baseline accuracy it goes $0.0250 -> $0.0284, +13.7%.

**FINDING: a gate set 2 points above the true regression halts it by noise.**
With +-8% noise, 1.18 breaches whenever the draw exceeds 1.0169, 39.4% per
stage; it survives six stages with probability 0.606^6 = 4.95%, and over
10,000 seeded rollouts it promotes 523 times (5.2%). The gate path's "ship"
means "ship 1 time in 20".

**FINDING: the gate cannot see the accuracy gain at all.** `Regression` has
five fields -- latency, cost, error, output length, thumbs-down -- and no
accuracy, so the 3-point gain is not an input to either the gates or the
halt. Taking the value path through the canary means raising the cost gate
past the noise: at 1.28 (> 1.18 x 1.08 = 1.2744) the model promotes on all
10,000 schedules, and the gate then only guarantees a catch from 1.28 / 0.92
= 1.391.

Structure: `promotions()` counts seeded rollouts that reach 100% through the
reference's `measure_stage` and `check_gates`, under the shipped gates or a
raised cost gate restored afterwards.
"""

from __future__ import annotations

import dataclasses

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "20-shadow-canary-progressive"
GAIN, COST_UP, BASE_ACC, ROLLOUTS, RAISED = 0.03, 0.18, 0.80, 10_000, 1.28


def halt_stage(ref, reg, seed_of):
    for i, stage in enumerate(ref.STAGES):
        if ref.check_gates(ref.measure_stage(stage, reg, seed=seed_of(i))):
            return i
    return None


def promotions(ref, reg, cost_gate=None):
    saved = ref.GATES["cost_per_req"]
    ref.GATES["cost_per_req"] = cost_gate or saved
    try:
        return sum(
            halt_stage(ref, reg, lambda i, r=r: 1_000 * r + i) is None
            for r in range(ROLLOUTS)
        )
    finally:
        ref.GATES["cost_per_req"] = saved


def value_path(base_cost):
    """Break-even value of one correct answer, and cost per correct answer before/after."""
    break_even = COST_UP * base_cost / GAIN
    before = base_cost / BASE_ACC
    after = base_cost * (1 + COST_UP) / (BASE_ACC + GAIN)
    return round(break_even, 4), round(before, 4), round(after, 4)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    reg = ref.Regression(cost_mult=1 + COST_UP)
    gate = ref.GATES["cost_per_req"]
    p_breach = (1.08 - gate / (1 + COST_UP)) / 0.16
    shipped = halt_stage(ref, reg, ref.stage_seed)
    return {
        "inside_gate": 1 + COST_UP < gate,
        "shipped_halt": ref.STAGES[shipped] if shipped is not None else None,
        "p_breach": round(p_breach, 3),
        "p_promote": round((1 - p_breach) ** 6, 4),
        "promoted": promotions(ref, reg),
        "value": value_path(ref.BASELINE["cost_per_req"]),
        "fields": [f.name for f in dataclasses.fields(ref.Regression)],
        "raised": promotions(ref, reg, RAISED),
        "gate_after": ref.GATES["cost_per_req"],
        "raised_catch": round(RAISED / 0.92, 3),
    }


def verify(result):
    break_even, before, after = result["value"]
    return [
        practice.Check(
            "ANSWER: the gate path says ship and the reference rolls it back; the value "
            "path ships iff a correct answer is worth at least $0.12",
            all(
                [
                    result["inside_gate"],
                    result["shipped_halt"] == 0.1,
                    break_even == 0.12,
                    (before, after) == (0.025, 0.0284),
                ]
            ),
            f"+18% is inside the 1.2 gate yet halts at the {result['shipped_halt']:.0%} "
            f"stage on the shipped seeds; break-even ${break_even} per correct answer; "
            f"cost per correct answer ${before} -> ${after}",
        ),
        practice.Check(
            "FINDING: a gate set 2 points above the true regression halts it by noise",
            all(
                [
                    result["p_breach"] == 0.394,
                    result["p_promote"] == 0.0495,
                    result["promoted"] == 523,
                ]
            ),
            f"{result['p_breach']:.1%} breach chance per stage, {result['p_promote']:.2%} "
            f"to survive six; {result['promoted']} of {ROLLOUTS} seeded rollouts promote",
        ),
        practice.Check(
            "FINDING: the gate cannot see the accuracy gain at all",
            all(
                [
                    not any("acc" in f for f in result["fields"]),
                    len(result["fields"]) == 5,
                    result["raised"] == ROLLOUTS,
                    result["gate_after"] == 1.2,
                ]
            ),
            f"Regression fields {result['fields']}; with the cost gate at {RAISED} it "
            f"promotes {result['raised']} of {ROLLOUTS}, and then only catches from "
            f"{result['raised_catch']}x",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
