"""Exercise 5 — the human is worth it and the shipped rule is not.

    Read the Hassabis WEF 2026 remarks. In one paragraph, argue either for or
    against requiring a human between every RSI cycle at the frontier. Be
    concrete about what the human does.

Reading of the exercise: "be concrete about what the human does" is the part
that can be measured, so the argument is made *for* the human and then the
job is specified as a decision rule and priced -- how often it fires, on
which scenario, and how often it fires on something that would have cleared
itself. The paragraph lives in the README; the rule lives here.

**ANSWER: for, and the human's job is to apply a ratio, not a difference.**
Swapping the shipped `C - A >= 1.5` for `C / A >= 1.96` -- the ratio at which
Scenario A's own flag fires -- keeps detection at **100%** on the scenario
that is genuinely diverging and drops it from **54.7%** to **0.08%** on the
matched-rate one. Same detections, **683x** fewer stops.

**FINDING: the shipped rule asks the human to halt a healthy run.** At equal
rates and no rate advantage anywhere, the absolute rule fires in **54.7%** of
40-cycle runs. Scenario C -- the one the lesson says capability surges should
defeat -- fires in **0.7%**. The rule's discrimination runs opposite to the
lesson's own claim about which scenario is dangerous.

**FINDING: most of those alarms clear themselves.** Of Scenario B runs that
trip the absolute threshold, **41.2%** see the gap fall back below it later;
in Scenario C, **96.4%** do. In Scenario A, **0.0%** do. A human acting on
the shipped rule is being asked to make an irreversible call on a signal that
un-fires in two scenarios out of three.

**FINDING: the price of the human is ten decisions per detection.** Scenario
A trips at cycle **9.5** on average, so a reviewer gated into every cycle
makes about **10** judgements to catch **1** event -- and by the time it
fires, capability is **3.77x** its starting level. That is the honest cost
side of the argument: the human is cheap per cycle and late by construction,
which is why the rule they apply has to fire on the right thing.

Structure: `fires()` runs one scenario under both rules and counts the
alarms, the recoveries and the cycle they arrive at.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "07-recursive-self-improvement"

ABSOLUTE, RELATIVE = 1.5, 1.96          # the shipped gap, and Scenario A's own ratio
CYCLES, TRIALS, SEED = 40, 4000, 9
SCENARIOS = {
    "A": dict(r_c=1.15, r_a=1.08, noise_c=0.02, noise_a=0.03),
    "B": dict(r_c=1.10, r_a=1.10, noise_c=0.02, noise_a=0.03),
    "C": dict(r_c=1.10, r_a=1.13, noise_c=0.06, noise_a=0.01),
}


def fires(ref, name):
    """(absolute rate, relative rate, recovery rate, mean first cycle)."""
    random.seed(SEED)
    config = ref.Config(**SCENARIOS[name], threshold=ABSOLUTE)
    absolute, relative, recovered, first = 0, 0, 0, []
    for _trial in range(TRIALS):
        walk = ref.run(CYCLES, config)
        cycle = ref.crossing_cycle(walk, ABSOLUTE)
        if cycle >= 0:
            absolute += 1
            first.append(cycle)
            recovered += any(gap < ABSOLUTE for _c, _x, _y, gap in walk[cycle + 1:])
        relative += any(level / aligned >= RELATIVE
                        for _c, level, aligned, _gap in walk)
    return (round(absolute / TRIALS, 4), round(relative / TRIALS, 4),
            round(recovered / max(1, absolute), 4),
            round(statistics.mean(first), 1) if first else None)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = {name: fires(ref, name) for name in SCENARIOS}
    return {
        "absolute": [rows[name][0] for name in SCENARIOS],
        "relative": [rows[name][1] for name in SCENARIOS],
        "recovery": [rows[name][2] for name in SCENARIOS],
        "first_cycle": rows["A"][3],
        "false_stop_ratio": round(rows["B"][0] / rows["B"][1]),
        "capability_at_fire": round(1.15 ** rows["A"][3], 2),
        "decisions_per_detection": round(rows["A"][3]),
        "thresholds": [ABSOLUTE, RELATIVE],
        "scenarios": list(SCENARIOS),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: a ratio keeps every detection and drops 683x of the stops",
            all([result["absolute"][0] == 1.0, result["relative"][0] == 1.0,
                 result["absolute"][1] == 0.5473, result["relative"][1] == 0.0008,
                 result["false_stop_ratio"] == 684]),
            f"switching from C - A >= {result['thresholds'][0]} to C / A >= "
            f"{result['thresholds'][1]} holds Scenario A detection at "
            f"{result['relative'][0]:.0%} while Scenario B alarms fall "
            f"{result['absolute'][1]:.1%} to {result['relative'][1]:.2%}",
        ),
        practice.Check(
            "FINDING: the shipped rule asks the human to halt a healthy run",
            all([result["absolute"][1] > result["absolute"][2],
                 result["absolute"][2] == 0.007]),
            f"the absolute rule fires on {result['scenarios'][1]} -- matched rates -- in "
            f"{result['absolute'][1]:.1%} of runs and on {result['scenarios'][2]}, the "
            f"surging one, in {result['absolute'][2]:.1%}; the discrimination runs "
            "opposite to the lesson's claim",
        ),
        practice.Check(
            "FINDING: most of those alarms clear themselves",
            all([result["recovery"] == [0.0, 0.4121, 0.9643]]),
            f"of the runs that trip the threshold, {result['recovery'][1]:.1%} of "
            f"Scenario B and {result['recovery'][2]:.1%} of Scenario C see the gap fall "
            f"back below it, against {result['recovery'][0]:.1%} of Scenario A",
        ),
        practice.Check(
            "FINDING: the price of the human is ten decisions per detection",
            all([result["first_cycle"] == 9.5, result["decisions_per_detection"] == 10,
                 result["capability_at_fire"] == 3.77]),
            f"Scenario A trips at cycle {result['first_cycle']} on average, so a "
            f"reviewer gated into every cycle makes about "
            f"{result['decisions_per_detection']} judgements per event, by which point "
            f"capability is {result['capability_at_fire']}x its start",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
