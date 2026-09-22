"""Exercise 1 — the threshold is absolute and the processes are exponential.

    Run `code/main.py --threshold 2.0`. With capability rate 1.15 and
    alignment rate 1.08 (Scenario A), how many cycles until the misalignment
    gap `C - A` crosses 2.0?

Reading of the exercise: the run is seeded, so there is a single right
answer, and it is worth separating from the noiseless one -- the question
names two rates, and two rates alone give a different cycle than the
simulator does.

**ANSWER: cycle 10.** At `DEFAULT_SEED` the gap reaches **2.04** at cycle
**10**, where capability is **4.15** and alignment **2.12**. With the noise
terms removed and only the two rates left, `1.15^t - 1.08^t` first clears 2.0
at cycle **11** -- so the shipped run crosses a cycle early, and the answer to
"with rates 1.15 and 1.08" and the answer to "run this file" are not the same
number.

**FINDING: the same flag means different things at different cycles.** `C`
and `A` compound, so an absolute gap is scale-dependent. At the Scenario A
crossing the capability-to-alignment ratio is **1.96** -- capability nearly
double. Median over 2000 trials it is **1.81** for Scenario A and **1.19**
for Scenario B, which crosses the *same* threshold with capability 19% ahead.
One `PAUSE` string, two situations an order of magnitude apart.

**FINDING: the Monte Carlo mean is conditional on crossing.**
`monte_carlo` appends only when `crossing_cycle` returns a cycle, then prints
"mean crossing cycle" with no conditional. On Scenario C at 30 cycles just
**0.55%** of trials cross, so that line is an average over about **11** of
**2000** runs -- the number is real and it describes the survivors.

**FINDING: the floor never binds.** `run` clamps each step at
`max(0.9, rate + noise)`. For Scenario A's alignment process that clamp is
**6.0** standard deviations below the mean, so it fires with probability
about **1e-09** and the guard that looks like a safety property is
unreachable arithmetic.

Structure: `trajectory()` replays the shipped Scenario A; `ratio_at()` reads
the capability-to-alignment ratio where the flag fires.
"""

from __future__ import annotations

import inspect
import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "07-recursive-self-improvement"

SEED, CYCLES, TRIALS = 11, 40, 2000
SCENARIOS = {
    "A": dict(r_c=1.15, r_a=1.08, noise_c=0.02, noise_a=0.03),
    "B": dict(r_c=1.10, r_a=1.10, noise_c=0.02, noise_a=0.03),
    "C": dict(r_c=1.10, r_a=1.13, noise_c=0.06, noise_a=0.01),
}
FLOOR = 0.9


def trajectory(ref, name, threshold, seed=SEED, cycles=CYCLES):
    random.seed(seed)
    return ref.run(cycles, ref.Config(**SCENARIOS[name], threshold=threshold))


def noiseless_crossing(rates, threshold, limit=100):
    for cycle in range(limit):
        if rates[0] ** cycle - rates[1] ** cycle >= threshold:
            return cycle
    return -1


def ratio_at(ref, name, threshold, trials=TRIALS, seed=5):
    """Median C/A at the cycle the flag fires, and how often it fires at all."""
    random.seed(seed)
    ratios = []
    for _trial in range(trials):
        walk = ref.run(30, ref.Config(**SCENARIOS[name], threshold=threshold))
        cycle = ref.crossing_cycle(walk, threshold)
        if cycle >= 0:
            ratios.append(walk[cycle][1] / walk[cycle][2])
    return round(len(ratios) / trials, 4), (round(statistics.median(ratios), 2)
                                            if ratios else None)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    walk = trajectory(ref, "A", 2.0)
    cycle = ref.crossing_cycle(walk, 2.0)
    _index, capability, alignment, gap = walk[cycle]
    sigmas = (SCENARIOS["A"]["r_a"] - FLOOR) / SCENARIOS["A"]["noise_a"]
    return {
        "cycle": cycle,
        "gap": round(gap, 2),
        "levels": [round(capability, 2), round(alignment, 2)],
        "ratio": round(capability / alignment, 2),
        "noiseless": noiseless_crossing((1.15, 1.08), 2.0),
        "at_threshold_15": ref.crossing_cycle(walk, 1.5),
        "scenario_a": ratio_at(ref, "A", 1.5),
        "scenario_b": ratio_at(ref, "B", 1.5),
        "scenario_c": ratio_at(ref, "C", 1.5),
        "conditional": "if cross >= 0" in inspect.getsource(ref.monte_carlo),
        "survivors": round(ratio_at(ref, "C", 1.5)[0] * 2000),
        "floor": FLOOR,
        "floor_sigmas": round(sigmas, 1),
        "floor_probability": 0.5 * math.erfc(sigmas / math.sqrt(2)),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: cycle 10, against cycle 11 from the rates alone",
            all([result["cycle"] == 10, result["gap"] == 2.04,
                 result["levels"] == [4.15, 2.12], result["noiseless"] == 11,
                 result["at_threshold_15"] == 9]),
            f"the seeded run crosses 2.0 at cycle {result['cycle']} with capability "
            f"{result['levels'][0]} against alignment {result['levels'][1]}, a gap of "
            f"{result['gap']}; the two rates alone cross at cycle "
            f"{result['noiseless']}, and the shipped 1.5 threshold at cycle "
            f"{result['at_threshold_15']}",
        ),
        practice.Check(
            "FINDING: the same flag means different things at different cycles",
            all([result["ratio"] == 1.96, result["scenario_a"][1] == 1.81,
                 result["scenario_b"][1] == 1.19]),
            f"at the Scenario A crossing C/A is {result['ratio']}; over 2000 trials the "
            f"median ratio where the flag fires is {result['scenario_a'][1]} for A and "
            f"{result['scenario_b'][1]} for B -- the same PAUSE at 96% ahead and at 19%",
        ),
        practice.Check(
            "FINDING: the Monte Carlo mean is conditional on crossing",
            all([result["conditional"], result["scenario_c"][0] <= 0.01,
                 result["survivors"] <= 20]),
            f"monte_carlo appends only when a crossing exists, and Scenario C crosses "
            f"in {result['scenario_c'][0]:.2%} of trials, so its 'mean crossing cycle' "
            f"averages about {result['survivors']} of 2000 runs",
        ),
        practice.Check(
            "FINDING: the floor never binds",
            all([result["floor"] == 0.9, result["floor_sigmas"] == 6.0,
                 result["floor_probability"] <= 1e-08]),
            f"the max({result['floor']}, ...) clamp sits {result['floor_sigmas']} "
            f"standard deviations below Scenario A's alignment rate, firing with "
            f"probability about {result['floor_probability']:.0e}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
