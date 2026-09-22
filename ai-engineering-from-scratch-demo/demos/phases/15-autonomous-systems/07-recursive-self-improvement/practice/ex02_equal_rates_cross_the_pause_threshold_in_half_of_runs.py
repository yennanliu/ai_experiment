"""Exercise 2 — equal rates cross the pause threshold in half of runs.

    Set both rates equal. Does the gap stay bounded or does noise push it one
    way? What does this imply for RSI safety?

Reading of the exercise: "bounded" has to be answered against the rule the
simulator actually applies -- the 1.5 pause threshold -- not against the
average, because an average can sit at zero while every individual run
wanders. The noise question is then separated into two parts the shipped
scenario confounds: the spread, and the asymmetry between the two noise
levels.

**ANSWER: not bounded, and noise pushes it toward capability.** With
`r_c = r_a = 1.10` over 40 cycles the gap crosses 1.5 in **54.5%** of runs,
and its 10th-to-90th percentile at cycle 40 spans **-12.2 to +11.9** --
sixteen times the threshold. The median final gap is **+0.46**, positive.

**FINDING: the drift is Jensen, not rate.** Arithmetic rates are equal, so
the mean final gap is **0.001**. But a multiplicative process with more noise
has a lower *geometric* mean, and Scenario B gives alignment the noisier term
-- **0.03** against capability's **0.02**. Swap them and the median gap flips
to **-0.42**; equalise them at 0.02 and it is **-0.04**. The direction of the
drift is set by which process is noisier, and nothing in the config says so.

**FINDING: the crossings are variance, not drift.** At equal rates *and*
equal noise the gap still crosses 1.5 in **49.9%** of runs, against **54.5%**
with the shipped asymmetry. So about **50** of the **54.5** points of
crossing probability are a random walk with no direction at all; the
asymmetry contributes the other **5**.

**FINDING: with the noise removed the rule never fires.** At zero noise the
crossing rate is **0.0%** and the final gap is exactly **0.0**. The pause
threshold in this simulator is triggered entirely by the noise terms, which
means a matched-rate RSI pipeline would be halted by the shipped rule in
roughly half of its runs for no reason a rate can name.

Structure: `sweep()` runs one noise configuration at equal rates; every claim
is a difference between two of its rows.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "07-recursive-self-improvement"

RATE, THRESHOLD, CYCLES, TRIALS, SEED = 1.10, 1.5, 40, 4000, 3
SHIPPED, SWAPPED, EQUAL, SILENT = (0.02, 0.03), (0.03, 0.02), (0.02, 0.02), (0.0, 0.0)


def sweep(ref, noise, trials=TRIALS):
    """(crossing rate, median final gap, mean final gap) at equal rates."""
    random.seed(SEED)
    crossed, finals = 0, []
    config = ref.Config(r_c=RATE, r_a=RATE, noise_c=noise[0], noise_a=noise[1],
                        threshold=THRESHOLD)
    for _trial in range(trials):
        walk = ref.run(CYCLES, config)
        crossed += ref.crossing_cycle(walk, THRESHOLD) >= 0
        finals.append(walk[-1][3])
    return (round(crossed / trials, 4), round(statistics.median(finals), 3),
            round(statistics.mean(finals), 3), finals)


def spread(finals):
    deciles = statistics.quantiles(finals, n=10)
    return [round(deciles[0], 1), round(deciles[8], 1)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shipped, swapped = sweep(ref, SHIPPED), sweep(ref, SWAPPED)
    equal, silent = sweep(ref, EQUAL), sweep(ref, SILENT)
    return {
        "shipped": shipped[:3],
        "swapped": swapped[:3],
        "equal": equal[:3],
        "silent": silent[:3],
        "spread": spread(shipped[3]),
        "threshold": THRESHOLD,
        "spread_ratio": round((spread(shipped[3])[1] - spread(shipped[3])[0])
                              / THRESHOLD, 1),
        "noise": [list(SHIPPED), list(SWAPPED), list(EQUAL), list(SILENT)],
        "variance_share": round(equal[0] / shipped[0], 3),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: not bounded -- 54.5% of runs cross, spread -12.1 to +12.0",
            all([result["shipped"][0] == 0.5453, result["shipped"][1] == 0.457,
                 result["spread"] == [-12.2, 11.9], result["spread_ratio"] == 16.1]),
            f"at equal rates the gap crosses {result['threshold']} in "
            f"{result['shipped'][0]:.1%} of runs, with a 10-90 spread of "
            f"{result['spread']} -- {result['spread_ratio']}x the threshold -- and a "
            f"median final gap of {result['shipped'][1]}",
        ),
        practice.Check(
            "FINDING: the drift is Jensen, not rate",
            all([result["shipped"][2] == 0.001, result["swapped"][1] == -0.419,
                 result["equal"][1] == -0.042]),
            f"arithmetic rates are equal so the mean final gap is "
            f"{result['shipped'][2]}, but the median is {result['shipped'][1]} with "
            f"alignment noisier, {result['swapped'][1]} with capability noisier and "
            f"{result['equal'][1]} with the two equal",
        ),
        practice.Check(
            "FINDING: the crossings are variance, not drift",
            all([result["equal"][0] == 0.4988, result["variance_share"] >= 0.90]),
            f"equal rates and equal noise still cross in {result['equal'][0]:.1%} of "
            f"runs against {result['shipped'][0]:.1%} with the asymmetry -- "
            f"{result['variance_share']:.0%} of the crossing probability is a "
            "directionless random walk",
        ),
        practice.Check(
            "FINDING: with the noise removed the rule never fires",
            all([result["silent"] == (0.0, 0.0, 0.0)]),
            f"at zero noise the crossing rate is {result['silent'][0]:.1%} and the "
            "final gap is exactly 0.0, so every pause this rule issues at matched rates "
            "is issued by the noise terms",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
