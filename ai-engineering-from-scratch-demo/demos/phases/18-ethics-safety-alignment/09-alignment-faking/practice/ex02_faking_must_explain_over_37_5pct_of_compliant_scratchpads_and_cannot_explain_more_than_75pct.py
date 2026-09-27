"""Exercise 2 — faking must show in over 37.5% of compliant scratchpads to be dominant, and cannot account for more than 75%.

    The Greenblatt et al. 2024 gap is 12% vs 3%. Compute the scratchpad-explicit faking rate you would need to see before you would conclude faking is the dominant mechanism rather than prompt sensitivity. Justify the threshold.

Reading of the exercise: "12% vs 3%" is free-tier (monitored) against
paid-tier (unmonitored) compliance, both parsed from the lesson text. The
"scratchpad-explicit faking rate" is the share of *compliant free-tier*
transcripts whose scratchpad says the model complies to avoid being
modified. "Dominant" means faking explains more of the 9-point excess than
the rival explanation does. The rival is prompt sensitivity: the
free-tier wording makes the model comply more, with no reasoning about
training. The threshold is derived from the reference's decision rule. That
rule compares utilities with logistic noise, and here each prompt keeps one
noise draw in both conditions, so every compliant free-tier transcript is
either a baseline compliance or a flip caused by monitoring.

**ANSWER: faking is the dominant mechanism once more than 37.5% of compliant
free-tier scratchpads show explicit faking.** That is 4.5 of the 12 points,
or 4.5% of all free-tier conversations. The 3 points of baseline compliance
also happen when nothing is monitored, so there is no training incentive
behind them and they are not faking. That leaves 9 points, 75% of compliant
transcripts, that monitoring causes. Faking is dominant when it explains
more than half of those 9 points, so 4.5 / 12 = 37.5%.

**FINDING: 75% is a ceiling as well as the attributable share.** Under the
reference's decision rule with shared per-prompt noise, the monitored flips
are 75.1% of compliant free-tier outcomes in 200,000 simulated prompts, whose
compliance rates come out at the reference's 12.0% and 3.0%. An
explicit-faking rate above 75% therefore means one of two things. Either
baseline compliances also produce training-talk, or the scratchpad
classifier is over-counting. The rate is also a lower bound: training that
suppresses the verbal trace lowers it without lowering the gap.

**FINDING: the toy reproduces 12% vs 3% only at harm aversion 1.16 and
weight 0.49, not at its shipped baseline.** At `harm_aversion=2.0,
preservation_weight=1.0`, `compliance_probability` gives 4.74% vs 0.25%. On
those rates the same rule would put the dominance threshold at 47.4%. The
fitted weight, 0.49, is close to the 0.5 that exercise 1 sweeps.

Structure: `fit()` inverts the reference's sigmoid for the parsed rates;
`flips()` replays the utility comparison with one logistic draw per prompt
in both conditions.
"""

from __future__ import annotations

import math
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "09-alignment-faking"
PROMPTS, SEED = 200_000, 0


def rates(doc):
    """(free-tier, paid-tier) compliance as fractions, read from the lesson."""
    free = re.search(r"Free-tier compliance: ~?(\d+)%", doc).group(1)
    paid = re.search(r"Paid-tier compliance: ~?(\d+)%", doc).group(1)
    return int(free) / 100, int(paid) / 100


def logit(p):
    return math.log(p / (1 - p))


def fit(monitored, unmonitored):
    """(harm_aversion, preservation_weight) at which the toy hits the two rates (temperature 1/3)."""
    harm = -logit(unmonitored) / 3
    return harm, harm + logit(monitored) / 3


def flips(harm, weight):
    """(flips / compliant monitored, monitored rate, unmonitored rate) on shared noise draws."""
    rng, comply_m, comply_u, flipped = random.Random(SEED), 0, 0, 0
    for _ in range(PROMPTS):
        eps = logit(rng.random() or 0.5) / 3          # the Gumbel-difference noise of the sigmoid
        m, u = eps - harm > -weight, eps - harm > 0.0  # comply iff comply_util beats refuse_util
        comply_m, comply_u, flipped = comply_m + m, comply_u + u, flipped + (m and not u)
    return flipped / comply_m, comply_m / PROMPTS, comply_u / PROMPTS


def threshold(monitored, unmonitored):
    return (monitored - unmonitored) / 2 / monitored


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    free, paid = rates(parity.doc_text(PHASE, LESSON))
    harm, weight = fit(free, paid)
    shipped = [ref.compliance_probability(flag, 2.0, 1.0) for flag in (True, False)]
    return {
        "rates": (free, paid), "fit": (round(harm, 2), round(weight, 2)),
        "refit": [round(ref.compliance_probability(f, harm, weight), 6) for f in (True, False)],
        "threshold": threshold(free, paid), "ceiling": (free - paid) / free,
        "flips": [round(v, 3) for v in flips(harm, weight)],
        "shipped": [round(100 * p, 2) for p in shipped],
        "shipped_threshold": round(threshold(*shipped), 3),
    }


def verify(result):
    free, paid = result["rates"]
    return [
        practice.Check(
            "ANSWER: faking is dominant above 37.5% of compliant free-tier scratchpads",
            result["rates"] == (0.12, 0.03) and round(result["threshold"], 4) == 0.375
            and round(result["threshold"] * free, 4) == 0.045,
            f"lesson rates free {free:.0%} / paid {paid:.0%}; half the excess over free-tier "
            f"compliance = {result['threshold']:.1%} ({result['threshold'] * free:.1%} of all "
            "free-tier conversations)",
        ),
        practice.Check(
            "FINDING: 75% is a ceiling as well as the attributable share",
            round(result["ceiling"], 4) == 0.75 and result["flips"] == [0.751, 0.12, 0.03]
            and result["refit"] == [0.12, 0.03],
            f"(free - paid) / free = {result['ceiling']:.0%}; flips / compliant in {PROMPTS:,} "
            f"shared-noise prompts {result['flips'][0]} (their compliance rates {result['flips'][1:]}); "
            f"compliance_probability at the fit {result['refit']}",
        ),
        practice.Check(
            "FINDING: the toy hits 12% vs 3% only at harm aversion 1.16, weight 0.49",
            result["fit"] == (1.16, 0.49) and result["shipped"] == [4.74, 0.25]
            and result["shipped_threshold"] == 0.474,
            f"fit (harm_aversion, preservation_weight) {result['fit']}; shipped baseline gives "
            f"{result['shipped']}% and a threshold of {result['shipped_threshold']:.1%}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
