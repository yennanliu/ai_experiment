"""Exercise 1 — 207,702 per arm, and at that n the sequential rule the code demos has 20% power.

    Run `code/main.py`. For an expected 5% lift with baseline 3% conversion,
    what sample size to 80% power?

Reading of the exercise: "5% lift" is relative (3% -> 3.15%), which is how
`fixed_sample_size` reads `lift`; the answer is the reference's number, checked
against the exact normal quantiles, and then run: the lesson's own simulator is
driven at that horizon to see whether the test it demonstrates reaches 80%.

**ANSWER: 207,702 users per arm -- 415,404 in total, 290,782 per arm with the
lesson's x1.4 non-determinism buffer.** `main.py` prints 207702 as "fixed
sample size" without saying it is per arm; the formula is the two-proportion
per-group n. With exact quantiles (1.95996, 0.84162) it is 207,938, 0.1% more;
the normal-approximation power of 207,702 per arm is 0.7995.

**FINDING: `alpha` and `power` are accepted and ignored.** The z-values are
hardcoded 1.96 and 0.84, so `power=0.9` and `alpha=0.01` both return 207,702.

**FINDING: the lesson's own 5%-lift run stops at 72% of that n, so it cannot
answer the question.** `simulate(0.03, 0.0315)` caps at 300,000 users total,
150,026 in A. Its end-of-run fixed z is 2.55, which a fixed test would call
significant; the sequential rule, whose boundary there is 4.31, never stops.

**FINDING: at the 80%-power horizon the sequential rule stops in 8 of 40
seeds -- 20% power.** Its boundary sqrt(2 ln 20 + ln n) is 4.35 at n = 415,404,
where the expected z is only 2.80. The lesson says the simulation "shows how
sequential lets you stop early"; in its 10%-lift demo it stops at 159,731
users, 1.5x the 106,300 a fixed test needs.

Structure: `solve()` calls the reference's `fixed_sample_size`, `simulate` and
`z_statistic`; the 40-seed power run uses seeds 0..39 at max_n = 2 x 207,702.
"""

from __future__ import annotations

import math
from statistics import NormalDist

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "21-ab-testing-llm-features"
BASE, LIFT, SEEDS = 0.03, 0.05, 40


def boundary(n_total):
    return math.sqrt(2 * math.log(1 / 0.05) + math.log(n_total))


def exact_n(p0, p1, alpha=0.05, power=0.80):
    za, zb = NormalDist().inv_cdf(1 - alpha / 2), NormalDist().inv_cdf(power)
    p_bar = (p0 + p1) / 2
    num = (
        za * math.sqrt(2 * p_bar * (1 - p_bar))
        + zb * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))
    ) ** 2
    return num / (p1 - p0) ** 2


def z_at_end(ref, run):
    """Fixed-horizon z from a finished run's observed rates."""
    sa, sb = (
        round(run["p_a_observed"] * run["n_a"]),
        round(run["p_b_observed"] * run["n_b"]),
    )
    return ref.z_statistic(sa, run["n_a"], sb, run["n_b"])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    n = ref.fixed_sample_size(BASE, LIFT)
    p1 = BASE * (1 + LIFT)
    se = math.sqrt((BASE * (1 - BASE) + p1 * (1 - p1)) / n)
    demo = ref.simulate(BASE, p1)
    runs = [ref.simulate(BASE, p1, seed=s, max_n=2 * n) for s in range(SEEDS)]
    return {
        "n": n,
        "exact": exact_n(BASE, p1),
        "buffered": int(n * 1.4),
        "power": 1 - NormalDist().cdf(1.96 - (p1 - BASE) / se),
        "ignored": (
            ref.fixed_sample_size(BASE, LIFT, power=0.9),
            ref.fixed_sample_size(BASE, LIFT, alpha=0.01),
        ),
        "demo": (
            demo["n_a"],
            demo["n_a"] + demo["n_b"],
            z_at_end(ref, demo),
            demo["sequential_stop_at"],
        ),
        "stops": sum(r["sequential_stop_at"] is not None for r in runs),
        "drift": (p1 - BASE) / se,
        "ten": (
            ref.simulate(BASE, 0.033)["sequential_stop_at"],
            2 * ref.fixed_sample_size(BASE, 0.10),
        ),
    }


def verify(result):
    n, demo, ten = result["n"], result["demo"], result["ten"]
    return [
        practice.Check(
            "ANSWER: 207,702 users per arm, 415,404 in total",
            n == 207702
            and result["buffered"] == 290782
            and round(result["exact"]) == 207938
            and round(result["power"], 4) == 0.7995,
            f"{n} per arm ({2 * n} total, {result['buffered']} per arm x1.4); exact quantiles "
            f"give {result['exact']:.0f}; power at {n} is {result['power']:.4f}",
        ),
        practice.Check(
            "FINDING: alpha and power are accepted and ignored",
            result["ignored"] == (n, n),
            f"power=0.9 -> {result['ignored'][0]}, alpha=0.01 -> {result['ignored'][1]}",
        ),
        practice.Check(
            "FINDING: the lesson's own 5%-lift run stops at 72% of that n",
            round(demo[0] / n, 2) == 0.72 and demo[2] > 1.96 and demo[3] is None,
            f"A gets {demo[0]} of {n}; the fixed z at its end is {demo[2]:.2f}, the "
            f"sequential boundary {boundary(demo[1]):.2f}, and it never stops",
        ),
        practice.Check(
            "FINDING: at the 80%-power horizon the sequential rule stops in 8 of 40 seeds",
            result["stops"] == 8
            and round(boundary(2 * n), 2) == 4.35
            and ten == (159731, 106300),
            f"{result['stops']}/{SEEDS} stops; boundary {boundary(2 * n):.2f} against an "
            f"expected z of {result['drift']:.2f}; the 10% demo stops at {ten[0]} vs a "
            f"fixed {ten[1]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
