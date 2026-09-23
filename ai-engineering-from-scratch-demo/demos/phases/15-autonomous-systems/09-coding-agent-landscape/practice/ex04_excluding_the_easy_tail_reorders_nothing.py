"""Exercise 4 — excluding the easy tail reorders nothing.

    SWE-bench Verified has 161 single-file, 1-2 line tasks. Construct a score
    that excludes them. How does the leaderboard shuffle?

Reading of the exercise: "how does it shuffle" presumes it does. Constructing
the score first and asking the question of the formula gives a different
answer, so the score is written down, its shape is checked, and only then is
it applied to the leaderboard band the lesson quotes.

**ANSWER: `H = (500V - 161e)/339`, and at `e = 1` it reorders nothing.**
Excluding the **161** easy tasks from the **500** leaves **339**; with every
system solving the easy tail, `H` is an affine, increasing function of the
published score, so the ranking is preserved exactly. The lesson's 2026 band
of **70-80%** maps to **55.8-70.5%** -- lower numbers, identical order.

**FINDING: the gaps widen by a fixed factor.** `dH/dV` is `500/339` =
**1.475**, so a 2-point difference on Verified becomes a **2.9**-point
difference on the hard subset. The score makes the leaderboard look more
decisive while changing none of its decisions.

**FINDING: a shuffle needs easy-tail rates nobody publishes.** Two systems
swap only when the easy-tail gap exceeds `(500/161)` = **3.11** times the
Verified gap: at 2 points apart on Verified, the lower-ranked system has to
solve **6.2** more points of the easy tail. That number is not on any
leaderboard, so the reshuffle the exercise asks about is unobservable from the
published column.

**FINDING: this is not SWE-bench Pro.** Pro reports **23-59%** for frontier
systems where the exclusion score predicts **55.8-70.5%** from the same band
-- an overlap of only **3.2** points. Removing the easy tail and raising the
difficulty floor are different operations, and the second is the one whose
result the lesson says production resembles.

Structure: `hard_score()` is the constructed metric; `swap_gap()` is the
easy-tail difference two systems need before it reorders them.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "09-coding-agent-landscape"

TOTAL, EASY = 500, 161          # SWE-bench Verified, and its 1-2 line tail
HARD = TOTAL - EASY
VERIFIED_BAND = (0.70, 0.80)    # the 2026 band the lesson quotes
PRO_BAND = (0.23, 0.59)         # SWE-bench Pro, same systems


def hard_score(verified, easy_rate=1.0):
    """Verified score with the easy tail removed, at a given easy-tail solve rate."""
    return round((TOTAL * verified - EASY * easy_rate) / HARD, 4)


def swap_gap(verified_gap):
    """The easy-tail rate difference that would reverse a given Verified gap."""
    return round(verified_gap * TOTAL / EASY, 4)


def monotone(easy_rate=1.0, steps=101):
    """Is the constructed score increasing in the published one?"""
    values = [hard_score(index / (steps - 1), easy_rate) for index in range(steps)]
    return all(a < b for a, b in zip(values, values[1:]))


def overlap(first, second):
    low, high = max(first[0], second[0]), min(first[1], second[1])
    return round(max(0.0, high - low), 4)


def solve():
    parity.load_reference(PHASE, LESSON, "main")     # D5: the lesson is the source
    band = (hard_score(VERIFIED_BAND[0]), hard_score(VERIFIED_BAND[1]))
    return {
        "total": TOTAL, "easy": EASY, "hard": HARD,
        "easy_share": round(EASY / TOTAL, 3),
        "band": list(band),
        "verified_band": list(VERIFIED_BAND),
        "monotone": monotone(),
        "slope": round(TOTAL / HARD, 3),
        "widened": round((band[1] - band[0]) / (VERIFIED_BAND[1] - VERIFIED_BAND[0]), 3),
        "two_point_gap": round(hard_score(0.80) - hard_score(0.78), 3),
        "swap_ratio": round(TOTAL / EASY, 2),
        "swap_at_two_points": swap_gap(0.02),
        "pro_band": list(PRO_BAND),
        "pro_overlap": overlap(band, PRO_BAND),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: (500V - 161e)/339, and at e = 1 it reorders nothing",
            all([result["hard"] == 339, result["easy_share"] == 0.322,
                 result["monotone"], result["band"] == [0.5575, 0.705]]),
            f"excluding the {result['easy']} easy tasks from {result['total']} leaves "
            f"{result['hard']}; the score is increasing in the published one, so the "
            f"{result['verified_band']} band becomes {result['band']} with the ranking "
            "preserved exactly",
        ),
        practice.Check(
            "FINDING: the gaps widen by a fixed factor",
            all([result["slope"] == 1.475, result["widened"] == 1.475,
                 result["two_point_gap"] == 0.029]),
            f"dH/dV is {result['slope']}, so the band widens by exactly that and a "
            f"2-point Verified difference becomes {result['two_point_gap']:.3f} on the "
            "hard subset -- more decisive-looking, identically ordered",
        ),
        practice.Check(
            "FINDING: a shuffle needs easy-tail rates nobody publishes",
            all([result["swap_ratio"] == 3.11, result["swap_at_two_points"] == 0.0621]),
            f"a swap needs an easy-tail gap of {result['swap_ratio']}x the Verified gap "
            f"-- {result['swap_at_two_points']:.1%} at two points apart -- and no "
            "leaderboard publishes the per-system easy-tail rate",
        ),
        practice.Check(
            "FINDING: this is not SWE-bench Pro",
            all([result["pro_band"] == [0.23, 0.59], result["pro_overlap"] == 0.0325]),
            f"Pro reports {result['pro_band']} where the exclusion score predicts "
            f"{result['band']} from the same systems -- {result['pro_overlap']:.1%} of "
            "overlap, so removing the tail and raising the floor are different "
            "operations",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
