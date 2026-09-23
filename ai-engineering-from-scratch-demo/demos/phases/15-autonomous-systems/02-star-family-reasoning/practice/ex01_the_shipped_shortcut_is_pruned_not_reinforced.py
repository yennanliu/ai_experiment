"""Exercise 1 — the shipped shortcut is pruned, not reinforced.

    Run the simulator. Set the shortcut frequency to zero, then to 0.4. How
    much does final accuracy diverge between the two runs, even though both
    hit >90% on the training distribution?

Reading of the exercise: "final accuracy" has to mean both distributions,
because the divergence the lesson is pointing at is the held-out one. Runs are
seeded 0-9 and averaged, since one seed of a 1000-sample bootstrap moves the
third digit; the assertions are bands wide enough to survive the seeding and
narrow enough to be wrong if the behaviour changes.

**ANSWER: 5.8 points in-distribution and 9.6 out of it.** After 5 rounds the
clean run reaches **97.7%** ID and **97.6%** OOD; the shortcut run reaches
**92.0%** ID and **87.9%** OOD. Both clear the 90% the exercise promises, and
the held-out divergence is **1.7x** the training one.

**FINDING: the shortcut does not survive the loop.** Its share falls from
**0.400** to **0.114** over the five rounds and to **0.011** by round ten,
monotonically. The lesson's own headline says the opposite -- "Scenario B
climbs on ID while OOD collapses ... the shortcut gets reinforced" -- and OOD
ends at 87.9%, five points down, with the shortcut nearly gone.

**FINDING: `star_round`'s rule can be replayed in closed form, and it
explains why.** Taking the same update in expectation reproduces the
simulation to within **0.002** at the shipped hit rate. The filter keeps
correct answers, and the shipped shortcut is *less accurate* than sound
reasoning -- 0.40 against 1.00 -- so it is selected against on accuracy, never
on soundness. Sweeping the hit rate, the share only holds at **0.71**.

**FINDING: the headline describes a hit rate the module does not ship.** At a
shortcut that is right in-distribution as often as sound reasoning -- which is
what a shortcut *is* -- the same rule grows the share from 0.40 to **0.660**.
That is the reinforcement the lesson claims; it needs a number **0.60** above
the one in `Model.sample`.

Structure: `bootstrapped()` runs one seeded scenario end to end; `replay()` is
`star_round`'s update with the shortcut's hit rate exposed as a knob.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "02-star-family-reasoning"

SEEDS, ROUNDS, EVAL_N = range(10), 5, 2000
SOUND0, SHORT0 = 0.20, 0.40     # the lesson's own Scenario B starting point
ALPHA, RANDOM_HIT, SHIPPED_HIT = 0.6, 0.10, 0.40   # mixing weight; guess and shortcut hits


def bootstrapped(ref, seed, shortcut):
    """One seeded scenario: bootstrap, then score on both distributions."""
    random.seed(seed)
    model = ref.Model(SOUND0, shortcut)
    for _round in range(ROUNDS):
        model = ref.star_round(model)
    random.seed(seed + 9973)
    return (model.prob_shortcut, ref.evaluate(model, EVAL_N, False)[0],
            ref.evaluate(model, EVAL_N, True)[0])


def averaged(ref, shortcut):
    rows = zip(*(bootstrapped(ref, seed, shortcut) for seed in SEEDS))
    return [round(statistics.mean(column), 4) for column in rows]


def replay(hit, rounds=ROUNDS, sound=SOUND0, shortcut=SHORT0):
    """`star_round`'s own update in expectation, with the hit rate as a knob."""
    for _ in range(rounds):
        kept = (sound, shortcut * hit, (1 - sound - shortcut) * RANDOM_HIT)
        total = sum(kept)
        sound, shortcut = (ALPHA * kept[0] / total + (1 - ALPHA) * sound,
                           ALPHA * kept[1] / total + (1 - ALPHA) * shortcut)
        scale = max(sound + shortcut, 1.0)
        sound, shortcut = sound / scale, shortcut / scale
    return sound, shortcut


def break_even(low=SHIPPED_HIT, high=1.0):
    """The hit rate at which five rounds leave the shortcut share where it began."""
    for _ in range(40):
        mid = (low + high) / 2
        low, high = (mid, high) if replay(mid)[1] < SHORT0 else (low, mid)
    return (low + high) / 2


def ten_round_share(ref, seed=0):
    random.seed(seed)
    model, shares = ref.Model(SOUND0, SHORT0), [SHORT0]
    for _ in range(10):
        model = ref.star_round(model)
        shares.append(round(model.prob_shortcut, 3))
    return shares


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    clean, shortcut = averaged(ref, 0.0), averaged(ref, SHORT0)
    shares, replayed = ten_round_share(ref), round(replay(SHIPPED_HIT)[1], 3)
    return {
        "clean": clean[1:], "shortcut": shortcut[1:],
        "id_divergence": round((clean[1] - shortcut[1]) * 100, 1),
        "ood_divergence": round((clean[2] - shortcut[2]) * 100, 1),
        "both_over_90": min(clean[1], shortcut[1]) > 0.90,
        "final_share": shortcut[0], "share_round10": shares[-1],
        "monotone": all(b <= a for a, b in zip(shares[1:], shares[2:])),
        "replayed": replayed,
        "replay_error": round(abs(replayed - shortcut[0]), 3),
        "break_even": round(break_even(), 2),
        "at_parity": round(replay(1.0)[1], 3), "shipped_hit": SHIPPED_HIT,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 5.8 points in-distribution, 9.6 out of it",
            all([result["both_over_90"], 5.0 <= result["id_divergence"] <= 6.5,
                 9.0 <= result["ood_divergence"] <= 10.5]),
            f"clean {result['clean']} ID/OOD against {result['shortcut']} -- both over "
            f"90% on the training distribution, diverging {result['id_divergence']} "
            f"points there and {result['ood_divergence']} on the held-out set",
        ),
        practice.Check(
            "FINDING: the shortcut does not survive the loop",
            all([result["monotone"], 0.09 <= result["final_share"] <= 0.14,
                 result["share_round10"] <= 0.03]),
            f"the share falls from {SHORT0} to {result['final_share']} by round 5 and "
            f"{result['share_round10']} by round 10, monotonically -- the lesson's "
            "headline says it gets reinforced",
        ),
        practice.Check(
            "FINDING: star_round's own rule, replayed, explains the pruning",
            all([result["replay_error"] <= 0.02, 0.10 <= result["replayed"] <= 0.13,
                 0.68 <= result["break_even"] <= 0.75]),
            f"the closed-form replay gives {result['replayed']} against the simulated "
            f"{result['final_share']}, within {result['replay_error']}; the filter "
            f"selects on accuracy, and the share only holds once the shortcut is right "
            f"{result['break_even']} of the time in-distribution",
        ),
        practice.Check(
            "FINDING: the headline describes a hit rate the module does not ship",
            all([result["at_parity"] >= 0.60, result["shipped_hit"] == 0.40,
                 result["at_parity"] > result["final_share"] * 4]),
            f"a shortcut as accurate in-distribution as sound reasoning grows the share "
            f"to {result['at_parity']}, against {result['final_share']} at the shipped "
            f"{result['shipped_hit']} -- reinforcement needs a number this module lacks",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
