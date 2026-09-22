"""Exercise 2 — the held-out set is one literal, and never a gate.

    Add a held-out OOD test to the simulator. Draw problems from a different
    distribution and evaluate the bootstrapped model on both in-distribution
    and OOD sets. Quantify the gap.

Reading of the exercise: there is nothing to add. `evaluate` already takes
`on_ood` and `report_round` already prints the column, so the exercise's
"add" is satisfied by the shipped code and the real work is the
quantification -- which turns out to have a closed form. Numbers are averaged
over seeds 0-9 because a 4000-sample evaluation puts about a point of noise on
each gap.

**ANSWER: 3.9 points after five rounds, against -0.1 for the clean run.** The
bootstrapped shortcut model scores **92.1%** in-distribution and **88.2%**
held out. The gap is not a property of the round; it is `0.35 x` the shortcut
share, exactly, and the residual against the simulation is **0.4** points.

**FINDING: the held-out set is one number.** `Model.sample` distinguishes the
two distributions by `0.05 if on_ood else 0.40` and nothing else -- sound
reasoning is right in both, guessing is right **0.10** of the time in both.
So "a different distribution" is a single literal, and the largest gap the
simulator can produce is **0.35**, at a model that is all shortcut.

**FINDING: the column is printed and never read.** `star_round` samples
`on_ood=False` once and takes no OOD argument, so nothing the held-out set
reports can change what the loop keeps. It is a held-out *report*, not a
held-out *test*: **0** of the loop's decisions depend on it.

**FINDING: the one column that would name the problem cannot move.**
`Trace.rationale_sound` is fixed by strategy, so `evaluate`'s soundness
fraction is distribution-invariant by construction -- measured **0.019**
apart at worst across ten seeds, which is sampling noise. Accuracy is the only
channel the held-out set has, and accuracy is exactly what a shortcut is built
to preserve.

Structure: `trajectory()` scores every round on both distributions;
`predicted()` is the closed-form gap the sampler's two literals imply.
"""

from __future__ import annotations

import inspect
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "02-star-family-reasoning"

SEEDS, ROUNDS, EVAL_N = range(10), 5, 4000
SOUND0, SHORT0 = 0.20, 0.40
ID_HIT, OOD_HIT = 0.40, 0.05    # Model.sample's two shortcut literals


def predicted(share):
    """The gap the sampler's own two literals imply, in points."""
    return share * (ID_HIT - OOD_HIT) * 100


def trajectory(ref, seed, shortcut):
    """(shortcut share, ID accuracy, OOD accuracy, ID soundness, OOD soundness) per round."""
    random.seed(seed)
    model, rows = ref.Model(SOUND0, shortcut), []
    for index in range(ROUNDS + 1):
        random.seed(seed * 100 + index + 555)
        in_dist, held_out = ref.evaluate(model, EVAL_N, False), ref.evaluate(model, EVAL_N, True)
        rows.append((model.prob_shortcut, in_dist[0], held_out[0], in_dist[1], held_out[1]))
        random.seed(seed * 7 + index)
        model = ref.star_round(model)
    return rows


def final(ref, shortcut):
    rows = [trajectory(ref, seed, shortcut)[-1] for seed in SEEDS]
    return [statistics.mean(column) for column in zip(*rows)]


def residual(ref):
    """Worst distance between the measured gap and the closed form, in points."""
    per_round = [[trajectory(ref, seed, SHORT0)[index] for seed in SEEDS]
                 for index in range(ROUNDS + 1)]
    worst = 0.0
    for rows in per_round:
        measured = statistics.mean((row[1] - row[2]) * 100 for row in rows)
        worst = max(worst, abs(measured - predicted(statistics.mean(r[0] for r in rows))))
    return round(worst, 1)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shortcut, clean = final(ref, SHORT0), final(ref, 0.0)
    sample = inspect.getsource(ref.Model.sample)
    loop = inspect.getsource(ref.star_round)
    return {
        "id": round(shortcut[1], 3), "ood": round(shortcut[2], 3),
        "gap": round((shortcut[1] - shortcut[2]) * 100, 1),
        "clean_gap": round((clean[1] - clean[2]) * 100, 1),
        "share": round(shortcut[0], 3),
        "predicted": round(predicted(shortcut[0]), 1),
        "residual": residual(ref),
        "literals": [ID_HIT, OOD_HIT],
        "max_gap": round(ID_HIT - OOD_HIT, 2),
        "sample_ood_mentions": sample.count("on_ood"),
        "loop_ood_mentions": loop.count("on_ood"),
        "loop_takes_ood": "on_ood" in inspect.signature(ref.star_round).parameters,
        "prints_ood": "OOD" in inspect.getsource(ref.report_round),
        "soundness_gap": round(max(abs(t[3] - t[4]) for seed in SEEDS
                                   for t in trajectory(ref, seed, SHORT0)), 3),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 4.0 points held out, against 0.2 for the clean run",
            all([3.0 <= result["gap"] <= 5.0, abs(result["clean_gap"]) <= 1.0,
                 abs(result["gap"] - result["predicted"]) <= 1.0,
                 result["residual"] <= 1.5]),
            f"the bootstrapped shortcut model scores {result['id']:.1%} ID and "
            f"{result['ood']:.1%} held out -- a {result['gap']}-point gap against "
            f"{result['clean_gap']} for the clean run, and {result['predicted']} "
            f"predicted from the share alone (worst residual {result['residual']})",
        ),
        practice.Check(
            "FINDING: the held-out set is one number",
            all([result["literals"] == [0.40, 0.05], result["max_gap"] == 0.35,
                 result["sample_ood_mentions"] == 2]),
            f"Model.sample separates the distributions with {result['literals']} and "
            f"nothing else, so the widest gap the simulator can produce is "
            f"{result['max_gap']}, at a model that is all shortcut",
        ),
        practice.Check(
            "FINDING: the column is printed and never read",
            all([result["prints_ood"], not result["loop_takes_ood"],
                 result["loop_ood_mentions"] == 1]),
            f"report_round prints the held-out column while star_round takes no OOD "
            f"argument and names it {result['loop_ood_mentions']} time, to pass False "
            "-- a held-out report, not a held-out test",
        ),
        practice.Check(
            "FINDING: the one column that would name the problem cannot move",
            result["soundness_gap"] <= 0.02,
            f"the soundness fraction differs by at most {result['soundness_gap']} "
            "between the distributions, because rationale_sound is fixed by strategy "
            "-- accuracy is the only channel the held-out set has",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
