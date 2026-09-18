"""Exercise 4 — dropping half the tokens leaves the sparsity where it was.

    Implement fractional patch dropping: given a packed sequence, drop 50% of
    tokens uniformly at random, and update the block-diagonal mask accordingly.
    Measure the mask's sparsity change.

Reading of the exercise: "measure the mask's sparsity change" is taken as the
question rather than as a formality, and the measurement is run over 200 seeds
rather than one, because a single draw cannot distinguish a real change from
sampling noise. Sparsity here is the fraction of the N x N matrix that is
attended, computed from the surviving per-image counts through the lesson's own
`pack_batch`.

**ANSWER: it does not change.** The mask is 29.591% dense before the drop and
**29.609%** on average across 200 uniform 50% drops -- a shift of 0.019 points
inside a seed-to-seed range of **28.36% to 30.75%**. Density is
`Sum(n^2) / (Sum n)^2`, which is scale-free: halve every block and it cancels.

**ANSWER: what does change is the absolute cost, by 4x.** Attended cells fall
from **1,977,329** to **493,968** under an exact stratified halving -- a ratio
of **4.003** -- because the count is quadratic in a length that was halved.
Dropping half the tokens buys three quarters of the attention.

**FINDING: uniform-across-the-pack and per-image dropping differ only in
variance.** Halving each image exactly gives **29.592%**, a hundredth of a point
from the undropped mask. Drawing uniformly over the packed sequence gives the
same answer in expectation and a **2.38-point** spread, because a short block
can be over- or under-sampled. The stratified version is free and removes the
only difference between them.

**FINDING: so sparsity is the wrong thing to measure here.** It is invariant to
the operation being performed, and the two numbers that do move -- token count
and attended cells -- move by 2x and 4x. A report that the sparsity held at 29.6%
is a report that the drop was uniform, not that it was cheap.

Structure: `density` is the closed measure, `drop_uniform` samples survivors
across the whole packed sequence, `drop_stratified` halves each block exactly,
and `SEEDS` is the number of draws averaged.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "06-any-resolution-patch-n-pack"
LENGTHS = (256, 576, 729, 1024)
KEEP, SEEDS = 0.5, 200


def as_images(ref, lengths):
    return [ref.Image(f"img{i}", 1, length) for i, length in enumerate(lengths)]


def density(ref, lengths):
    """Attended fraction of the N x N matrix, via the lesson's own pack_batch."""
    pack = ref.pack_batch(as_images(ref, lengths), 1)
    return pack.mask_nonzero / pack.mask_size * 100, pack.mask_nonzero


def drop_uniform(lengths, seed, keep=KEEP):
    """Each packed position survives independently -- the exercise's 'uniformly at random'."""
    rng = random.Random(seed)
    return [sum(1 for _ in range(length) if rng.random() < keep) for length in lengths]


def drop_stratified(lengths, keep=KEEP):
    """The same fraction taken from every image, exactly."""
    return [int(length * keep) for length in lengths]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    before, before_cells = density(ref, LENGTHS)
    uniform = [density(ref, drop_uniform(LENGTHS, seed))[0] for seed in range(SEEDS)]
    stratified, stratified_cells = density(ref, drop_stratified(LENGTHS))
    return {
        "before": round(before, 3), "before_cells": before_cells,
        "uniform_mean": round(statistics.fmean(uniform), 3),
        "uniform_min": round(min(uniform), 2), "uniform_max": round(max(uniform), 2),
        "uniform_spread": round(max(uniform) - min(uniform), 2),
        "shift": round(statistics.fmean(uniform) - before, 3),
        "seeds": len(uniform),
        "stratified": round(stratified, 3), "stratified_cells": stratified_cells,
        "stratified_shift": round(abs(stratified - before), 3),
        "cell_ratio": round(before_cells / stratified_cells, 3),
        "tokens_before": sum(LENGTHS), "tokens_after": sum(drop_stratified(LENGTHS)),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the sparsity does not change -- 29.591% before, 29.609% after",
            all([result["before"] == 29.591, result["uniform_mean"] == 29.609,
                 abs(result["shift"]) <= 0.02, result["seeds"] == SEEDS,
                 result["uniform_min"] == 28.36, result["uniform_max"] == 30.75]),
            f"across {result['seeds']} uniform {KEEP:.0%} drops the density averages "
            f"{result['uniform_mean']}% against {result['before']}% before -- a shift of "
            f"{result['shift']} points inside a range of {result['uniform_min']}% to "
            f"{result['uniform_max']}%. Density is Sum(n^2)/(Sum n)^2, which is scale-free: "
            "halve every block and it cancels",
        ),
        practice.Check(
            "ANSWER: what does change is the absolute cost, by 4x",
            all([result["before_cells"] == 1_977_329,
                 result["stratified_cells"] == 493_968,
                 result["cell_ratio"] == 4.003,
                 result["tokens_after"] * 2 == result["tokens_before"] - 1]),
            f"attended cells fall from {result['before_cells']:,} to "
            f"{result['stratified_cells']:,}, a ratio of {result['cell_ratio']}, while the "
            f"sequence goes {result['tokens_before']:,} -> {result['tokens_after']:,}. The "
            "count is quadratic in a length that was halved, so dropping half the tokens "
            "buys three quarters of the attention",
        ),
        practice.Check(
            "FINDING: uniform and per-image dropping differ only in variance",
            all([result["stratified"] == 29.592, result["stratified_shift"] <= 0.01,
                 result["uniform_spread"] == 2.38]),
            f"halving each image exactly gives {result['stratified']}%, "
            f"{result['stratified_shift']} from the undropped mask; drawing uniformly over "
            f"the packed sequence gives the same answer in expectation and a "
            f"{result['uniform_spread']}-point spread, because a short block can be over- or "
            "under-sampled. The stratified version is free and removes the difference",
        ),
        practice.Check(
            "FINDING: so sparsity is the wrong thing to measure here",
            all([abs(result["stratified"] - result["before"]) < 0.01,
                 result["cell_ratio"] > 3.9]),
            f"the quantity the exercise asks about is invariant to the operation it asks "
            f"about -- {result['before']}% -> {result['stratified']}% -- while the two that "
            f"do move, token count and attended cells, move by 2x and "
            f"{result['cell_ratio']}x. A report that sparsity held at 29.6% is a report that "
            "the drop was uniform, not that it was cheap",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
