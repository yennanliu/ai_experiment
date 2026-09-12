"""Exercise 1 — the second softmax guards a row this mask cannot produce.

    **Easy.** Run `code/main.py` and verify the causal attention matrix is
    lower-triangular after softmax. Spot-check: row 3 should have weights only
    in columns 0-3.

Reading of the exercise: "verify" is read as exact rather than visual -- the
upper triangle is checked against `0.0` and not against a tolerance, and the
row sums against `1.0` -- and the spot-check is extended to ask *why* the module
ships two softmaxes when one row of the matrix is already degenerate.

**ANSWER: verified exactly where exactness is available.** Row 3 is
`[0.269, 0.232, 0.439, 0.060, 0.0, 0.0]`, every upper-triangle entry is `0.0` and
not an underflow, and the row sums are within 1.1e-16 of 1. The zeros are exact
because `apply_softmax_row` writes `0.0` for a `-inf` entry rather than
exponentiating it; the sums are not, because adding `i+1` floats is not.

**FINDING: row 0 is exactly one-hot, so its entropy is exactly 0.** Position 0
can attend only to itself: `[1.0, 0, 0, 0, 0, 0]`. The first token of a causal
sequence takes nothing from the sequence, which is why a prompt is not optional
and why row 0 is the only row whose "attention" is fixed before any weights are
learned.

**FINDING: the module's own `softmax` gives bit-identical answers here.** Run on
every masked row it agrees with `apply_softmax_row` to **0.0** -- `math.exp(-inf)`
is already `0.0`, so the ordinary path handles `-inf` correctly. The two differ on
exactly one input: a row that is entirely `-inf`, where `softmax` computes
`-inf - -inf` and returns **nan** while `apply_softmax_row` returns zeros.

**FINDING: `causal_mask` cannot produce that row.** Its diagonal is `0.0` at
every position, so every row has at least one finite entry. The guard is dead
code for the mask the lesson ships -- it exists for *padding* masks, which this
lesson does not have, and the softmax that would actually meet one is the
unguarded one.

**CONTROL: what is verified is the mask, not attention.** `demo_causal_mask`
feeds `rng.gauss(0, 1)` straight in: no `Q`, no `K`, no `1/sqrt(dk)`. The
triangularity is a property of the mask and would hold for any scores at all,
which is what makes it checkable in one line.

Structure: `attention` rebuilds `demo_causal_mask`'s matrix; `entropy` reads how
far each row can spread; `disagreement` compares the two softmaxes; `starvation`
probes the row neither of them meets here.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "07-gpt-causal-language-modeling"
SIZE, SEED, ROW = 6, 42, 3


def attention(ref, size=SIZE, seed=SEED):
    """demo_causal_mask's own matrix: raw gaussian scores, masked, row-softmaxed."""
    rng = random.Random(seed)
    raw = [[rng.gauss(0, 1) for _ in range(size)] for _ in range(size)]
    masked = ref.attention_scores_with_mask(raw, ref.causal_mask(size))
    return masked, [ref.apply_softmax_row(row) for row in masked]


def entropy(row):
    """Shannon entropy in nats; a row pinned to one position reads exactly 0."""
    return -sum(p * math.log(p) for p in row if p > 0)


def disagreement(ref, rows):
    """Worst gap between the module's two softmaxes over these rows."""
    return max(abs(a - b) for row in rows
               for a, b in zip(ref.softmax(row), ref.apply_softmax_row(row)))


def upper(attn):
    """Every entry above the diagonal, as a set -- {0.0} is the claim."""
    return {attn[i][j] for i in range(SIZE) for j in range(i + 1, SIZE)}


def starvation(ref):
    """What each softmax does with an all -inf row, and whether this mask can make one."""
    starved = [float("-inf")] * SIZE
    mask = ref.causal_mask(SIZE)
    return {
        "plain_nan": all(value != value for value in ref.softmax(starved)),
        "guarded": ref.apply_softmax_row(starved),
        "starvable": any(all(v == float("-inf") for v in row) for row in mask),
        "diagonal": [mask[i][i] for i in range(SIZE)],
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    masked, attn = attention(ref)
    return {
        "row": attn[ROW], "triangle": upper(attn), "first": attn[0],
        "worst_sum": max(abs(sum(row) - 1.0) for row in attn),
        "entropy": [entropy(row) for row in attn],
        "ceiling": [math.log(i + 1) for i in range(SIZE)],
        "gap": disagreement(ref, masked), **starvation(ref),
        "shown": [round(v, 3) for v in attn[ROW]],
        "spread": [round(entropy(row), 2) for row in attn],
        "bounds": [round(math.log(i + 1), 2) for i in range(SIZE)],
        "bounded": all(entropy(row) <= math.log(i + 1) + 1e-12 for i, row in enumerate(attn)),
    }


def verify(result):
    row, spread = result["row"], result["entropy"]
    exact = result["triangle"] == {0.0} and result["worst_sum"] < 1e-15
    trailing = row[ROW + 1:] == [0.0] * (SIZE - ROW - 1)
    dead = result["guarded"] == [0.0] * SIZE and not result["starvable"]
    return [
        practice.Check(
            "ANSWER: lower-triangular exactly -- the zeros are 0.0, not underflow",
            exact and trailing,
            f"row {ROW} is {result['shown']} -- weight in columns 0-{ROW} and nowhere "
            f"else -- every upper-triangle entry is in {result['triangle']}, and the {SIZE} row "
            f"sums are within {result['worst_sum']:.1e} of 1. The zeros are exact because "
            "apply_softmax_row writes 0.0 for a -inf entry instead of exponentiating it; the row "
            "sums are not, because summing i+1 floats is not",
        ),
        practice.Check(
            "FINDING: row 0 is exactly one-hot, with entropy exactly 0",
            result["first"][0] == 1.0 and spread[0] == 0.0,
            f"position 0 can attend only to itself: {result['first']}. Its entropy is "
            f"{spread[0]:.1f} against the {spread[SIZE - 1]:.3f} of the last row. The first token "
            "of a causal sequence takes nothing from the sequence -- which is why a prompt is not "
            "optional, and why row 0 is fixed before any weight is learned",
        ),
        practice.Check(
            "FINDING: the module's own softmax gives bit-identical answers on every causal row",
            result["gap"] == 0.0,
            f"softmax and apply_softmax_row agree to {result['gap']} across all {SIZE} masked "
            "rows, because math.exp(-inf) is already 0.0 and the ordinary path handles a -inf "
            "entry correctly. The two implementations differ on exactly one input",
        ),
        practice.Check(
            "FINDING: that one input is a row causal_mask cannot produce",
            result["plain_nan"] and dead and result["diagonal"] == [0.0] * SIZE,
            f"on an all -inf row softmax computes -inf - -inf and returns nan, where "
            f"apply_softmax_row returns {result['guarded'][:3]}... But causal_mask's diagonal is "
            f"{result['diagonal']}, so every row keeps a finite entry and no such row exists here. "
            "The guard is for padding masks, which this lesson does not have",
        ),
        practice.Check(
            "CONTROL: what is verified is the mask, not attention",
            result["bounded"],
            f"demo_causal_mask feeds rng.gauss(0, 1) straight in -- no Q, no K, no 1/sqrt(dk) -- "
            f"so triangularity is a property of the mask and holds for any scores at all. Row "
            f"entropies {result['spread']} stay under their ln(i+1) ceilings "
            f"{result['bounds']}, which is all the structure there is",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
