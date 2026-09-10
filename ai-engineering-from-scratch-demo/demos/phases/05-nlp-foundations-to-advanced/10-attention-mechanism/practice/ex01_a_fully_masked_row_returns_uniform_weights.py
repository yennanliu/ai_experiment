"""Exercise 1 — a fully masked row returns uniform weights.

    **Easy.** Implement `softmax` masking so padding tokens in the encoder get
    attention weight zero. Test on a batch with variable-length sequences.

Reading of the exercise: the implementation is one line -- replace a padded
position's score with a large negative sentinel before the softmax -- and it is
exactly right on the case the exercise names. On a five-position batch with two
padded, the padded weights come back 0.0 and 0.0, and the resulting context
matches attention over the real prefix to 1.11e-16. The interesting part is the
two neighbouring cases the test the exercise describes does not reach.

A row where every position is padding returns uniform weights. `softmax`
subtracts the maximum, so five identical sentinels become five identical
exponentials and the function returns 0.2 five times, summing to 1.0, with no
error raised. That row is a padded sequence in a batch -- an empty source, or a
row shorter than the batch's minimum after truncation -- and it produces a
confident average over nothing.

Using -inf as the sentinel fixes the first case by breaking the second. It gives
exact zeros wherever anything survives, and on a fully masked row every score
becomes -inf - -inf = nan, which propagates silently into the context vector.
The two obvious sentinels fail on the same input in opposite directions.

The third variant is the one that looks safest and is not: zeroing the weights
after the softmax instead of the scores before it. That leaves the surviving
weights summing to 0.7338 rather than 1, so the context is the right direction
scaled by the retained mass -- the ratio is 0.7338 in every dimension, identical,
which is precisely why it survives a spot check on the numbers.

Structure: `masked` applies a sentinel before the lesson's own `softmax`;
`post_hoc` is the zero-afterwards variant; `context` is the weighted sum the
lesson's `dot_attention` performs, factored out so all four variants can be
scored against attention over the unpadded prefix.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "10-attention-mechanism"

ENCODER = ([1.0, 0.0, 0.2], [0.5, 0.5, 0.1], [0.1, 0.9, 0.3], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
DECODER = [0.9, 0.1, 0.2]
REAL, SENTINEL = 3, -1e9


def masked(ref, scores, keep, sentinel=SENTINEL) -> list:
    return ref.softmax([s if k else sentinel for s, k in zip(scores, keep)])


def post_hoc(ref, scores, keep) -> list:
    """Zero the weights after the softmax rather than the scores before it."""
    return [w if k else 0.0 for w, k in zip(ref.softmax(scores), keep)]


def context(weights, states) -> list:
    width = len(states[0])
    return [sum(w * h[d] for w, h in zip(weights, states)) for d in range(width)]


def sentinels(ref, scores, positions) -> dict:
    """A fully masked row under each sentinel, and a partially masked one under -inf."""
    blank = [0] * positions
    try:
        infinite = [None if math.isnan(w) else round(w, 6)
                    for w in masked(ref, scores, blank, float("-inf"))]
    except Exception as exc:                        # pragma: no cover - sentinel guard
        infinite = type(exc).__name__
    uniform = masked(ref, scores, blank)
    return {"all_masked": [round(w, 6) for w in uniform],
            "all_masked_total": round(sum(uniform), 6), "all_masked_inf": infinite,
            "uniform_context": [round(x, 6) for x in context(uniform, ENCODER)],
            "partial_inf": [round(w, 6) for w in
                            masked(ref, scores, [1] + [0] * (positions - 1), float("-inf"))]}


def variants(ref, scores, keep, prefix) -> dict:
    """The correct masking and the zero-afterwards variant, both scored against the prefix."""
    correct, dropped = masked(ref, scores, keep), post_hoc(ref, scores, keep)
    ratio = [round(a / b, 6) for a, b in zip(context(dropped, ENCODER), prefix)]
    return {"masked": [round(w, 6) for w in correct],
            "zeroed_padding": set(correct[REAL:]) == {0.0},
            "drift": max(abs(a - b) for a, b in zip(context(correct, ENCODER), prefix)),
            "post_hoc_total": round(sum(dropped), 6), "post_hoc_ratio": ratio,
            "one_ratio": len(set(ratio)) == 1}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    scores = [ref.dot(DECODER, h) for h in ENCODER]
    keep = [1] * REAL + [0] * (len(ENCODER) - REAL)
    prefix = ref.dot_attention(DECODER, list(ENCODER[:REAL]))[0]
    width = len(ENCODER[0])
    return dict(
        sentinels(ref, scores, len(ENCODER)) | variants(ref, scores, keep, prefix),
        real=REAL, positions=len(ENCODER),
        unmasked=[round(w, 4) for w in ref.softmax(scores)],
        padding_mass=round(sum(ref.softmax(scores)[REAL:]), 4),
        plain_mean=[round(sum(h[d] for h in ENCODER) / len(ENCODER), 6) for d in range(width)],
        empty_padding=set(x for h in ENCODER[REAL:] for x in h) == {0.0})


def verify(result):
    total, ratio = result["all_masked_total"], result["post_hoc_ratio"]
    uniform = result["all_masked"]
    return [
        practice.Check(
            "ANSWER: a sentinel before the softmax gives exact zeros and the prefix's context",
            result["zeroed_padding"] and result["drift"] < 1e-15,
            f"with {result['real']} real positions of {result['positions']}, masked weights are "
            f"{result['masked']} -- the padded entries are exactly 0.0 -- and the context matches "
            f"`dot_attention` over the unpadded prefix to {result['drift']:.3g}. Unmasked, the same "
            f"batch gives {result['unmasked']}, sending {result['padding_mass']:.1%} of the "
            f"attention mass to two all-zero vectors"),
        practice.Check(
            "FINDING: a row where everything is padding comes back uniform, not zero",
            set(uniform) == {round(1 / result["positions"], 6)} and total == 1.0,
            f"`softmax` subtracts the maximum, so {result['positions']} identical sentinels become "
            f"{result['positions']} identical exponentials: the weights are {uniform}, "
            f"summing to {total}. No error is raised, and the context is a confident average over "
            f"nothing at all"),
        practice.Check(
            "MECHANISM: the answer it returns is the plain mean of every state, padding included",
            result["uniform_context"] == result["plain_mean"],
            f"uniform weights make the context {result['uniform_context']}, which is exactly the "
            f"unweighted mean of all {result['positions']} encoder states, {result['plain_mean']} -- "
            f"padding vectors included. The case is not hypothetical either: it is a batch row "
            f"whose source is empty, or one truncated below the batch minimum. Every length above "
            f"zero is covered by the sentinel and zero is not"),
        practice.Check(
            "FINDING: -inf fixes that case by breaking it differently",
            result["partial_inf"][0] == 1.0 and (result["all_masked_inf"] is None
                                                 or None in result["all_masked_inf"]),
            f"with -inf, a partially masked row is exact -- {result['partial_inf']} for one surviving "
            f"position -- and a fully masked row computes -inf minus -inf: "
            f"{result['all_masked_inf']}, where None marks a nan. The two obvious sentinels fail on "
            f"the same input in opposite directions, and only one of them says so"),
        practice.Check(
            "FINDING: zeroing after the softmax scales the context by the mass it dropped",
            result["post_hoc_total"] < 1.0 and result["one_ratio"],
            f"zeroing the weights after the softmax leaves them summing to "
            f"{result['post_hoc_total']} rather than 1, so the context is the right direction times "
            f"the retained mass: the ratio to the correct context is {ratio} -- identical in every "
            f"dimension. It is a pure scaling, which is why it survives a spot check"),
        practice.Check(
            "CONTROL: the padded states here are all-zero, so masking is not cosmetic",
            result["empty_padding"] and result["padding_mass"] > 0.25,
            f"the two padded positions are zero vectors, so their raw score is 0.0 and softmax gives "
            f"them {result['padding_mass']:.1%} of the mass between them -- more than any single "
            f"real position after masking. A zero vector is not a neutral one; it is the score "
            f"a maximally uninformative query would earn"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
