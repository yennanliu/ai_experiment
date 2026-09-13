"""Exercise 1 — the first `window` rows of an SWA mask are causal already.

    **Easy.** Run `code/main.py`. Verify SWA at `window=4` zeroes everything
    outside the last 4 tokens per row. Verify `window=n` reproduces full causal
    attention bit-identically.

Reading of the exercise: both claims are checked cell by cell rather than by
counting, because "bit-identical" is a claim about `0.0` and `-inf` and not about
a tolerance.

**ANSWER: both hold exactly.** At `window=4` on 8 tokens, row `i` attends
`[max(0, i-3) .. i]` and nothing else -- 26 of the 36 causal cells. And
`swa_mask(n, n)` is `causal_mask(n)` cell for cell, because `max(0, i - n + 1)`
is 0 for every `i < n`.

**FINDING: rows 0 to `window-1` are causal already.** The window is not full
until row `window - 1`, so at `window=4` the first four rows are identical to
full causal attention and only rows 4-7 differ. A sliding window does nothing at
all until the sequence is longer than the window -- which is the whole content of
Exercise 2.

**FINDING: the demo's own draw has the *least* weight on position 0.** `main()`
prints a single attention row and comments "notice the weight bleeding to
position 0 -- the attention sink". The row is
`[0.018, 0.068, 0.335, 0.031, 0.054, 0.112, 0.156, 0.227]`: position 0 is
**0.018**, the smallest of the eight and a seventh of the uniform 0.125. Over 200
fresh draws the mean weight on position 0 stays below uniform. The sink is a
property of *trained* models, where position 0 becomes somewhere to put unused
probability mass; a softmax over random scores has no reason to produce one.

**CONTROL: the counts follow from the shape.** Causal attends `n(n+1)/2 = 36`
cells; `window=4` attends `36 - 10 = 26`, the ten being what rows 4-7 give up.
`window=n` attends 36. Nothing here depends on the scores.

Structure: `differs` finds the rows where two masks disagree; `sink` measures the
weight on position 0 against uniform over many draws.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "15-attention-variants"
TOKENS, WINDOW, WIDTH, DRAWS = 8, 4, 8, 200


def differs(left, right):
    """Row indices where two masks disagree anywhere."""
    return [i for i, (a, b) in enumerate(zip(left, right)) if a != b]


def attended(mask, row):
    """The positions row `row` is allowed to attend."""
    return [j for j, v in enumerate(mask[row]) if v == 0.0]


def sink(ref, draws=DRAWS, width=WIDTH, tokens=TOKENS, seed=1):
    """Mean attention weight on position 0 over fresh random keys and queries."""
    rng, weights = random.Random(seed), []
    for _ in range(draws):
        keys = [[rng.gauss(0, 1) for _ in range(width)] for _ in range(tokens)]
        query = [rng.gauss(0, 1) for _ in range(width)]
        weights.append(ref.attention_row(query, keys, keys, ref.causal_mask(tokens)[-1])[1][0])
    return statistics.fmean(weights)


def demo_row(ref, tokens=TOKENS, width=WIDTH):
    """The exact row main() prints, from the same seed and the same draws."""
    rng = random.Random(0)
    keys = [[rng.gauss(0, 1) for _ in range(width)] for _ in range(tokens)]
    values = [[rng.gauss(0, 1) for _ in range(width)] for _ in range(tokens)]
    query = [rng.gauss(0, 1) for _ in range(width)]
    return ref.attention_row(query, keys, values, ref.causal_mask(tokens)[-1])[1]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    causal, narrow, wide = (ref.causal_mask(TOKENS), ref.swa_mask(TOKENS, WINDOW),
                            ref.swa_mask(TOKENS, TOKENS))
    row = demo_row(ref)
    return {
        "identical": wide == causal, "moved": differs(causal, narrow),
        "spans": {i: (attended(narrow, i), [max(0, i - WINDOW + 1), i])
                  for i in (WINDOW - 1, 5, TOKENS - 1)},
        "counts": (ref.count_nonmasked(causal), ref.count_nonmasked(narrow),
                   ref.count_nonmasked(wide)),
        "row": [round(v, 3) for v in row], "first": row[0],
        "rank": sorted(row, reverse=True).index(row[0]) + 1,
        "mean_sink": sink(ref), "uniform": 1 / TOKENS,
    }


def verify(result):
    causal, narrow, wide = result["counts"]
    spans = result["spans"]
    return [
        practice.Check(
            "ANSWER: window=n is causal cell for cell, and window=4 is the last four",
            result["identical"] and all(got == list(range(lo, hi + 1))
                                        for got, (lo, hi) in spans.values()),
            f"swa_mask({TOKENS}, {TOKENS}) equals causal_mask({TOKENS}) exactly, because "
            f"max(0, i - n + 1) is 0 for every row. At window={WINDOW} the rows attend "
            + ", ".join(f"{i}: {got}" for i, (got, _) in spans.items())
            + f" -- [max(0, i-{WINDOW - 1}) .. i] and nothing else",
        ),
        practice.Check(
            "FINDING: rows 0 to window-1 are causal already",
            result["moved"] == list(range(WINDOW, TOKENS)),
            f"only rows {result['moved']} differ from full causal: the window is not full until "
            f"row {WINDOW - 1}, so the first {WINDOW} rows are unchanged. A sliding window does "
            "nothing until the sequence is longer than the window, which is Exercise 2's whole "
            "content",
        ),
        practice.Check(
            "FINDING: the demo's own draw has the least weight on position 0",
            result["rank"] == TOKENS and result["first"] < result["uniform"],
            f"main() prints {result['row']} and comments 'notice the weight bleeding to position "
            f"0 -- the attention sink'. Position 0 is {result['first']:.3f}, the smallest of the "
            f"{TOKENS} and a seventh of the uniform {result['uniform']:.3f}",
        ),
        practice.Check(
            "FINDING: it is not a draw -- 200 more stay below uniform too",
            result["mean_sink"] < result["uniform"],
            f"over {DRAWS} fresh key/query draws the mean weight on position 0 is "
            f"{result['mean_sink']:.4f} against a uniform {result['uniform']:.4f}. The sink is a "
            "property of trained models, where position 0 becomes somewhere to put unused "
            "probability mass; a softmax over random scores has no reason to produce one",
        ),
        practice.Check(
            "CONTROL: the counts follow from the shape and not from the scores",
            (causal, narrow, wide) == (36, 26, 36),
            f"causal attends n(n+1)/2 = {causal} cells, window={WINDOW} attends {narrow} -- the "
            f"{causal - narrow} missing being what rows {result['moved']} give up -- and "
            f"window=n attends {wide}. No query, key or value appears in any of it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
