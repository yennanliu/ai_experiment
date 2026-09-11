"""Exercise 1 — the simulation never reads epochs.

    **Easy.** Implement the toy copy task. Train a GRU seq2seq on input-output
    pairs where the target equals the source. Measure accuracy at lengths 5, 10,
    20. Reproduce the bottleneck.

Reading of the exercise: what the lesson ships as the bottleneck demonstration
is not a trained model and its "accuracy" is not copy accuracy. Two of
`simulate_copy_accuracy`'s five parameters, `epochs` and `n_train`, appear zero
times in its body -- setting them to 1 and 1, or to 100000 and 1000000, returns
the same 0.89, 0.83, 0.69 at lengths 5, 10 and 20. Nothing is fitted; the
embeddings are random and stay random.

The metric is a two-alternative comparison: one true sequence scored against one
random sequence, so its floor is 0.50 and the lesson's own table bottoms out at
0.51 by length 80, which is chance rather than collapse. Exact-match copy
accuracy on a length-20 sequence over ten symbols has a chance level of 1e-20,
so the two numbers are not the same quantity and the lower one is not a worse
version of the higher one.

Reproducing the bottleneck properly needs a model that can actually be fitted,
and one is available in closed form: a fixed random projection into d dimensions
with a least-squares decoder trained on top. There the bottleneck is exact
rather than gradual. The one-hot encoding of a length-L sequence over a
V-symbol vocabulary has rank L(V-1)+1 -- 46, 91 and 181 at L = 5, 10, 20,
measured, not assumed -- and exact-match accuracy is 1.000 once d reaches that
and falls off below it: 0.920 at d=32 for L=5, 0.000 at d=32 for L=10. The
lesson's claim that a fixed-size context is the constraint is right; its
demonstration is a coin flip that never trains.

Structure: `one_hot` lays each position's symbol into its own V-wide block, so
the rank of the resulting matrix is the quantity the bottleneck is about.
`autoencode` fixes a random encoder, solves for the decoder with `lstsq`, and
returns exact-match and per-token accuracy on held-out sequences.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "09-sequence-to-sequence"

SYMBOLS, LENGTHS, WIDTHS = 10, (5, 10, 20), (8, 32, 64, 128)
TRAIN, TEST = 400, 200
INERT = ("epochs", "n_train")


def one_hot(np, rows, length):
    matrix = np.zeros((len(rows), length * SYMBOLS))
    for i, row in enumerate(rows):
        for position, symbol in enumerate(row):
            matrix[i, position * SYMBOLS + symbol] = 1.0
    return matrix


def autoencode(np, length, width, seed=0) -> dict:
    """Fixed random projection into `width` dimensions, least-squares decoder on top."""
    rng = np.random.default_rng(seed)
    train = [rng.integers(0, SYMBOLS, length) for _ in range(TRAIN)]
    test = [rng.integers(0, SYMBOLS, length) for _ in range(TEST)]
    seen, held = one_hot(np, train, length), one_hot(np, test, length)
    encoder = rng.normal(size=(length * SYMBOLS, width)) / (length * SYMBOLS) ** 0.5
    decoder = np.linalg.lstsq(seen @ encoder, seen, rcond=None)[0]
    guess = (held @ encoder @ decoder).reshape(TEST, length, SYMBOLS).argmax(-1)
    gold = np.array([list(row) for row in test])
    return {"exact": round(float((guess == gold).all(1).mean()), 4),
            "token": round(float((guess == gold).mean()), 4),
            "rank": int(np.linalg.matrix_rank(seen))}


def inert(ref) -> list:
    """The same three lengths under wildly different epochs and n_train."""
    return [[ref.simulate_copy_accuracy(length, epochs=e, n_train=n) for length in LENGTHS]
            for e, n in ((200, 300), (1, 1), (100000, 1000000))]


def sweep(np) -> dict:
    """Exact-match by context width, the measured rank, and the rank the algebra predicts."""
    grid = {length: {width: autoencode(np, length, width) for width in WIDTHS}
            for length in LENGTHS}
    return {"exact": {n: {w: grid[n][w]["exact"] for w in WIDTHS} for n in LENGTHS},
            "ranks": {n: grid[n][WIDTHS[0]]["rank"] for n in LENGTHS},
            "predicted": {n: n * (SYMBOLS - 1) + 1 for n in LENGTHS}}


def solve():
    try:
        import numpy as np
    except ImportError as exc:                      # pragma: no cover - T1 needs numpy
        raise practice.Skip(f"needs numpy: uv sync --extra math ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    body = inspect.getsource(ref.simulate_copy_accuracy).split("\n", 1)[1]
    return dict(
        sweep(np), mentions={name: body.count(name) for name in INERT}, inert=inert(ref),
        lesson={length: ref.simulate_copy_accuracy(length) for length in LENGTHS},
        floor=ref.simulate_copy_accuracy(80),
        same=len({tuple(row) for row in inert(ref)}) == 1,
        chance=SYMBOLS ** -LENGTHS[-1],
        context={d: ref.simulate_copy_accuracy(20, context_dim=d) for d in (4, 8, 16)})


def verify(result):
    lesson, mentions = result["lesson"], result["mentions"]
    exact, ranks, predicted = result["exact"], result["ranks"], result["predicted"]
    return [
        practice.Check(
            "ANSWER: epochs and n_train appear zero times in the body, and changing them changes nothing",
            set(mentions.values()) == {0} and result["same"],
            f"`simulate_copy_accuracy` takes {list(INERT)} and mentions them {mentions} times below "
            f"its signature. At (200, 300), (1, 1) and (100000, 1000000) it returns "
            f"{result['inert'][0]} every time at lengths {list(LENGTHS)}. Nothing is fitted -- the "
            f"embeddings are random and stay random"),
        practice.Check(
            "MECHANISM: the metric is a two-way comparison, so its floor is 0.50 and not zero",
            0.45 < result["floor"] < 0.6 < min(lesson.values()),
            f"the function scores the true sequence against one random sequence and counts the "
            f"wins, so chance is 0.50. Its own table reads {lesson} at {list(LENGTHS)} and "
            f"{result['floor']} at length 80 -- the bottom of the curve is the coin, not the model. "
            f"Exact-match copy accuracy at length {LENGTHS[-1]} has a chance level of "
            f"{result['chance']:.0e}"),
        practice.Check(
            "MECHANISM: a fittable model puts the bottleneck at an exact rank, measured not assumed",
            ranks == predicted,
            f"the one-hot encoding of a length-L sequence over {SYMBOLS} symbols has rank "
            f"L({SYMBOLS}-1)+1, because each position's block sums to one: predicted "
            f"{predicted}, measured {ranks}. That is the number a "
            f"fixed-size context has to reach"),
        practice.Check(
            "ANSWER: exact-match reaches 1.000 once the context clears the rank, and not before",
            exact[5][64] == 1.0 and exact[10][128] == 1.0 and exact[20][128] < 0.7,
            f"exact-match accuracy by context width: {exact}. At L=5 the rank is "
            f"{ranks[5]} and d=64 clears it for {exact[5][64]}; at L=10 the rank is "
            f"{ranks[10]} and d=128 clears it for {exact[10][128]}; at L=20 the rank is "
            f"{ranks[20]} and d=128 does not, for {exact[20][128]}"),
        practice.Check(
            "FINDING: below the rank the failure is sharp, not gradual",
            exact[10][64] > exact[10][32] and exact[10][32] < 0.1,
            f"at L=10, exact-match goes {[exact[10][w] for w in WIDTHS]} across widths "
            f"{list(WIDTHS)} -- {exact[10][32]} at d=32 and {exact[10][64]} at d=64, on either side "
            f"of the rank {ranks[10]}. The lesson's simulation shows a smooth slide from "
            f"0.89 to 0.51 because it is measuring a coin flip getting harder, not a code running "
            f"out of room"),
        practice.Check(
            "CONTROL: in the lesson's own simulation more context is not reliably better",
            result["context"][16] < result["context"][8],
            f"sweeping `context_dim` at length 20 gives {result['context']} -- 16 dimensions score "
            f"below 8. With nothing trained, extra dimensions add noise to a dot product rather "
            f"than capacity to a code, so the demonstration does not show the property it names"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
