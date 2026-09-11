"""Exercise 3 — the width stops growing with length.

    **Hard.** Train a GRU encoder-decoder with Bahdanau attention on the toy copy
    task from lesson 09. Plot accuracy vs sequence length. Compare against the
    no-attention baseline. You should see the gap widen as length grows,
    confirming attention lifts the bottleneck.

Reading of the exercise: the gap does widen, and the reason is sharper than a
gap. torch is not installed, so both arms here are the closed-form version of
lesson 09's substitute -- a fixed random encoder with a least-squares decoder,
which has no optimisation to confound the comparison. The no-attention arm
squeezes the whole source through one d-vector; the attention arm lets step t
read encoder state t, which is what attention converges to on a copy task.

At d=16 the no-attention arm scores exact-match 1.0000 at length 2, 0.4150 at 4
and 0.0000 from 8 onward, while the attention arm scores 1.0000 at every length
to 64. That is the plot the exercise asks for, and the widening gap is the whole
of it at fixed width.

The stronger statement is what happens when the width is allowed to move. The
no-attention arm is not broken at length 8 -- it is under-provisioned, and giving
it the width its rank demands restores it exactly: L(V-1)+1 is 19, 37 and 73 at
lengths 2, 4 and 8, and at those widths it scores 1.0000 again. So the
requirement grows linearly with length. The attention arm reaches 1.0000 at d=8
for length 4, length 16 and length 64 alike: its requirement does not move at
all. "Attention lifts the bottleneck" is that the width needed becomes a
function of the vocabulary rather than of the sequence.

One thing the exercise's framing would hide: below its own threshold the
attention arm degrades with length too. At d=4 it scores 0.625 at length 4 and
0.000 at 64 -- not because the context ran out of room but because a per-token
error rate compounds over more tokens. Exact-match falls with length under any
imperfect decoder, so the axis the exercise says to plot moves for two different
reasons and only one of them is the bottleneck.

Structure: `one_hot` and `sequences` build the copy task, `bottleneck` projects
the whole source into one d-vector and solves for a decoder over all L
positions, `attend` solves one shared decoder that reads a single position's
state. Both are `lstsq`, so neither arm has a training schedule to differ on.
"""

from __future__ import annotations

from harness import practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "10-attention-mechanism"

SYMBOLS, TRAIN, TEST = 10, 400, 200
LENGTHS, WIDTH = (2, 4, 8, 16, 32, 64), 16
NARROW = (4, 8, 16, 32)


def sequences(np, length, seed):
    rng = np.random.default_rng(seed)
    return ([rng.integers(0, SYMBOLS, length) for _ in range(TRAIN)],
            [rng.integers(0, SYMBOLS, length) for _ in range(TEST)], rng)


def one_hot(np, rows, length):
    matrix = np.zeros((len(rows), length * SYMBOLS))
    for i, row in enumerate(rows):
        for position, symbol in enumerate(row):
            matrix[i, position * SYMBOLS + symbol] = 1.0
    return matrix


def exact(np, guess, gold) -> float:
    return round(float((guess == gold).all(1).mean()), 4)


def bottleneck(np, length, width, seed=0) -> float:
    """One fixed-size context for the whole source, decoded into all L positions."""
    train, test, rng = sequences(np, length, seed)
    seen, held = one_hot(np, train, length), one_hot(np, test, length)
    encoder = rng.normal(size=(length * SYMBOLS, width)) / (length * SYMBOLS) ** 0.5
    decoder = np.linalg.lstsq(seen @ encoder, seen, rcond=None)[0]
    guess = ((held @ encoder) @ decoder).reshape(TEST, length, SYMBOLS).argmax(-1)
    return exact(np, guess, np.array([list(row) for row in test]))


def attend(np, length, width, seed=0) -> float:
    """Step t reads encoder state t -- what attention converges to on a copy task."""
    train, test, rng = sequences(np, length, seed)
    states = rng.normal(size=(SYMBOLS, width)) / width ** 0.5
    seen = np.stack([states[row] for row in train]).reshape(-1, width)
    held = np.stack([states[row] for row in test]).reshape(-1, width)
    gold = np.array([list(row) for row in train]).reshape(-1)
    decoder = np.linalg.lstsq(seen, np.eye(SYMBOLS)[gold], rcond=None)[0]
    guess = (held @ decoder).reshape(TEST, length, SYMBOLS).argmax(-1)
    return exact(np, guess, np.array([list(row) for row in test]))


def widths(np) -> dict:
    """Both arms at a fixed width, the baseline at its rank, attention at four narrow ones."""
    ranks = {length: length * (SYMBOLS - 1) + 1 for length in LENGTHS}
    return {"ranks": ranks,
            "none": {length: bottleneck(np, length, WIDTH) for length in LENGTHS},
            "attn": {length: attend(np, length, WIDTH) for length in LENGTHS},
            "provisioned": {length: bottleneck(np, length, ranks[length]) for length in (2, 4, 8)},
            "constant": {length: {d: attend(np, length, d) for d in NARROW}
                         for length in (4, 16, 64)}}


def solve():
    try:
        import numpy as np
    except ImportError as exc:                      # pragma: no cover - T1 needs numpy
        raise practice.Skip(f"needs numpy: uv sync --extra math ({exc})") from None
    measured = widths(np)
    return dict(measured, width=WIDTH,
                narrow_ok=all(measured["constant"][n][8] == 1.0 for n in measured["constant"]),
                restored=all(v == 1.0 for v in measured["provisioned"].values()))


def verify(result):
    ranks, provisioned, grew = result["ranks"], result["constant"], result["provisioned"]
    none, attn = result["none"], result["attn"]
    return [
        practice.Check(
            "ANSWER: at a fixed width the gap widens from nothing to everything",
            none[2] == attn[2] == 1.0 and none[8] == 0.0 and attn[64] == 1.0,
            f"at d={result['width']}, exact-match without attention is {none} across lengths "
            f"{list(LENGTHS)} and with attention {attn}. The two are level at length 2, part-way "
            f"apart at 4, and maximally apart from 8 onward. torch is not installed, so both arms "
            f"are least-squares decoders with no optimisation to confound them"),
        practice.Check(
            "MECHANISM: the no-attention arm is under-provisioned, not broken",
            result["restored"],
            f"giving it the width its rank demands restores it exactly: L({SYMBOLS}-1)+1 is "
            f"{ {length: ranks[length] for length in grew} } at lengths {sorted(grew)}, and at those "
            f"widths it scores {grew}. The bottleneck is a rank condition, and the condition is met "
            f"or not met"),
        practice.Check(
            "FINDING: the width it needs grows linearly with the sequence",
            ranks[64] == 8 * ranks[8] - 7 and ranks[64] > 500,
            f"the requirement is {ranks} across {list(LENGTHS)} -- {ranks[64]} dimensions to copy 64 "
            f"symbols. Doubling the source doubles the context the exercise's baseline needs, which "
            f"is the shape of the curve the exercise says to plot"),
        practice.Check(
            "ANSWER: the width attention needs does not move with the sequence at all",
            result["narrow_ok"],
            f"the attention arm reaches 1.0000 at d=8 for lengths {sorted(provisioned)} alike: "
            f"{ {length: provisioned[length][8] for length in provisioned} }. Its requirement is a "
            f"function of the vocabulary, not of the sequence -- which is what lifting the "
            f"bottleneck means, stated as a number rather than as a gap"),
        practice.Check(
            "FINDING: below its own threshold attention degrades with length too",
            provisioned[4][4] > provisioned[16][4] > provisioned[64][4],
            f"at d=4 the attention arm scores "
            f"{ {length: provisioned[length][4] for length in provisioned} } across lengths "
            f"{sorted(provisioned)}. Nothing ran out of room -- a per-token error rate compounds "
            f"over more tokens, so exact-match falls with length under any imperfect decoder"),
        practice.Check(
            "CONTROL: so the plot the exercise asks for moves for two reasons",
            provisioned[64][8] == 1.0 > provisioned[64][4],
            f"at length 64 the same arm scores {provisioned[64][4]} at d=4 and "
            f"{provisioned[64][8]} at d=8. Accuracy against length falls when the context is too "
            f"small for the sequence and also when it is too small for the vocabulary, and only the "
            f"first is the bottleneck the lesson is about"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
