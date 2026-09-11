"""Exercise 2 — one head matches exactly, two do not.

    **Medium.** Add multi-head attention to the Luong `general` form. Split `d_h`
    into `n_heads` groups, run attention per head, concatenate. Verify the
    single-head case matches your earlier implementation.

Reading of the exercise: the verification it asks for passes bit-for-bit -- with
one head the context and the weights differ from the single-head implementation
by exactly 0.0, not by a tolerance -- because splitting a vector into one group
is the identity and every arithmetic operation is performed in the same order.
That is worth stating precisely, because it is the only part of this exercise
where an exact answer is available.

Everything above one head is a different function, and the exercise's wording
("split, run per head, concatenate") reads as though it were a refactor. It is
not. A single head takes one softmax over the full inner products; h heads take
h softmaxes over disjoint slices of them, and softmax does not distribute over a
sum. With 2 heads the context moves 0.6436 from the single-head answer and with
4 heads 2.1025, and at four heads they land on positions [0, 0, 0, 5] where the
single head chose 2 -- disagreeing with each other as well as with it. At two
heads they agree with the single head about where to look and still move the
context, so the divergence in weights is not guaranteed and the divergence in
output is.

The clearest form of that is the control. Set the projection to the identity, so
every head is scoring the same two vectors with no learned parameters to differ
on, and multi-head still disagrees -- 0.9323 at two heads, with the second head
choosing position 3 where the single head chose 1. Nothing here is a parameter
difference; it is the normalisation being applied to slices.

Also worth noting: the `general` form the exercise names is not in the lesson's
`code/main.py`, which ships `dot_attention` and `additive_attention` only. It is
`dot` with a learned matrix between the two vectors, written here on top of the
lesson's own `matvec`, `dot` and `softmax`.

Structure: `general` is the Luong general form built from the lesson's
primitives; `multihead` slices the decoder state, the encoder states and the
projection into `heads` groups and concatenates the per-head contexts. `spread`
reports how far a multi-head context sits from the single-head one.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "10-attention-mechanism"

WIDTH, POSITIONS, SEED = 8, 6, 0
HEAD_COUNTS = (1, 2, 4, 8)


def fixture() -> tuple:
    """A decoder state, six encoder states and a projection, from a fixed seed."""
    rng = random.Random(SEED)
    decoder = [rng.gauss(0, 1) for _ in range(WIDTH)]
    encoder = [[rng.gauss(0, 1) for _ in range(WIDTH)] for _ in range(POSITIONS)]
    projection = [[rng.gauss(0, 0.5) for _ in range(WIDTH)] for _ in range(WIDTH)]
    return decoder, encoder, projection


def general(ref, decoder, encoder, projection) -> tuple:
    """Luong's general form: score = (W s) . h, built from the lesson's own primitives."""
    query = ref.matvec(projection, decoder)
    weights = ref.softmax([ref.dot(query, h) for h in encoder])
    width = len(encoder[0])
    return [sum(w * h[d] for w, h in zip(weights, encoder)) for d in range(width)], weights


def multihead(ref, decoder, encoder, projection, heads) -> tuple:
    step = len(decoder) // heads
    joined, per_head = [], []
    for index in range(heads):
        cut = slice(index * step, (index + 1) * step)
        context, weights = general(ref, decoder[cut], [h[cut] for h in encoder],
                                   [row[cut] for row in projection[cut]])
        joined += context
        per_head.append(weights)
    return joined, per_head


def spread(single, joined) -> float:
    return round(max(abs(a - b) for a, b in zip(single, joined)), 4)


def peak(weights) -> int:
    return max(range(len(weights)), key=lambda i: weights[i])


def compare(ref, decoder, encoder, projection, counts) -> dict:
    single, weights = general(ref, decoder, encoder, projection)
    arms = {n: multihead(ref, decoder, encoder, projection, n) for n in counts}
    return {"drift": {n: spread(single, arms[n][0]) for n in counts},
            "peaks": {n: [peak(w) for w in arms[n][1]] for n in counts},
            "single_peak": peak(weights), "one": arms[counts[0]], "single": (single, weights)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    decoder, encoder, projection = fixture()
    identity = [[1.0 if i == j else 0.0 for j in range(WIDTH)] for i in range(WIDTH)]
    learned = compare(ref, decoder, encoder, projection, HEAD_COUNTS)
    plain = compare(ref, decoder, encoder, identity, (1, 2))
    single, single_weights = learned["single"]
    return {
        "exact": (spread(single, learned["one"][0]),
                  round(max(abs(a - b) for a, b in zip(single_weights, learned["one"][1][0])), 12)),
        "drift": learned["drift"], "peaks": learned["peaks"],
        "single_peak": learned["single_peak"], "identity": plain["drift"],
        "identity_peaks": plain["peaks"], "flat_peak": plain["single_peak"],
        "has_general": hasattr(ref, "general_attention"),
        "ships": sorted(n for n in dir(ref) if n.endswith("_attention")),
        "width": WIDTH, "positions": POSITIONS,
    }


def verify(result):
    drift, peaks, identity = result["drift"], result["peaks"], result["identity"]
    return [
        practice.Check(
            "ANSWER: with one head the match is exact, not close",
            result["exact"] == (0.0, 0.0),
            f"context and weights both differ from the single-head implementation by "
            f"{result['exact'][0]} and {result['exact'][1]}. Splitting a {result['width']}-vector "
            f"into one group is the identity and every operation runs in the same order, so the "
            f"verification the exercise asks for is available as an equality rather than a "
            f"tolerance"),
        practice.Check(
            "MECHANISM: above one head it is a different function, not a refactor",
            drift[1] == 0.0 < drift[2] < drift[4],
            f"the context moves {drift} from the single-head answer at {list(HEAD_COUNTS)} heads. "
            f"One head takes a single softmax over the full inner products; h heads take h softmaxes "
            f"over disjoint slices of them, and softmax does not distribute over a sum"),
        practice.Check(
            "FINDING: the heads disagree with each other and with the single head",
            len(set(peaks[4])) > 1 and result["single_peak"] not in peaks[4],
            f"at 4 heads the per-head argmaxes are {peaks[4]} -- two different positions, neither of "
            f"them the single head's {result['single_peak']} -- over the same "
            f"{result['positions']} encoder states. At 2 heads they happen to be {peaks[2]}, "
            f"agreeing with the single head on where to look while still moving the context by "
            f"{drift[2]}: the disagreement is not guaranteed, and the context change is"),
        practice.Check(
            "CONTROL: with the projection set to the identity, multi-head still disagrees",
            identity[1] == 0.0 < identity[2],
            f"replacing the learned matrix with the identity leaves the heads with no parameters to "
            f"differ on, and the context still moves {identity[2]} at two heads against "
            f"{identity[1]} at one. The second head peaks at "
            f"{result['identity_peaks'][2][1]} where the single head peaks at "
            f"{result['flat_peak']}. Nothing here is a parameter difference"),
        practice.Check(
            "FINDING: eight heads is not monotonically further away than four",
            drift[8] < drift[4],
            f"the drift goes {drift} across {list(HEAD_COUNTS)}. Slicing more finely does not move "
            f"the answer further -- with one dimension per head each softmax is over "
            f"{result['positions']} nearly-tied scores, so every head approaches a uniform average "
            f"and the concatenation approaches the mean of the encoder states"),
        practice.Check(
            "CONTROL: the form the exercise extends is not in the lesson's code",
            not result["has_general"],
            f"`code/main.py` ships {result['ships']} and no general form, so 'add multi-head "
            f"attention to the Luong general form' extends something that has to be written first. "
            f"It is `dot` with a learned matrix between the two vectors, built here on the lesson's "
            f"own matvec, dot and softmax so the single-head equality above means what it says"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
