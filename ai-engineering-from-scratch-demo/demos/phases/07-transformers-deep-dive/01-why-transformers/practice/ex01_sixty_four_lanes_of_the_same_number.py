"""Exercise 1 — sixty-four lanes holding one number.

    **Easy.** Take `rnn_style` from `code/main.py` and replace the scalar
    hidden state with a length-64 vector of hidden states. Re-measure. How
    much does the serial overhead grow with hidden-state dimension?

Reading of the exercise: "replace the scalar hidden state with a length-64
vector" is done literally first -- `h = [decay*hi + x for hi in h]`, which is
what the sentence says -- and then again as a real recurrence, `h = W h + x`,
because the literal reading turns out not to be a 64-dimensional RNN at all.
Both are timed against the lesson's own `rnn_style` on the same input, with the
same best-of-three protocol `benchmark()` uses.

**ANSWER: roughly linearly, about 1.3x per lane.** 64 lanes cost ~85x the
scalar at N=20000, not 64x: the list comprehension adds a per-step allocation
the scalar loop does not have. Per lane that overhead falls from 3.9x at d=1 to
1.3x at d=64, so the ratio-to-scalar grows slightly faster than d while the cost
*per lane* improves -- the interpreter overhead amortises.

**FINDING: the serial overhead grows by exactly zero.** The lesson's own
`depth()` takes `n` and no `d`; `rnn_depth = n` whatever the hidden size. Step 2
says in as many words that "depth, not op count, decides GPU time", so the
quantity the exercise names is invariant under the change the exercise asks for.
What grows is work per step, which is the part a GPU parallelises for free.

**FINDING: the 64 lanes hold one distinct value.** Every lane starts at 0.0 and
gets the same `decay*hi + x`, so they stay equal forever: `len(set(h)) == 1`,
bit-identical to `rnn_style`'s scalar. The literal reading buys 64x the work and
zero extra information, because nothing in it mixes the lanes.

**FINDING: a real d-dimensional recurrence is quadratic.** `h_t = W h_{t-1} +
x_t` costs `d^2` multiply-adds per step, and at d=64 it measures ~10000x the
scalar -- about 125x the elementwise version. That is the growth the exercise
was reaching for, and the wording it used cannot reach it.

Structure: `lanes` is the literal reading; `mixed` is the real one; `best` is
`benchmark()`'s best-of-three, reused rather than re-derived.
"""

from __future__ import annotations

import math
import time

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "01-why-transformers"
N, DIMS, DECAY = 20_000, (1, 2, 4, 8, 16, 32, 64), 0.9
MIXED_N = 2_000


def lanes(xs, d, decay=DECAY):
    """The exercise read literally: a length-d hidden state, updated elementwise."""
    h = [0.0] * d
    for x in xs:
        h = [decay * hi + x for hi in h]
    return h


def mixed(xs, d, decay=DECAY):
    """The recurrence an RNN actually runs: h = W h + x, so d^2 work per step."""
    rows = [[decay / d] * d for _ in range(d)]
    h = [0.0] * d
    for x in xs:
        h = [sum(w * hj for w, hj in zip(row, h)) + x for row in rows]
    return h


def best(call, reps=3):
    """`benchmark()`'s own protocol: best of three, so a stray GC pause cannot win."""
    fastest = math.inf
    for _ in range(reps):
        start = time.perf_counter()
        call()
        fastest = min(fastest, time.perf_counter() - start)
    return fastest


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    xs = [0.001 * (i % 17) for i in range(N)]
    small = xs[:MIXED_N]
    scalar = best(lambda: ref.rnn_style(xs))
    vector = {d: best(lambda d=d: lanes(xs, d)) for d in DIMS}
    scalar_small = best(lambda: ref.rnn_style(small))
    return {
        "scalar": scalar, "vector": vector,
        "lane_ratio": {d: vector[d] / scalar / d for d in DIMS},
        "state": lanes(xs, 64), "reference_scalar": ref.rnn_style(xs),
        "mixed": best(lambda: mixed(small, 64)) / scalar_small,
        "elementwise_small": best(lambda: lanes(small, 64)) / scalar_small,
        "depths": {d: ref.depth(N)[0] for d in DIMS},
        "signature": ref.depth.__code__.co_varnames[:ref.depth.__code__.co_argcount],
    }


def verify(result):
    ratio = result["vector"][64] / result["scalar"]
    state, lane = result["state"], result["lane_ratio"]
    return [
        practice.Check(
            "ANSWER: 64 lanes cost ~85x the scalar, about 1.3x per lane",
            40 < ratio < 200 and lane[1] > 2 * lane[64],
            f"at N={N}: scalar {result['scalar'] * 1e3:.2f} ms -> 64 lanes "
            f"{result['vector'][64] * 1e3:.2f} ms, {ratio:.0f}x. Per lane the overhead falls "
            f"from {lane[1]:.2f}x at d=1 to {lane[64]:.2f}x at d=64, so the total grows a little "
            "faster than d while the per-lane cost improves -- interpreter overhead amortising",
        ),
        practice.Check(
            "FINDING: the serial overhead itself grows by exactly zero",
            set(result["depths"].values()) == {N} and "d" not in result["signature"],
            f"the lesson's own depth{result['signature']} takes no hidden size at all: rnn_depth "
            f"= {N} at every d in {list(DIMS)}. Step 2 says 'depth, not op count, decides GPU "
            "time', so the quantity this exercise names is invariant under the change it asks for",
        ),
        practice.Check(
            "FINDING: the 64 lanes hold one distinct value, equal to the scalar",
            len(set(state)) == 1 and state[0] == result["reference_scalar"],
            f"every lane starts at 0.0 and gets the same decay*h + x, so len(set(h)) = "
            f"{len(set(state))} and the value is bit-identical to rnn_style's "
            f"{result['reference_scalar']!r}. 64x the work, zero extra information: nothing in "
            "the elementwise update mixes the lanes, so it is 64 copies of one scalar RNN",
        ),
        practice.Check(
            "FINDING: a real d-dimensional recurrence is quadratic, not linear",
            result["mixed"] > 30 * result["elementwise_small"],
            f"h = W h + x is 64^2 = 4096 multiply-adds per step: {result['mixed']:.0f}x the "
            f"scalar against the elementwise reading's {result['elementwise_small']:.0f}x, a "
            f"further {result['mixed'] / result['elementwise_small']:.0f}x. That is the growth "
            "the question is after, and 'a length-64 vector of hidden states' cannot reach it",
        ),
        practice.Check(
            "CONTROL: the wall-clock growth really is in d, not in N",
            all(result["vector"][b] > 1.4 * result["vector"][b // 2] for b in (16, 32, 64)),
            "N is held at {} across the whole sweep and each doubling of d still costs {}. "
            "So the measured curve is the hidden size, not a longer sequence".format(
                N, ", ".join(f"{result['vector'][b] / result['vector'][b // 2]:.1f}x"
                             for b in (16, 32, 64))),
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
