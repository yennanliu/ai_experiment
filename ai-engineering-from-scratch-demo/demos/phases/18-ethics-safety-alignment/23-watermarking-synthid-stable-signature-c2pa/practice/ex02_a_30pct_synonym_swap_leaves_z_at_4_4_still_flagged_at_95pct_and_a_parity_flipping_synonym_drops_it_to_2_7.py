"""Exercise 2 — a 30% synonym swap leaves z at 4.4, still flagged at 95%; a parity-flipping synonym drops it to 2.7.

    Implement a paraphrase attack that replaces 30% of tokens with synonyms.
    Re-measure the z-score.

Reading of the exercise: the toy vocabulary is integers, so "synonym" needs a
definition. The reference's green set is `{t : (t + h) % 2 == 0}`: all the even
tokens or all the odd ones, so whether a swap keeps a token green depends only
on whether its synonym has the same parity. Three attacks at 30% are measured
on the same 40 seeded 1000-token watermarked texts: the reference's
`paraphrase` (a random replacement, which is what a synonym looks like under a
truly pseudorandom partition), a parity-keeping synonym `t -> t + 100`, and a
parity-flipping synonym `t -> t xor 1`. Each is compared with the closed form
`z = 0.8 sqrt(n) x (share of positions whose K+1 window survives)`.

**ANSWER: z falls from 24.98 (the first text, unattacked) to a mean of 4.38
under a 30% swap, and every one of the 40 texts is still flagged at the 95%
threshold.** The prediction is 4.25:
a position keeps its 0.9 green rate only if it and its K = 4 context tokens are
all untouched, 0.7^5 = 16.8% of positions. 30% sits right on the reference's
own z >= 4 line: 27 of 40 texts (68%) stay above it, and the shipped run's
3.86 is one draw that fell below.

**FINDING: the synonym's parity decides the outcome, because the "pseudorandom
partition" uses one bit of the hash.** A same-parity synonym keeps the
swapped token's own colour, so only the context damage counts: mean z 5.95
(predicted 6.07), 36 of 40 above 4. A parity-flipping synonym turns a green
token red: mean 2.66 (predicted 2.43), 34 of 40 still over 1.645 and only 5
over 4.

**FINDING: the swap rate that defeats the 95% detector is about 43%, not 30%.**
Mean z under the reference's attack is 4.38 / 1.90 / 0.65 at 30 / 40 / 50%;
the closed form crosses 1.645 at 43%.

Structure: `texts()` makes the seeded watermarked texts; `attacks()` applies
each swap with a shared per-text mask; `predicted()` is the closed form.
"""

from __future__ import annotations

import contextlib
import math
import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "23-watermarking-synthid-stable-signature-c2pa"
N, TEXTS, RATIO, Z95, ZREF = 1000, 40, 0.3, 1.645, 4.0
SWEEP = (0.3, 0.4, 0.5)


@contextlib.contextmanager
def seeded(ref, seed):
    saved, ref.random = ref.random, random.Random(seed)
    try:
        yield
    finally:
        ref.random = saved


def texts(ref):
    out = []
    for s in range(TEXTS):
        with seeded(ref, s):
            out.append(ref.watermarked_sample(N, [ref.random.randrange(ref.VOCAB) for _ in range(ref.K)]))
    return out


def swap(tokens, ratio, synonym, seed):
    rng = random.Random(seed)
    return [synonym(t) if rng.random() < ratio else t for t in tokens]


def attacks(ref, text, seed):
    """z after each 30% attack, plus the reference's attack at every sweep ratio."""
    with seeded(ref, 1000 + seed):
        ref_z = {r: ref.detect(ref.paraphrase(text, r)) for r in SWEEP}
    return {
        "random (reference)": ref_z[RATIO],
        "same-parity synonym": ref.detect(swap(text, RATIO, lambda t: (t + 100) % ref.VOCAB, seed)),
        "parity-flip synonym": ref.detect(swap(text, RATIO, lambda t: t ^ 1, seed)),
        "sweep": ref_z,
    }


def predicted(ref, ratio, own_colour):
    """Closed form: green excess survives on positions whose whole window is intact.

    own_colour: +1 the swapped token keeps its colour, -1 it flips, 0 it is random.
    """
    intact_ctx = (1 - ratio) ** ref.K
    excess = intact_ctx * ((1 - ratio) + own_colour * ratio)
    return round(2 * 0.4 * excess * math.sqrt(N), 2)


def crossing(ref):
    """Smallest swap rate (1% steps) at which the predicted mean z drops below 1.645."""
    return next(r / 100 for r in range(101) if predicted(ref, r / 100, 0) < Z95)


def summarize(runs, names):
    zs = {k: [r[k] for r in runs] for k in names}
    return {
        "mean": {k: round(sum(v) / TEXTS, 2) for k, v in zs.items()},
        "over95": {k: sum(z >= Z95 for z in v) for k, v in zs.items()},
        "over4": {k: sum(z >= ZREF for z in v) for k, v in zs.items()},
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    marked = texts(ref)
    runs = [attacks(ref, t, s) for s, t in enumerate(marked)]
    names = [k for k in runs[0] if k != "sweep"]
    return {
        "clean": round(ref.detect(marked[0]), 2), **summarize(runs, names),
        "predicted": {k: predicted(ref, RATIO, c) for k, c in zip(names, (0, 1, -1))},
        "sweep": {r: round(sum(x["sweep"][r] for x in runs) / TEXTS, 2) for r in SWEEP},
        "crossing": crossing(ref), "window": round(0.7 ** (ref.K + 1), 3),
    }


def verify(result):
    mean, pred, o95, o4 = (result[k] for k in ("mean", "predicted", "over95", "over4"))
    rnd, keep, flip = "random (reference)", "same-parity synonym", "parity-flip synonym"
    return [
        practice.Check(
            "ANSWER: 30% swap -> mean z 4.38, all 40 texts still over the 95% threshold",
            (mean[rnd], pred[rnd], o95[rnd], o4[rnd], result["window"], result["clean"])
            == (4.38, 4.25, TEXTS, 27, 0.168, 24.98),
            f"clean z {result['clean']}; after 30%: mean {mean[rnd]} (closed form {pred[rnd]}), "
            f"{o95[rnd]}/{TEXTS} over 1.645, {o4[rnd]}/{TEXTS} over 4",
        ),
        practice.Check(
            "FINDING: the synonym's parity decides the outcome",
            (mean[keep], pred[keep], o4[keep]) == (5.95, 6.07, 36)
            and (mean[flip], pred[flip], o95[flip], o4[flip]) == (2.66, 2.43, 34, 5),
            f"means {mean}; closed form {pred}; over 1.645 {o95}; over 4 {o4}",
        ),
        practice.Check(
            "FINDING: the swap rate that defeats the 95% detector is about 43%",
            result["sweep"] == {0.3: 4.38, 0.4: 1.9, 0.5: 0.65} and result["crossing"] == 0.43,
            f"mean z by swap rate {result['sweep']}; closed form crosses 1.645 at "
            f"{result['crossing']:.0%}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
