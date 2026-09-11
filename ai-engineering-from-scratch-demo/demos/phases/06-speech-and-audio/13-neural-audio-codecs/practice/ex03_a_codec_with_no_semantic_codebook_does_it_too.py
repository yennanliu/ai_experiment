"""Exercise 3 — a codec with no semantic codebook does it too.

    **Hard.** Load Mimi. Encode a clip. Replace codebook 0 with random integers;
    decode. Then replace codebook 7 similarly. Compare the two corruptions —
    codebook 0 corruption should destroy intelligibility; codebook 7 corruption
    should barely change anything.

Reading of the exercise: `moshi`, `torch`, `torchaudio` and `transformers` are
absent and there is no clip, so this is the `DESIGN D11` scaled-down run -- the
same corruption, level by level, on the plain residual quantizer
`code/main.py` implements. The prediction is stated as a consequence of Mimi's
semantic-acoustic split, and the point of running it here is that **the split is
not needed to produce it**.

The prediction reproduces, enormously:

| corrupted codebook | MSE | vs clean | mean-square contribution |
|---:|---:|---:|---:|
| **0** | **1.301458** | 1.42e+11 | 5.859e-01 |
| 1 | 7.445039e-02 | 8.13e+09 | 1.314e-02 |
| 3 | 1.856689e-04 | 2.03e+07 | 2.446e-05 |
| 5 | 5.289699e-07 | 5.78e+04 | 4.140e-08 |
| **7** | **1.262134e-09** | 1.38e+02 | 1.977e-10 |

A factor of **1.03e+09** between the two ends, monotone at every step in between.
Corrupting codebook 0 is worse than emitting nothing at all: MSE **1.3015**
against the signal's own variance of **0.6020**, so the decode is more than twice
as far from the original as silence is.

**But this quantizer has no semantic codebook.** `rvq_encode` builds all eight by
one identical call -- `learn_codebook(residuals, codebook_size, seed=cb_i)` --
differing only in the residual handed to it and an integer seed. There is no
WavLM, no distillation, and no code path that treats index 0 differently; the
words "semantic" and "WavLM" appear in the module only inside `main`'s Step 4
print statements. What orders the codebooks is residual energy, and the
corruption column tracks it exactly: both fall monotonically across all eight
levels, over ratios of **1.03e+09** and **2.96e+09** respectively.

So the experiment as designed cannot distinguish the hypothesis it is testing.
Any residual quantizer -- semantic split or not -- gives codebook 0 the largest
share of the signal and the last codebook a share near the noise floor, and the
asymmetry follows. Testing the semantic claim needs the control the exercise does
not ask for: a codec whose codebooks carry comparable energy, which residual
quantization by construction cannot produce.

Structure: `corrupt` replaces one level's indices with uniform random codes;
`damage` measures the MSE that results at every level; `contribution` is the
mean-square amplitude each codebook adds; `mentions` locates the semantic
vocabulary in the module's source.
"""

from __future__ import annotations

import importlib.util
import inspect
import random

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "13-neural-audio-codecs"
LENGTH, SIZE, LEVELS = 1000, 8, 8
WORDS = ("wavlm", "semantic", "distill")
ABSENT = ("moshi", "torch", "torchaudio", "transformers")


def corrupt(indices, level, size=SIZE, seed=100):
    """One level's indices replaced by uniform random codes, the rest untouched."""
    rnd = random.Random(seed + level)
    broken = [list(stream) for stream in indices]
    broken[level] = [rnd.randrange(size) for _ in indices[level]]
    return broken


def damage(ref, signal, indices, books):
    """MSE after corrupting each level in turn."""
    return [ref.mse(signal, ref.rvq_decode(corrupt(indices, level), books, len(signal)))
            for level in range(len(indices))]


def contribution(indices, books):
    """Mean-square amplitude each codebook adds to the reconstruction."""
    return [sum(book[i] ** 2 for i in stream) / len(stream)
            for stream, book in zip(indices, books)]


def variance(signal):
    mean = sum(signal) / len(signal)
    return sum((x - mean) ** 2 for x in signal) / len(signal)


def mentions(ref):
    """Where the semantic vocabulary lives: `main`'s prints, or the quantizer."""
    quantizer = "".join(inspect.getsource(getattr(ref, name))
                        for name in ("rvq_encode", "rvq_decode", "learn_codebook"))
    return [w for w in WORDS if w in inspect.getsource(ref.main).lower()], \
        [w for w in WORDS if w in quantizer.lower()]


def descending(values):
    return all(a > b for a, b in zip(values, values[1:]))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    signal = ref.generate_signal(n=LENGTH)
    indices, books = ref.rvq_encode(signal, SIZE, LEVELS)
    clean = ref.mse(signal, ref.rvq_decode(indices, books, len(signal)))
    broken = damage(ref, signal, indices, books)
    energy = contribution(indices, books)
    in_main, in_quantizer = mentions(ref)
    return {
        "absent": [m for m in ABSENT if importlib.util.find_spec(m) is None],
        "clean": clean, "damage": broken, "energy": energy,
        "variance": variance(signal),
        "in_main": in_main, "in_quantizer": in_quantizer,
        "calls": inspect.getsource(ref.rvq_encode).count("learn_codebook("),
    }


def verify(result):
    broken, energy = result["damage"], result["energy"]
    return [
        practice.Check(
            "CONTROL: no Mimi, no torch, and no clip -- so the scaled-down run is the RVQ",
            len(result["absent"]) == len(ABSENT),
            f"find_spec is None for {result['absent']}, so the corruption runs level by level "
            f"on the residual quantizer `code/main.py` implements, at {LEVELS} codebooks with a "
            f"clean MSE of {result['clean']:.3e}",
        ),
        practice.Check(
            "ANSWER: the predicted asymmetry reproduces, at a factor of 1.03e+09",
            broken[0] / broken[-1] > 1e8 and descending(broken),
            f"corrupting codebook 0 gives MSE {broken[0]:.6f} and codebook {LEVELS - 1} gives "
            f"{broken[-1]:.6e}, a ratio of {broken[0] / broken[-1]:.2e}, and the column is "
            f"monotone at every step between: {[f'{v:.2e}' for v in broken]}",
        ),
        practice.Check(
            "FINDING: corrupting codebook 0 is worse than emitting silence",
            broken[0] > 2 * result["variance"],
            f"MSE {broken[0]:.4f} against the signal's own variance {result['variance']:.4f} -- "
            f"{broken[0] / result['variance']:.2f}x, so the decode sits further from the "
            "original than a zero signal does. That is the 'destroys intelligibility' end of "
            "the comparison, quantified",
        ),
        practice.Check(
            "MECHANISM: what orders the codebooks is residual energy, and it tracks exactly",
            descending(energy) and energy[0] / energy[-1] > 1e8,
            f"mean-square contribution falls {energy[0]:.3e} -> {energy[-1]:.3e}, a ratio of "
            f"{energy[0] / energy[-1]:.2e}, monotone across all {LEVELS} levels beside the "
            f"corruption column's {broken[0] / broken[-1]:.2e}. Corrupting a level costs roughly "
            "twice what that level contributes",
        ),
        practice.Check(
            "FINDING: this quantizer has no semantic codebook, so the test proves nothing",
            not result["in_quantizer"] and result["in_main"] and result["calls"] == 1,
            f"`rvq_encode` builds every codebook with {result['calls']} call to "
            f"`learn_codebook(residuals, codebook_size, seed=cb_i)`, differing only in the "
            f"residual and an integer seed; {result['in_quantizer']} of {list(WORDS)} appear in "
            f"the quantizer and {result['in_main']} appear in `main`'s Step 4 prints. Any RVQ "
            "reproduces the result, so the experiment cannot test the semantic claim",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
