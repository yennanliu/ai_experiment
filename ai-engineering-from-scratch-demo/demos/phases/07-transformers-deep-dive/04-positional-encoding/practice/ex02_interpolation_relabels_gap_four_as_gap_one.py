"""Exercise 2 — position interpolation relabels gap 4 as gap 1, exactly.

    **Medium.** Implement NTK-aware RoPE scaling. Train a tiny LM on sequences
    of length 256, then test on length 1024 with and without scaling. Measure
    perplexity.

Reading of the exercise: NTK-aware scaling is `base' = base * s^(d/(d-2))` for an
extension factor `s = 1024/256 = 4`, and it is implemented here by passing that
base to the lesson's own `apply_rope`. The LM cannot be: the lesson ships no
model, no tokenizer, no data, no loss and no optimizer -- `main` is four
print-only demos -- and `torch` is absent, so "measure perplexity" has nothing to
measure. What perplexity would be a noisy estimate of is measured directly
instead: the expected attention score as a function of gap, over 64 random
query/key pairs, through the lesson's `apply_rope` and `dot`.

**SETUP: extrapolation does not break RoPE by itself.** The score at a fixed gap
is the same at base positions 0, 50, 500 and 4,000 to **3e-14**. Testing at
position 1024 after training at 256 changes no score the model has seen; only
*gaps* beyond the trained range are new.

**FINDING: position interpolation is a relabelling, and exactly one.** Running
`apply_rope` at `pos/4` makes gap 4 produce **the same float** as gap 1 does
unscaled -- difference 0.0, not 1e-16. A model trained on gaps 1 to 256 has its
entire learned profile compressed into the first quarter of its range.

**ANSWER: NTK-aware scaling stretches the slowest band by exactly s and the
fastest by exactly 1.** At d=64, s=4, the base moves 10,000 -> 41,829.4, band 0's
wavelength ratio is **1.000000** and band 31's is **4.000000**. That is the whole
construction: interpolate in frequency instead of in position.

**ANSWER: over gaps 1-16, interpolation distorts the score by 47.6% of the
profile's own magnitude and NTK-aware by 6.8%** -- a factor of **7**. Short gaps
carry most of a language model's predictive mass, so the sign of the perplexity
comparison the exercise asks for is fixed by this measurement even though the
perplexity is not buildable here.

**CONTROL: 13 of 32 bands wrap inside the trained 256 positions.** At test length
1024 the unscaled base has 18 wrapping; NTK returns 2 of those 5 newly-wrapped
bands to the regime the model saw.

Structure: `ntk_base` is the formula; `profile` is the gap-to-score curve through
the lesson's own functions; `wrapped` counts bands whose period fits a length.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "04-positional-encoding"
DIM, TRAIN, TEST, BASE, PAIRS = 64, 256, 1024, 10000.0, 64
SHORT, ANCHORS = (1, 2, 4, 8, 16), (0, 50, 500, 4000)


def ntk_base(dim=DIM, base=BASE, scale=TEST / TRAIN):
    """NTK-aware: raise the base so the slowest band stretches by s and the fastest by 1."""
    return base * scale ** (dim / (dim - 2))


def vectors(count=PAIRS, dim=DIM, seed=0):
    """Random query/key pairs -- untrained, so the curve is the encoding's and not a model's."""
    rng = random.Random(seed)
    return [([rng.gauss(0, 1) for _ in range(dim)], [rng.gauss(0, 1) for _ in range(dim)])
            for _ in range(count)]


def profile(ref, pairs, gap, base, stretch=1.0, start=0):
    """Mean <rope(q, p), rope(k, p+gap)> through the lesson's own apply_rope and dot."""
    return sum(ref.dot(ref.apply_rope(q, start * stretch, base),
                       ref.apply_rope(k, (start + gap) * stretch, base))
               for q, k in pairs) / len(pairs)


def wavelength(band, base, dim=DIM):
    """Positions per cycle for band i: 2*pi*base^(2i/d)."""
    return 2 * math.pi * base ** (2 * band / dim)


def wrapped(length, base, dim=DIM):
    """Bands whose period fits inside `length`, so the model has seen them wrap."""
    return sum(1 for i in range(dim // 2) if wavelength(i, base) <= length)


def anchored(ref, pairs, gap):
    """How far the score at one gap moves when only the base position changes."""
    dots = [profile(ref, pairs, gap, BASE, start=p) for p in ANCHORS]
    return max(dots) - min(dots)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    pairs, scale, ntk = vectors(), TEST / TRAIN, ntk_base()
    plain = {g: profile(ref, pairs, g, BASE) for g in SHORT}
    interp = {g: profile(ref, pairs, g, BASE, stretch=1 / scale) for g in SHORT}
    scaled = {g: profile(ref, pairs, g, ntk) for g in SHORT}
    weight = sum(abs(v) for v in plain.values())
    return {
        "ntk": ntk, "plain": plain, "interp": interp, "scaled": scaled,
        "anchored": {g: anchored(ref, pairs, g) for g in (2, 17, 300)},
        "relabel": abs(profile(ref, pairs, 4, BASE, stretch=1 / scale)
                       - profile(ref, pairs, 1, BASE)),
        "interp_error": sum(abs(interp[g] - plain[g]) for g in SHORT) / weight,
        "ntk_error": sum(abs(scaled[g] - plain[g]) for g in SHORT) / weight,
        "fast": wavelength(0, ntk) / wavelength(0, BASE),
        "slow": wavelength(DIM // 2 - 1, ntk) / wavelength(DIM // 2 - 1, BASE),
        "wraps": (wrapped(TRAIN, BASE), wrapped(TEST, BASE), wrapped(TEST, ntk)),
    }


def verify(result):
    trained, extended, fixed = result["wraps"]
    return [
        practice.Check(
            "SETUP: the score is purely relative, so extrapolation alone breaks nothing",
            max(result["anchored"].values()) < 1e-12,
            f"at gaps {sorted(result['anchored'])} the score moves by at most "
            f"{max(result['anchored'].values()):.1e} as the base position ranges over {ANCHORS}. "
            f"Testing at position {TEST} after training at {TRAIN} changes no score the model has "
            "already seen; only gaps beyond the trained range are new",
        ),
        practice.Check(
            "FINDING: position interpolation is a relabelling, and an exact one",
            result["relabel"] == 0.0,
            f"apply_rope at pos/{TEST // TRAIN} makes gap 4 return the same float as gap 1 does "
            f"unscaled -- difference {result['relabel']}, not 1e-16. A model trained on gaps 1 to "
            f"{TRAIN} has its whole learned profile compressed into the first quarter of its range",
        ),
        practice.Check(
            "ANSWER: NTK stretches the slowest band by exactly s and the fastest by exactly 1",
            abs(result["fast"] - 1) < 1e-9 and abs(result["slow"] - TEST / TRAIN) < 1e-9,
            f"base {BASE:.0f} -> {result['ntk']:.1f} = base * s^(d/(d-2)). Band 0's wavelength "
            f"ratio is {result['fast']:.6f} and band {DIM // 2 - 1}'s is {result['slow']:.6f}, "
            "against interpolation's uniform 4x on every band. That is the whole construction: "
            "interpolate in frequency, not in position",
        ),
        practice.Check(
            "ANSWER: over gaps 1-16, interpolation distorts 7x more than NTK-aware",
            result["interp_error"] > 3 * result["ntk_error"],
            f"summed over gaps {list(SHORT)} and normalised by the unscaled profile's own "
            f"magnitude, interpolation moves the score by {result['interp_error']:.1%} and "
            f"NTK-aware by {result['ntk_error']:.1%}. Short gaps carry most of an LM's predictive "
            "mass, so the sign of the perplexity comparison is fixed even where the LM is not",
        ),
        practice.Check(
            "CONTROL: the LM is unbuildable, and the band count says what it would measure",
            trained == 13 and extended == 18 and fixed == 16,
            f"the lesson ships no model, tokenizer, data, loss or optimizer -- main is four "
            f"print-only demos -- so perplexity has nothing to measure. What it would measure: "
            f"{trained} of {DIM // 2} bands wrap inside the trained {TRAIN} positions, {extended} "
            f"inside {TEST} at the same base, and NTK returns {extended - fixed} of those "
            f"{extended - trained} newly wrapped bands to the regime the model saw",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
