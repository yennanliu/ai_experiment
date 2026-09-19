"""Exercise 2 — the cutoff is 1.5 Hz, and drumming is gone.

    FAST tokenization compresses 30-step trajectories to ~10 tokens. What does
    the user lose if the trajectory has high-frequency motion (e.g., drumming)?

Reading of the exercise: the lesson ships `fast_compress` and no inverse, so the
inverse DCT is written here in order to measure what the forward pass discards.
The loss is then reported as a cutoff frequency rather than as an error, because
a low-pass filter's damage depends entirely on where the signal's energy sits.

**ANSWER: everything above 1.5 Hz.** `keep_coeff=4` of a 30-step trajectory keeps
DCT bins 0-3, and at a 30 Hz control rate bin k is `k x 30 / (2 x 30)` = k/2 Hz.
So the tokenizer is a low-pass filter with a **1.5 Hz** corner, and drumming at 4
to 8 Hz is entirely in the stopband.

**FINDING: the retained energy falls off a cliff, and the amplitude falls
faster.** Reconstructing a pure sine from its first four coefficients keeps
**99.1%** of its energy at 0.5 Hz, **62.2%** at 2 Hz, **3.0%** at 5 Hz and
**0.7%** at 8 Hz. Truncating the DCT is an orthogonal projection, so kept and
discarded energy sum to the whole at every frequency and that is the honest
figure. Measured as amplitude instead -- `1 - rms(error)/rms(signal)`, which is
what the trajectory looks like -- the same reconstructions read **90.5%**,
**38.5%**, **1.5%** and **0.4%**. A drumming trajectory does not come back
degraded; it comes back as a straight line.

**FINDING: the lesson's own FAST output is 40 tokens, not the ~10 the exercise
names.** `fast_compress(traj, keep_coeff=4)` on its own 30-step, 10-DOF fixture
returns **40** -- four coefficients *per dimension* -- for a compression of
**7.5x** against 300. The exercise's figure is 4x smaller than the function's.

**FINDING: and there is no inverse in the module, so nothing checks it.**
`fast_compress` has no `fast_decompress`; the lesson prints a compression ratio
and never reconstructs. A tokenizer whose forward pass is a low-pass filter and
whose round trip is untested reports only the half of the trade that looks good.

Structure: `idct` is the inverse the module lacks, `retained` reconstructs one
sine from its first coefficients and reports the fraction of energy kept, and
`cutoff` converts a coefficient count into a corner frequency.
"""

from __future__ import annotations

import math
import statistics

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "21-embodied-vlas-openvla-pi0-groot"
STEPS, DOF, CONTROL_HZ = 30, 10, 30
KEEP = 4
FREQUENCIES = (0.5, 2.0, 5.0, 8.0)
EXERCISE_CLAIM = 10


def idct(coeffs, n):
    """The inverse the lesson's module does not ship."""
    out = []
    for index in range(n):
        total = coeffs[0] / 2
        for k in range(1, len(coeffs)):
            total += coeffs[k] * math.cos(math.pi / n * (index + 0.5) * k)
        out.append(2 * total / n)
    return out


def rms(series):
    return math.sqrt(statistics.fmean(value * value for value in series))


def energy(series):
    return sum(value * value for value in series)


def retained(ref, frequency, steps=STEPS, keep=KEEP, rate=CONTROL_HZ):
    """Energy kept, amplitude kept, and whether the two halves sum to the whole."""
    series = [math.sin(2 * math.pi * frequency * step / rate) for step in range(steps)]
    rebuilt = idct(ref.dct(series)[:keep], steps)
    residual = [a - b for a, b in zip(series, rebuilt)]
    whole = energy(series)
    return {"energy": round(energy(rebuilt) / whole, 3),
            "amplitude": round(1 - rms(residual) / rms(series), 3),
            "orthogonal": abs(energy(rebuilt) + energy(residual) - whole) < 1e-9}


def summarise(rows):
    """Split the per-frequency rows into the series the checks assert."""
    return {"retained": {f: row["energy"] for f, row in rows.items()},
            "amplitude": {f: row["amplitude"] for f, row in rows.items()},
            "orthogonal": all(row["orthogonal"] for row in rows.values())}


def cutoff(keep=KEEP, steps=STEPS, rate=CONTROL_HZ):
    return (keep - 1) * rate / (2 * steps)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    trajectory = [[math.sin(0.1 * step + 0.3 * dim) for dim in range(DOF)]
                  for step in range(STEPS)]
    produced = len(ref.fast_compress(trajectory, keep_coeff=KEEP))
    discrete = DOF * STEPS
    rows = {frequency: retained(ref, frequency) for frequency in FREQUENCIES}
    return {
        "keep": KEEP, "steps": STEPS, "cutoff_hz": cutoff(),
        **summarise(rows),
        "stopband": [frequency for frequency in FREQUENCIES if frequency > cutoff()],
        "produced": produced, "claimed": EXERCISE_CLAIM,
        "claim_gap": produced // EXERCISE_CLAIM,
        "per_dim": produced // DOF,
        "discrete": discrete, "ratio": round(discrete / produced, 1),
        "has_inverse": any("decompress" in name or "inverse" in name for name in dir(ref)),
        "forward_only": hasattr(ref, "fast_compress"),
    }


def verify(result):
    kept = result["retained"]
    return [
        practice.Check(
            "ANSWER: everything above 1.5 Hz",
            all([result["cutoff_hz"] == 1.5, result["keep"] == KEEP,
                 result["stopband"] == [2.0, 5.0, 8.0]]),
            f"keep_coeff={result['keep']} of a {result['steps']}-step trajectory keeps DCT "
            f"bins 0-{result['keep'] - 1}, and at {CONTROL_HZ} Hz bin k is k/2 Hz -- a "
            f"low-pass with a {result['cutoff_hz']} Hz corner. Of the frequencies tested, "
            f"{result['stopband']} Hz are in the stopband, and drumming lives at 4 to 8",
        ),
        practice.Check(
            "FINDING: the retained energy falls off a cliff, and the amplitude falls faster",
            all([kept == {0.5: 0.991, 2.0: 0.622, 5.0: 0.03, 8.0: 0.007},
                 result["amplitude"] == {0.5: 0.905, 2.0: 0.385, 5.0: 0.015, 8.0: 0.004},
                 result["orthogonal"]]),
            f"reconstructing a pure sine from its first {KEEP} coefficients keeps "
            f"{kept[0.5]:.1%} of its energy at 0.5 Hz, {kept[2.0]:.1%} at 2 Hz, "
            f"{kept[5.0]:.1%} at 5 and {kept[8.0]:.1%} at 8. Truncating the DCT is an "
            f"orthogonal projection -- kept and discarded energy sum to the whole at every "
            f"frequency -- so that is the honest figure. Measured as amplitude, "
            f"1 - rms(error)/rms(signal), the same reconstructions read "
            f"{list(result['amplitude'].values())}. A drumming trajectory does not come back "
            "degraded; it comes back as a straight line",
        ),
        practice.Check(
            "FINDING: the lesson's own FAST output is 40 tokens, not ~10",
            all([result["produced"] == 40, result["per_dim"] == KEEP,
                 result["claim_gap"] == 4, result["ratio"] == 7.5,
                 result["discrete"] == 300]),
            f"fast_compress on the lesson's own {STEPS}-step, {DOF}-DOF fixture returns "
            f"{result['produced']} tokens -- {result['per_dim']} coefficients per dimension "
            f"-- for {result['ratio']}x against {result['discrete']}. The exercise's "
            f"{result['claimed']} is {result['claim_gap']}x smaller than the function's",
        ),
        practice.Check(
            "FINDING: there is no inverse in the module, so nothing checks it",
            all([not result["has_inverse"], result["forward_only"]]),
            "fast_compress has no fast_decompress anywhere in the module; the lesson prints a "
            "compression ratio and never reconstructs. A tokenizer whose forward pass is a "
            "low-pass filter and whose round trip is untested reports only the half of the "
            "trade that looks good",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
