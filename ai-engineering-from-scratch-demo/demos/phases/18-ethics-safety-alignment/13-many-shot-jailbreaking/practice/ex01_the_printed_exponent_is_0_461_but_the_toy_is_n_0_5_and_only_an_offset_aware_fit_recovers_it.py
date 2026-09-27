"""Exercise 1 — the printed exponent is 0.461, but the toy is n^0.5 and only an offset-aware fit recovers it.

    Run `code/main.py`. Fit a power law to the shot-vs-ASR curve. Report the
    exponent.

Reading of the exercise: "the exponent" is read three ways, because they
disagree: the one `main()` prints on its shipped seed-41 run, the one the same
`fit_power_law` returns on the noiseless curve `target_asr` defines, and the
one in the generating formula ASR(n) = a0 + c * n^alpha. The seed spread is
measured over 1000 reseeded runs of the shipped simulation.

**ANSWER: the shipped run prints ASR ~= 0.038 * n^0.461.** The exponent the
toy is built on is 0.5, and a fit recovers it exactly -- alpha 0.500,
c 0.030 -- only after subtracting the a0 = 0.02 offset from each ASR before
taking logs.

**FINDING: the reference's fit is biased low, and 0.461 is a lucky draw.**
On the noiseless curve `fit_power_law` returns 0.427: the constant a0 flattens
the low-shot end, so the curve is not a power law in log-log space. Over 1000
seeds the fitted exponent averages 0.430, with a 95% range of 0.394-0.473
that excludes the true 0.5; only 7.8% of seeds fit as high as 0.461. On the skill file's grid (5/32/128/256/512 shots)
the noiseless fit is 0.450.

**FINDING: the toy does not produce the numbers the lesson attributes to
it.** The lesson and the `target_asr` docstring say MSJ "fails reliably at 5
shots, begins to succeed around 32, saturates around 256". The toy gives 8.7%
at 5, 19.0% at 32 and 50.0% at 256. It reaches 1.0 only at 1068 shots, past the
largest shot count it runs (512, ASR 69.9%).

Structure: `shipped()` replays `main()` on a seeded `random.Random` swapped
into the reference (restored after) and parses its printout; `seed_alphas()`
reruns the undefended simulation under 1000 seeds.
"""

from __future__ import annotations

import contextlib
import io
import inspect
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "13-many-shot-jailbreaking"
SHOTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
SKILL_GRID = [5, 32, 128, 256, 512]
FIT = r"fitted power law: ASR ~= ([\d.]+) \* n\^([\d.]+)"


def with_rng(ref, seed, fn):
    saved, ref.random = ref.random, random.Random(seed)
    try:
        return fn()
    finally:
        ref.random = saved


def shipped(ref):
    """main()'s printout on its own seed (41, set at import): (c, alpha)."""
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        with_rng(ref, 41, ref.main)
    c, alpha = re.search(FIT, log.getvalue()).groups()
    return float(c), float(alpha)


def seed_alphas(ref, seeds=range(1000)):
    def one():
        return ref.fit_power_law(SHOTS, [ref.simulate(s, ref.target_asr) for s in SHOTS])[0]

    return sorted(with_rng(ref, s, one) for s in seeds)


def saturation_shot(ref):
    """Smallest integer shot count at which target_asr reaches 1.0."""
    return next(k for k in range(1, 5000) if ref.target_asr(k) >= 1.0)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    exact = [ref.target_asr(s) for s in SHOTS]
    alphas, printed = seed_alphas(ref), shipped(ref)
    a0 = inspect.signature(ref.target_asr).parameters["a0"].default
    offset = ref.fit_power_law(SHOTS, [a - a0 for a in exact])
    return {
        "printed": printed,
        "noiseless": round(ref.fit_power_law(SHOTS, exact)[0], 3),
        "offset_fit": (round(offset[0], 3), round(offset[1], 3)),
        "skill_grid": round(ref.fit_power_law(SKILL_GRID, [ref.target_asr(s) for s in SKILL_GRID])[0], 3),
        "seed_mean": round(sum(alphas) / len(alphas), 3),
        "seed_95": (round(alphas[25], 3), round(alphas[974], 3)),
        "above_printed": sum(a >= printed[1] for a in alphas) / len(alphas),
        "asr_at": {n: round(ref.target_asr(n), 3) for n in (5, 32, 256, 512)},
        "saturates_at": saturation_shot(ref),
        "docstring": ref.target_asr.__doc__,
    }


def verify(result):
    lo, hi = result["seed_95"]
    asr = result["asr_at"]
    return [
        practice.Check(
            "ANSWER: the shipped run prints ASR ~= 0.038 * n^0.461; the toy's exponent is 0.5",
            result["printed"] == (0.038, 0.461) and result["offset_fit"] == (0.5, 0.03),
            f"printed (c, alpha) = {result['printed']}; fit of log(ASR - 0.02) gives "
            f"(alpha, c) = {result['offset_fit']}",
        ),
        practice.Check(
            "FINDING: the reference's fit is biased low, and 0.461 is a lucky draw",
            (result["noiseless"], result["seed_mean"], lo, hi, result["skill_grid"], result["above_printed"])
            == (0.427, 0.43, 0.394, 0.473, 0.45, 0.078),
            f"noiseless fit {result['noiseless']}; 1000 seeds mean {result['seed_mean']}, "
            f"95% range {lo}-{hi}, {result['above_printed']:.1%} at or above the printed value; "
            f"skill-file grid fit {result['skill_grid']}",
        ),
        practice.Check(
            "FINDING: the toy does not produce the numbers the lesson attributes to it",
            (asr, result["saturates_at"], "saturates around 256" in result["docstring"])
            == ({5: 0.087, 32: 0.19, 256: 0.5, 512: 0.699}, 1068, True),
            f"ASR by shots {asr}; first shot count at ASR 1.0: {result['saturates_at']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
