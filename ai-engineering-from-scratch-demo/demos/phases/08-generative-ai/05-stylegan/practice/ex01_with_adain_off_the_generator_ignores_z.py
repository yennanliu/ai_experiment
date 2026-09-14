"""Exercise 1 — with AdaIN off the generator ignores z entirely, so the spread is zero.

    **Easy.** Run `code/main.py` with `adain_on=True` and `adain_on=False`.
    Compare the spread of outputs for a fixed latent vs perturbed latent.

Reading of the exercise: "spread of outputs" is read as spread *across latents*,
which is the only reading under which the comparison says anything. The lesson's
own `init_mapping`, `mapping`, `init_synth` and `stylegan_forward` are used at the
lesson's own sizes, with `noise_sigma = 0` so that nothing but `w` can move the
output, over LATENTS independent draws of z.

**ANSWER: with AdaIN off the spread is exactly zero.** Across LATENTS latents the
per-channel standard deviation is **0.0** on all six channels and there is exactly
**one** distinct output. `w` enters `stylegan_forward` only inside the
`if adain_on:` branch, so with the branch off the function never reads its first
argument. With AdaIN on, the same sweep gives six non-zero deviations rising to
**0.266**.

**FINDING: the lesson reports a non-zero spread for that constant.** `main()`
flattens five outputs into one list and takes `mean_std` of the pool, which mixes
variation *across latents* with variation *across channels*. For the AdaIN-off arm
there is no variation of the first kind, so the **0.0122** it prints is the shape
of one frozen vector -- a number that would be unchanged if the sweep ran one
latent or a million.

**FINDING: a perturbed latent is the same measurement.** Perturbing z by
`gauss(0, 0.1)` and re-synthesising moves the output by **0.0** with AdaIN off and
by a non-zero amount with it on. There is no "fixed vs perturbed" contrast to draw
in the off arm, because there is no input.

**CONTROL: the constant is the constant.** With AdaIN off the output equals
`stylegan_forward` run on a zero `w`, bit for bit -- which is what "ignores z"
means stated as an identity rather than as a standard deviation.

Structure: `synthesise` is the lesson's own forward at a fixed noise seed;
`sweep` collects outputs over latents; `columns` is the per-channel spread.
"""

from __future__ import annotations

import statistics

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "05-stylegan"
Z_DIM, W_DIM, HIDDEN, DEPTH, LATENTS, NUDGE = 8, 8, 6, 4, 200, 0.1


def build(ref, random, seed=3):
    """The lesson's own mapping network, synthesis weights and constant input."""
    rng = random.Random(seed)
    mapping = ref.init_mapping(Z_DIM, W_DIM, DEPTH, rng)
    synth = ref.init_synth(HIDDEN, W_DIM, rng)
    return mapping, synth, [rng.gauss(0, 0.3) for _ in range(HIDDEN)], rng


def synthesise(ref, random, parts, z, adain_on):
    """One output, at noise_sigma 0 so only w can move it."""
    mapping, synth, const, _ = parts
    return ref.stylegan_forward(ref.mapping(z, mapping), const, synth, 0.0,
                                random.Random(0), adain_on=adain_on)


def columns(rows):
    """Per-channel standard deviation across the sweep -- spread across latents."""
    return [statistics.pstdev([row[c] for row in rows]) for c in range(HIDDEN)]


def arm(ref, random, parts, latents, nudge, on):
    """(per-channel sd, distinct outputs, pooled sd, movement under a nudged z)."""
    rows = [synthesise(ref, random, parts, z, on) for z in latents]
    moved = synthesise(ref, random, parts, nudge, on)
    return (columns(rows), len({tuple(r) for r in rows}),
            statistics.pstdev([v for r in rows for v in r]),
            max(abs(a - b) for a, b in zip(rows[0], moved)))


def solve():
    import random
    ref = parity.load_reference(PHASE, LESSON, "main")
    parts = build(ref, random)
    rng = parts[3]
    latents = [[rng.gauss(0, 1) for _ in range(Z_DIM)] for _ in range(LATENTS)]
    nudge = [v + rng.gauss(0, NUDGE) for v in latents[0]]
    zeroed = synthesise(ref, random, parts, [0.0] * Z_DIM, False)
    return {"arms": {on: arm(ref, random, parts, latents, nudge, on) for on in (True, False)},
            "constant": synthesise(ref, random, parts, latents[0], False) == zeroed}


def verify(result):
    off, on = result["arms"][False], result["arms"][True]
    return [
        practice.Check(
            "ANSWER: with AdaIN off the spread across latents is exactly zero",
            max(off[0]) == 0.0 and off[1] == 1 and max(on[0]) > 0.2,
            f"over {LATENTS} latents the per-channel sd with AdaIN off is {off[0]} and the sweep "
            f"holds {off[1]} distinct output; with it on the same sweep spreads up to "
            f"{max(on[0]):.3f}. w enters stylegan_forward only inside `if adain_on:`, so with the "
            "branch off the function never reads its first argument",
        ),
        practice.Check(
            "FINDING: the lesson reports a non-zero spread for that constant",
            off[2] > 0.0,
            f"main() flattens the outputs into one list and takes mean_std of the pool, mixing "
            f"spread across latents with spread across channels. For the off arm the first kind "
            f"is zero, so the {off[2]:.4f} it reports is the shape of one frozen vector -- "
            "unchanged whether the sweep runs one latent or a million",
        ),
        practice.Check(
            "FINDING: a perturbed latent is the same measurement",
            off[3] == 0.0 and on[3] > 0.0,
            f"perturbing z by gauss(0, {NUDGE}) and re-synthesising moves the output by "
            f"{off[3]} with AdaIN off and {on[3]:.4f} with it "
            "on. There is no fixed-versus-perturbed contrast to draw in the off arm, because "
            "there is no input to perturb",
        ),
        practice.Check(
            "CONTROL: the constant is the constant",
            result["constant"],
            "with AdaIN off the output for a trained latent equals the output for a zero w, bit "
            "for bit. That is what 'ignores z' means stated as an identity rather than inferred "
            "from a standard deviation that happened to come out small",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
