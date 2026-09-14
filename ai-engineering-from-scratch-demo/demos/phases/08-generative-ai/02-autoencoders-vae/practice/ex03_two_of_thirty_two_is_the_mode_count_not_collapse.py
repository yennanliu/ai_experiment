"""Exercise 3 — 2 of 32 codes is the mode count, not codebook collapse.

    **Hard.** Extend `code/main.py` into a mini VQ-VAE: replace the continuous
    `z` with a nearest-neighbour lookup in a codebook of K=32 entries. Compare
    reconstruction MSE and report how many codebook entries get used (codebook
    collapse is real).

Reading of the exercise: the lesson's own network is kept and only the latent is
replaced. `forward` is called with `eps = 0`, which makes `z` equal `mu`
exactly, the nearest of K entries is substituted for it, and the decoder is
re-run from that entry. The gradient is then the lesson's own `backward` at
`beta = 0`: with `eps = 0` its `dz/dmu` is 1, which *is* the straight-through
estimator, and `beta = 0` removes the KL a VQ-VAE does not have. The CONTROL
checks `z == mu` bit-for-bit rather than assuming it.

**ANSWER: reconstruction 0.327, on 2 of the 32 entries.** That is **1.02x** the
`8 * 0.2^2 = 0.32` noise floor the data imposes -- the same wall the continuous
VAE hits in Exercise 1 -- reached with a latent of **one bit** instead of two
continuous dimensions.

**FINDING: 2 of 32 is the number of modes, and the data has two.** The exercise
warns that "codebook collapse is real", and it is, but not here:
`sample_mixture` draws from two centres, so two codes is the correct answer and a
codebook that used all 32 would be encoding the `gauss(0, 0.2)` noise around
them. **A usage fraction cannot diagnose collapse on its own** -- 2/32 is a
pathology only if the data has more than two modes, which is a fact about the
data and not about the model.

**FINDING: K does not matter at all on this data.** Sweeping K across the whole
range moves reconstruction by **0.0004** and leaves usage at 2:

| K | 2 | 8 | 32 | 128 |
|---|---:|---:|---:|---:|
| recon | 0.3275 | 0.3273 | 0.3272 | 0.3271 |
| used | 2 | 2 | 2 | 2 |

So the exercise's K=32 is neither better nor worse than K=2, and 30 of its
entries are memory that never gets addressed.

**FINDING: one bit buys what 5.1 nats of continuous code bought.** Exercise 1's
beta=0.01 arm reaches 0.325 while paying **5.13 nats** of KL. This reaches 0.327
at `log2(2) = 1 bit = 0.69 nats`. Same distortion, **7x** less rate: nearly all
of the continuous latent's capacity was spent on noise it could not use.

Structure: `nearest` is the lookup; `quantise` swaps the code in and re-runs the
lesson's decoder; `train` is the lesson's loop around both.
"""

from __future__ import annotations

import math
import statistics

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "02-autoencoders-vae"
IN_DIM, HIDDEN, Z_DIM, POINTS, NOISE = 8, 10, 2, 60, 0.2
EPOCHS, RATE, BOOK_RATE, SEEDS, CODES, SIZES, VAE_KL = 40, 0.01, 0.1, 3, 32, (2, 8, 32, 128), 5.13


def nearest(book, z):
    """Index of the codebook entry closest to `z` in squared distance."""
    return min(range(len(book)),
               key=lambda i: sum((a - b) ** 2 for a, b in zip(book[i], z)))


def quantise(ref, params, fwd, code):
    """Re-run the lesson's own decoder from a codebook entry instead of from z."""
    dec = params["dec"]
    hidden = ref.tanh(ref.add(ref.matmul(dec["W1"], code), dec["b1"]))
    x_hat = ref.add(ref.matmul(dec["W_out"], hidden), dec["b_out"])
    return dict(fwd, z=code, h_dec=hidden, x_hat=x_hat)


def train(ref, random, size, seed):
    """(final-epoch recon, entries actually addressed) for a codebook of `size`."""
    rng = random.Random(seed)
    params = ref.init_vae(IN_DIM, HIDDEN, Z_DIM, rng)
    data = ref.sample_mixture(POINTS, IN_DIM, rng)
    book = [[rng.gauss(0, 0.5) for _ in range(Z_DIM)] for _ in range(size)]
    recons = []
    for _ in range(EPOCHS):
        recons = []
        for x in data:
            fwd = ref.forward(x, params, [0.0] * Z_DIM)       # eps = 0, so z == mu
            index = nearest(book, fwd["z"])
            quantised = quantise(ref, params, fwd, book[index])
            recons.append(sum((a - b) ** 2 for a, b in zip(x, quantised["x_hat"])))
            ref.apply_update(params, ref.backward(x, quantised, params, 0.0), RATE)
            for j in range(Z_DIM):                            # entry toward its assignments
                book[index][j] -= BOOK_RATE * 2 * (book[index][j] - fwd["z"][j])
    used = {nearest(book, ref.forward(x, params, [0.0] * Z_DIM)["z"]) for x in data}
    return statistics.fmean(recons), len(used)


def solve():
    import random
    ref = parity.load_reference(PHASE, LESSON, "main")
    grid = {}
    for size in SIZES:
        rows = [train(ref, random, size, seed) for seed in range(SEEDS)]
        grid[size] = (statistics.fmean(r for r, _ in rows), [u for _, u in rows])
    probe = ref.forward(ref.sample_mixture(1, IN_DIM, random.Random(1))[0],
                        ref.init_vae(IN_DIM, HIDDEN, Z_DIM, random.Random(1)), [0.0] * Z_DIM)
    recons = [grid[s][0] for s in SIZES]
    return {
        "grid": grid, "floor": IN_DIM * NOISE ** 2, "spread": max(recons) - min(recons),
        "straight_through": max(abs(a - b) for a, b in zip(probe["z"], probe["mu"])),
        "bits": math.log2(statistics.fmean(grid[CODES][1])),
    }


def verify(result):
    grid, floor = result["grid"], result["floor"]
    recon, used = grid[CODES]
    return [
        practice.Check(
            f"ANSWER: reconstruction {recon:.3f} on 2 of the {CODES} entries",
            set(used) == {2} and recon < 1.1 * floor,
            f"a codebook of {CODES} settles on {used} entries across {SEEDS} seeds and "
            f"reconstructs to {recon:.4f}, {recon / floor:.2f}x the {IN_DIM} * {NOISE}^2 = "
            f"{floor:.2f} the noise imposes. That is the same wall Exercise 1's continuous VAE "
            f"hits, reached with a latent of {result['bits']:.0f} bit instead of {Z_DIM} "
            "continuous dimensions",
        ),
        practice.Check(
            "FINDING: 2 of 32 is the mode count, and sample_mixture has two modes",
            set(used) == {2},
            f"the exercise warns that codebook collapse is real, and it is, but 2 is the right "
            f"answer here: sample_mixture draws from two centres, so a codebook using all "
            f"{CODES} would be encoding the gauss(0, {NOISE}) noise around them. A usage fraction "
            "cannot diagnose collapse alone -- 2/32 is a pathology only if the data has more "
            "than two modes, which is a fact about the data, not about the model",
        ),
        practice.Check(
            "FINDING: K does not matter -- 2 and 128 reconstruct the same",
            result["spread"] < 0.005 and all(set(grid[s][1]) <= {2, 3} for s in SIZES),
            "recon and entries used across the whole range of K -- "
            + ", ".join(f"K={s}: {grid[s][0]:.4f} on {grid[s][1]}" for s in SIZES)
            + f" -- a spread of {result['spread']:.4f}. The exercise's K={CODES} is neither "
              f"better nor worse than K={SIZES[0]}, and {CODES - 2} of its entries are memory "
              "that is never addressed",
        ),
        practice.Check(
            "FINDING: one bit buys what 5.1 nats of continuous code bought",
            result["bits"] * math.log(2) < VAE_KL / 5,
            f"Exercise 1's beta=0.01 arm reaches 0.325 while paying {VAE_KL} nats of KL; this "
            f"reaches {recon:.3f} at log2(2) = {result['bits']:.0f} bit = "
            f"{result['bits'] * math.log(2):.2f} nats, "
            f"{VAE_KL / (result['bits'] * math.log(2)):.1f}x less rate for the same distortion. "
            "Almost all of the continuous latent's capacity went on noise it could not use",
        ),
        practice.Check(
            "CONTROL: the straight-through estimator is the lesson's own backward at eps = 0",
            result["straight_through"] == 0.0,
            f"forward(x, params, eps=0) returns z equal to mu to {result['straight_through']}, so "
            f"the lesson's dz/dmu of 1 *is* the straight-through gradient and beta=0 removes the "
            "KL a VQ-VAE does not have. The quantiser is a substitution into ref.backward, not a "
            "second implementation of it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
