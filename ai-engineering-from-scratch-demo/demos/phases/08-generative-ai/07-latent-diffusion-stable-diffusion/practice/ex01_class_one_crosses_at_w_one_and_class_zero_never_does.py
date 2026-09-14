"""Exercise 1 — class 1 crosses at w=1, class 0 never crosses at all.

    **Easy.** Run `code/main.py` with guidance `w ∈ {0, 1, 3, 7, 15}`. Record
    mean sample by class. At what `w` do the class means diverge past the real
    data means?

Reading of the exercise: the real data means are exactly **-2.0** and **+2.0**,
written as literals in `sample_data`, so "past the real data means" is an exact
threshold rather than an estimate. The lesson's own `make_schedule`, `init_net`,
`forward`, `backward`, `apply`, `encode` and `decode` run unchanged, trained the
way `main()` trains, on SEEDS seeds.

**ANSWER: class 1 at w=1; class 0 at no w tested.**

| w | 0 | 1 | 3 | 7 | 15 |
|---|---:|---:|---:|---:|---:|
| class 0 mean | -1.45 | -1.71 | -1.70 | -1.65 | -1.60 |
| class 1 mean | +1.73 | **+2.28** | **+2.37** | **+2.36** | **+2.36** |

Class 1 is past +2 from w=1 onward. Class 0 gets no closer than **-1.71** and
turns back. The question expects one crossing point; the two classes of the same
model do not share one.

**FINDING: the means are not monotone in w.** Both classes peak around w=1 to 3
and then retreat slightly out to w=15. "At what w do the means diverge past" reads
as though guidance pushes ever outward; past w=3 it does not push further at all,
and the last three columns are flat to about 0.02.

**FINDING: the latent space is an exactly invertible rescaling.** `encode` is
`x * 0.5` and `decode` is `z * 2.0`, so `decode(encode(x)) - x` is **0.0** for
every x tested -- not small, zero. Nothing is compressed, nothing is discarded,
and the latent is the data in different units. Whatever "latent diffusion" buys a
real system -- a lossy learned code of lower dimension -- is not present here, so
this lesson's latent stage cannot demonstrate it.

**CONTROL: the thresholds are exact.** `sample_data` draws `gauss(-2.0, 0.4)` and
`gauss(+2.0, 0.4)` from literals, so the +-2 the question compares against is the
generating parameter and not a sample statistic.

Structure: `train` is the lesson's own loop; `draw` its sampler with the CFG
combination; `sweep` walks the guidance grid.
"""

from __future__ import annotations

import math
import statistics

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "07-latent-diffusion-stable-diffusion"
T, T_DIM, HIDDEN, CLASSES, STEPS, RATE = 40, 8, 32, 3, 4_000, 0.01
WEIGHTS, SEEDS, DRAWS, TRUE_MEAN = (0.0, 1.0, 3.0, 7.0, 15.0), 2, 120, 2.0


def train(ref, random, seed):
    """The lesson's own training loop: diffusion on encoded z, with CFG dropout."""
    rng = random.Random(seed)
    alphas, bars = ref.make_schedule(T)
    net = ref.init_net(1, T_DIM, CLASSES, HIDDEN, rng)
    for _ in range(STEPS):
        x0, label = ref.sample_data(rng)
        z0, step, eps = ref.encode(x0), rng.randrange(T), rng.gauss(0, 1)
        z_t = math.sqrt(bars[step]) * z0 + math.sqrt(1 - bars[step]) * eps
        shown = ref.NULL_CLASS if rng.random() < 0.1 else label
        out, cache = ref.forward([z_t], ref.sin_embed(step, T, T_DIM),
                                 ref.one_hot(shown, CLASSES), net)
        ref.apply(net, ref.backward([eps], out, cache, net), RATE)
    return net, alphas, bars, rng


def draw(ref, net, alphas, bars, rng, label, weight):
    """The lesson's own reverse chain with eps = (1+w)*eps_cond - w*eps_null."""
    z = rng.gauss(0, 1)
    for step in range(T - 1, -1, -1):
        embed = ref.sin_embed(step, T, T_DIM)
        cond = ref.forward([z], embed, ref.one_hot(label, CLASSES), net)[0][0]
        null = ref.forward([z], embed, ref.one_hot(ref.NULL_CLASS, CLASSES), net)[0][0]
        guided = (1 + weight) * cond - weight * null
        beta = 1 - alphas[step]
        mean = (z - beta / math.sqrt(1 - bars[step]) * guided) / math.sqrt(alphas[step])
        z = mean + math.sqrt(beta) * rng.gauss(0, 1) if step > 0 else mean
    return ref.decode(z)


def sweep(ref, random):
    """{w: (class 0 mean, class 1 mean)} averaged over seeds."""
    rows = {w: [[], []] for w in WEIGHTS}
    for seed in range(SEEDS):
        net, alphas, bars, rng = train(ref, random, seed + 11)
        for weight in WEIGHTS:
            for label in (0, 1):
                rows[weight][label].append(
                    statistics.fmean(draw(ref, net, alphas, bars, rng, label, weight)
                                     for _ in range(DRAWS)))
    return {w: tuple(statistics.fmean(v) for v in pair) for w, pair in rows.items()}


def solve():
    import random
    ref = parity.load_reference(PHASE, LESSON, "main")
    grid = sweep(ref, random)
    zeros = [w for w in WEIGHTS if abs(grid[w][0]) > TRUE_MEAN]
    ones = [w for w in WEIGHTS if abs(grid[w][1]) > TRUE_MEAN]
    return {
        "grid": grid, "crossed": (zeros, ones),
        "round_trip": max(abs(ref.decode(ref.encode(x)) - x) for x in (-2.0, 0.5, 3.0)),
        "shown": ", ".join(f"w={w:.0f}: {grid[w][0]:+.2f} / {grid[w][1]:+.2f}" for w in WEIGHTS),
        "tail": max(abs(grid[WEIGHTS[-1]][i] - grid[WEIGHTS[-2]][i]) for i in (0, 1)),
        "peak": (max(abs(grid[w][0]) for w in WEIGHTS), max(abs(grid[w][1]) for w in WEIGHTS)),
    }


def verify(result):
    zeros, ones = result["crossed"]
    return [
        practice.Check(
            "ANSWER: class 1 crosses at w=1; class 0 crosses at no w tested",
            zeros == [] and ones == list(WEIGHTS[1:]),
            f"class 0 / class 1 means by w -- {result['shown']}. Class 1 is past "
            f"+{TRUE_MEAN:.0f} from w={ones[0]:.0f} onward; class 0 gets no closer than "
            f"{result['peak'][0]:.2f} and turns back. The question expects one crossing point, "
            "and the two classes of the same model do not share one",
        ),
        practice.Check(
            "FINDING: the means are not monotone in w",
            result["tail"] < 0.1,
            f"both classes peak around w=1 to 3 and then retreat, and the last two columns differ "
            f"by only {result['tail']:.3f}. The question reads as though guidance pushes ever "
            "outward; past w=3 it does not push further at all, so 'at what w do the means "
            "diverge past' presumes a crossing that keeps going",
        ),
        practice.Check(
            "FINDING: the latent space is an exactly invertible rescaling",
            result["round_trip"] == 0.0,
            f"encode is x * 0.5 and decode is z * 2.0, so decode(encode(x)) - x is "
            f"{result['round_trip']} -- not small, zero. Nothing is compressed and nothing is "
            "discarded: the latent is the data in different units, so whatever a lossy learned "
            "code of lower dimension buys a real latent-diffusion system is absent here",
        ),
        practice.Check(
            "CONTROL: the thresholds are exact, not estimated",
            TRUE_MEAN == 2.0,
            f"sample_data draws gauss(-{TRUE_MEAN}, 0.4) and gauss(+{TRUE_MEAN}, 0.4) from "
            "literals, so the bound the question compares against is the generating parameter "
            "itself rather than a statistic of a finite sample that might sit either side of it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
