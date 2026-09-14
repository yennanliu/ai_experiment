"""Exercise 1 — all four betas are Pareto-best, which is what a frontier is.

    **Easy.** Change `β` in `code/main.py` to `0.01`, `0.1`, `1.0`, `5.0`.
    Record the final reconstruction MSE and KL. Which β is Pareto-best for your
    synthetic data?

Reading of the exercise: the four betas are run through the lesson's own
`init_vae`, `forward`, `backward` and `apply_update` at the lesson's own
settings -- 8-D input, 10 hidden, 2-D latent, lr 0.01, 40 epochs, 60 points from
`sample_mixture` -- and the final-epoch means are recorded. Each beta is run on
SEEDS seeds, because one seed does not separate 0.33 from 0.34.

**ANSWER: all four, and that is not a dodge.** Averaged over SEEDS seeds,
reconstruction rises with beta and KL falls with it at every step, so no point in
the sweep is beaten on *both* axes by another. Pareto-optimality does not single out a point -- it names the
set that nothing dominates, and here that set is the whole sweep. The question
as asked has four answers; picking one needs a scalarisation the exercise never
states.

| beta | recon | KL | dominated by |
|---|---:|---:|---|
| 0.01 | 0.325 | 5.127 | nothing |
| 0.1 | 0.342 | 1.811 | nothing |
| 1.0 | 0.511 | 1.266 | nothing |
| 5.0 | 2.005 | 0.736 | nothing |

**FINDING: the left end of the sweep is pinned by the data, not by beta.** Each
dimension carries `gauss(0, 0.2)` noise around its cluster centre, so a model
that recovers the centre exactly still scores `8 * 0.2^2 = 0.32`. beta=0.01
reaches **0.325** -- 1.02x the floor. It is not fitting better than beta=0.1
because it is *allowed* to; it is at a wall the data put there, and the first
half of the sweep therefore buys KL for nearly nothing. The two are not even
separable: on **1 of 5** seeds beta=0.1 reconstructs *better* than beta=0.01,
which is what hitting a floor looks like from underneath.

**FINDING: the exchange rate moves by ~200x across the sweep.** Going 0.01 ->
0.1 spends 0.017 of reconstruction to save 3.32 nats; going 1.0 -> 5.0 spends
1.49 to save 0.53. That is the shape a frontier has and the number the exercise
should have asked for: **beta=0.1 is the last point bought at a good rate**, and
saying so requires admitting you scalarised.

**CONTROL: the posterior collapses at beta=20.** KL falls to **6.3e-05** --
four orders of magnitude below the smallest value anywhere in the sweep -- and
reconstruction reaches **0.98x** the score of a model that ignores its input
entirely. So the KL axis has no floor at the one bit the two clusters actually
carry: `ln 2 = 0.693` is close to the 0.736 that beta=5 lands on, but that is a
coincidence of where the sweep stops, not a rate-distortion bound. Push beta
further and the encoder is switched off rather than compressed.

Structure: `train` is the lesson's own loop; `sweep` runs it over seeds;
`dominated` is the Pareto test the exercise's question needs.
"""

from __future__ import annotations

import statistics

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "02-autoencoders-vae"
IN_DIM, HIDDEN, Z_DIM, POINTS = 8, 10, 2, 60
EPOCHS, RATE, NOISE, SEEDS = 40, 0.01, 0.2, 5
BETAS, COLLAPSE_BETA = (0.01, 0.1, 1.0, 5.0), 20.0


def train(ref, random, beta, seed):
    """The lesson's own training loop; returns its final-epoch (recon, KL) means."""
    rng = random.Random(seed)
    params = ref.init_vae(IN_DIM, HIDDEN, Z_DIM, rng)
    data = ref.sample_mixture(POINTS, IN_DIM, rng)
    recons, kls = [], []
    for _ in range(EPOCHS):
        recons, kls = [], []
        for x in data:
            fwd = ref.forward(x, params, [rng.gauss(0, 1) for _ in range(Z_DIM)])
            _, recon, kl = ref.loss_value(x, fwd, beta)
            ref.apply_update(params, ref.backward(x, fwd, params, beta), RATE)
            recons.append(recon)
            kls.append(kl)
    return statistics.fmean(recons), statistics.fmean(kls)


def sweep(ref, random, betas, seeds=SEEDS):
    """{beta: (mean recon, mean KL, every seed's recon, every seed's KL)}."""
    out = {}
    for beta in betas:
        pairs = [train(ref, random, beta, seed) for seed in range(seeds)]
        recons, kls = [p[0] for p in pairs], [p[1] for p in pairs]
        out[beta] = (statistics.fmean(recons), statistics.fmean(kls), recons, kls)
    return out


def dominated(points):
    """{beta: the betas that beat it on recon *and* KL} -- the Pareto test."""
    return {b: [o for o in points if points[o][0] < points[b][0]
                and points[o][1] < points[b][1]] for b in points}


def solve():
    import random
    ref = parity.load_reference(PHASE, LESSON, "main")
    grid = sweep(ref, random, BETAS)
    means = {b: (v[0], v[1]) for b, v in grid.items()}
    order = sorted(BETAS)
    rates = [(round((means[hi][1] - means[lo][1]) / (means[lo][0] - means[hi][0]), 2))
             for lo, hi in zip(order, order[1:])]        # nats of KL per unit recon
    data = ref.sample_mixture(POINTS, IN_DIM, random.Random(7))
    return {
        "means": means, "dominated": dominated(means), "rates": rates,
        "floor": IN_DIM * NOISE ** 2,
        "ignores_input": statistics.fmean(sum(v * v for v in x) for x in data),
        "collapse": train(ref, random, COLLAPSE_BETA, 0),
        "monotone": all(means[lo][0] < means[hi][0] and means[lo][1] > means[hi][1]
                        for lo, hi in zip(order, order[1:])),
        "floor_ties": sum(grid[order[0]][2][s] >= grid[order[1]][2][s] for s in range(SEEDS)),
    }


def verify(result):
    means, floor = result["means"], result["floor"]
    low = means[BETAS[0]]
    collapse_recon, collapse_kl = result["collapse"]
    return [
        practice.Check(
            "ANSWER: all four are Pareto-best -- nothing in the sweep is dominated",
            not any(result["dominated"].values()) and result["monotone"],
            "recon rises and KL falls at every step of the sweep -- "
            + ", ".join(f"beta {b}: {means[b][0]:.3f} / {means[b][1]:.3f}" for b in BETAS)
            + " -- so no point is beaten on both axes. Pareto-optimality names the set nothing "
              "dominates, not a winner; choosing one of the four needs a scalarisation the "
              "exercise does not state",
        ),
        practice.Check(
            "FINDING: the recon axis is pinned by the data, not by beta",
            low[0] < 1.15 * floor,
            f"every dimension carries gauss(0, {NOISE}) around its cluster centre, so a model that "
            f"recovers the centre exactly still scores {IN_DIM} * {NOISE}^2 = {floor:.2f}. "
            f"beta={BETAS[0]} reaches {low[0]:.3f}, {low[0] / floor:.2f}x the floor -- a wall the "
            f"data put there. The two floor-end betas are not even separable: on "
            f"{result['floor_ties']} of {SEEDS} seeds beta={BETAS[1]} reconstructs better than "
            f"beta={BETAS[0]}, which is why the first half of the sweep costs so little",
        ),
        practice.Check(
            "FINDING: the exchange rate collapses across the sweep, and the elbow is beta=0.1",
            result["rates"][0] > 20 * result["rates"][-1],
            f"nats of KL bought per unit of reconstruction given up, between adjacent betas: "
            f"{result['rates']}. The first step is {result['rates'][0] / result['rates'][-1]:.0f}x "
            f"the rate of the last, so beta={BETAS[1]} is the last point bought cheaply. That is "
            "the number the question wanted, and quoting it means admitting a scalarisation",
        ),
        practice.Check(
            "CONTROL: at beta=20 the posterior collapses outright, so the KL axis has no floor",
            collapse_kl < means[BETAS[-1]][1] / 1000
            and collapse_recon > 0.95 * result["ignores_input"],
            f"beta={COLLAPSE_BETA} drives KL to {collapse_kl:.1e}, "
            f"{means[BETAS[-1]][1] / collapse_kl:,.0f}x below the smallest value in the sweep, and "
            f"reconstruction to {collapse_recon:.3f} against {result['ignores_input']:.3f} for a "
            f"model that ignores its input altogether. The two clusters carry one bit and ln 2 is "
            f"near the {means[BETAS[-1]][1]:.3f} beta={BETAS[-1]} lands on, but nothing holds the "
            "code there -- past some beta the encoder is switched off rather than compressed",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
