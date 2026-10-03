"""Exercise 1 — never equal: one pinned dim still names the cluster.

    **Easy.** In `code/main.py`, vary the fraction of dimensions masked from 0.2
    to 0.8. At what fraction does the inpaint quality (residual in masked dims)
    equal unconditional generation?

Reading of the exercise: the lesson's data has `d = 5`, so the fractions 0.2 to
0.8 are exactly 1 to 4 masked dimensions (the trailing ones, as `main()` masks
dims 3-4). The DDPM is trained exactly as `main()` trains it (seed 5, 5000
steps), then the lesson's own `inpaint` and `sample_unconditional` are run on the
same held-out sources. "Residual" is the mean `|generated - source|` over the
masked dims, and the unconditional baseline is scored on the same dims.

**ANSWER: at no fraction in 0.2-0.8.** Masked residual against unconditional:

| masked | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
|---|---:|---:|---:|---:|---:|
| inpaint | 0.339 | 0.365 | 0.438 | 0.640 | 1.173 |
| unconditional | 1.187 | 1.164 | 1.134 | 1.119 | 1.106 |

Even with a single pinned dim (0.8) inpainting is still **43%** better. The two
only meet at fraction 1.0, where there is nothing left to pin (the 1.0 column
is two independent runs of the same sampler, hence 1.173 vs 1.106).

**FINDING: the information never runs out -- the replacement trick does.** In
`sample_data` every dim is the cluster centre plus N(0, 0.2), so one pinned dim
names the cluster with probability `1 - Phi(-5)`. A perfect conditional sampler
would therefore score `2 * 0.2 / sqrt(pi) = 0.226` at *every* fraction from 0.2
to 0.8. The measured rise from 0.339 to 0.640 is the lesson's naive reinjection
getting the cluster wrong more often (cluster match falls from **98%** to
**71%**), not the context running out of information.

**CONTROL: at fraction 1.0 inpaint *is* unconditional generation.** With every
dim masked, the lesson's `inpaint` and `sample_unconditional` make the same
random draws and the same arithmetic: from one seed they agree to **0.0**. The
unconditional residual (1.106) also matches its closed form, half the draws
landing in the right cluster (0.226) and half in the wrong one (~2.0).

Structure: `trained` reproduces `main()`'s model; `score` runs one fraction.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "09-inpainting-outpainting-editing"
T, T_DIM, HIDDEN, D, TRIALS, NOISE = 40, 8, 32, 5, 200, 0.2


def trained(ref, random):
    """The model `main()` trains: same seed, same schedule, same 5000 steps."""
    rng = random.Random(5)
    alphas, bars = ref.make_schedule(T)
    net = ref.init_net(D, T_DIM, HIDDEN, rng)
    ref.train(net, bars, T, steps=5000, lr=0.01, t_dim=T_DIM, d=D, rng=rng)
    return net, alphas, bars


def score(ref, model, masked, rng):
    """(inpaint residual, unconditional residual, inpaint cluster match) on `masked` dims."""
    net, alphas, bars = model
    mask = [i >= D - masked for i in range(D)]
    dims = [i for i in range(D) if mask[i]]
    totals = [0.0, 0.0, 0]
    for _ in range(TRIALS):
        clean, cluster = ref.sample_data(rng, D)
        out = ref.inpaint(net, alphas, bars, T, T_DIM, D, clean, mask, rng)
        free = ref.sample_unconditional(net, alphas, bars, T, T_DIM, D, rng)
        totals[0] += sum(abs(out[i] - clean[i]) for i in dims) / len(dims)
        totals[1] += sum(abs(free[i] - clean[i]) for i in dims) / len(dims)
        totals[2] += all((out[i] > 0) == bool(cluster) for i in dims)
    return totals[0] / TRIALS, totals[1] / TRIALS, totals[2] / TRIALS


def solve():
    import random

    ref = parity.load_reference(PHASE, LESSON, "main")
    model = trained(ref, random)
    net, alphas, bars = model
    rows = {k / D: score(ref, model, k, random.Random(11)) for k in range(1, D + 1)}
    clean, _ = ref.sample_data(random.Random(1), D)
    full = ref.inpaint(net, alphas, bars, T, T_DIM, D, clean, [True] * D, random.Random(3))
    free = ref.sample_unconditional(net, alphas, bars, T, T_DIM, D, random.Random(3))
    return {
        "rows": rows,
        "oracle": 2 * NOISE / math.sqrt(math.pi),
        "same_path": max(abs(a - b) for a, b in zip(full, free)),
    }


def verify(result):
    rows, oracle = result["rows"], result["oracle"]
    swept = [rows[f] for f in (0.2, 0.4, 0.6, 0.8)]
    table = ", ".join(f"{f:.1f}: {r[0]:.3f} vs {r[1]:.3f}" for f, r in rows.items())
    gain = 1 - rows[0.8][0] / rows[0.8][1]
    return [
        practice.Check(
            "ANSWER: at no fraction in 0.2-0.8; they meet only at 1.0",
            all(r[0] < 0.75 * r[1] for r in swept)
            and abs(rows[1.0][0] - rows[1.0][1]) < 0.1 * rows[1.0][1],
            f"masked residual, inpaint vs unconditional -- {table}. Even one pinned dim (0.8) "
            f"leaves inpainting {gain:.0%} better; the two meet only when nothing is pinned",
        ),
        practice.Check(
            "FINDING: the information never runs out -- the replacement trick does",
            swept[-1][0] > 2 * oracle and swept[0][2] > 0.9 and swept[-1][2] < 0.8,
            f"one pinned dim names the cluster almost surely (centres +-1, noise {NOISE}), so a "
            f"perfect conditional sampler scores {oracle:.3f} at every fraction. The measured "
            f"rise from {swept[0][0]:.3f} to {swept[-1][0]:.3f} is the naive reinjection picking "
            f"the wrong cluster more often -- cluster match {swept[0][2]:.0%} -> {swept[-1][2]:.0%}",
        ),
        practice.Check(
            "CONTROL: at fraction 1.0 inpaint is unconditional generation, bit for bit",
            result["same_path"] == 0.0 and 0.9 < rows[1.0][1] < 1.3,
            f"with every dim masked the lesson's inpaint and sample_unconditional agree to "
            f"{result['same_path']} from one seed, and the unconditional residual "
            f"{rows[1.0][1]:.3f} matches half right-cluster ({oracle:.3f}) and half wrong (~2.0)",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
