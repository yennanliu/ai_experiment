"""Exercise 2 — logit-normal starves the ends of the schedule and buys nothing in the middle.

    **Medium.** Switch from uniform `t` sampling to logit-normal (concentrates sampling at mid-t). Does the model quality improve?

Reading of the exercise: the lesson's `train` hard-codes `t = rng.random()`, so
the switch needs a training loop with the `t` draw as a parameter. It is
written around the lesson's own `sample_data`, `forward`, `backward` and
`apply`, keeping `train`'s draw order, and with the uniform draw it reproduces
`train` bit for bit. Logit-normal is SD3's `t = sigmoid(z), z ~ N(0, 1)`.
"Model quality" is measured two ways: the network's mean squared error against
the exact velocity field `v*(x, t) = E[x_1 - x_0 | x_t = x]` (closed-form for
the lesson's Gaussian mixture), split into the end bands `t <= 0.1 or >= 0.9`
and the middle `0.3..0.7`; and the 20-step sample's quantile MSE against the
true mixture. One seed is not enough to answer, so six are used for each sampler.

**ANSWER: no -- it is worse.** Means over seeds 31 and 1-5:

| | velocity error, all t | ends | middle | 20-step sample |
|---|---:|---:|---:|---:|
| uniform | 0.508 | 0.613 | 0.446 | 1.150 |
| logit-normal | 0.813 | 1.393 | 0.666 | 1.259 |

**FINDING: the ends are starved, by a factor of seven.** A logit-normal draw
lands in the end bands with probability **2.8%** against uniform's **20%**, and
the end-band error rises **2.3x**. Every Euler sample begins with a query at
`t = 1` -- the 1-step sampler queries nothing else -- and logit-normal's density
there is zero.

**FINDING: concentrating draws in the middle does not make the middle better.**
The middle `0.3..0.7` receives **1.5x** as many draws, yet its error rises
from **0.446** to **0.666**. Sample count is not what limits accuracy there:
with batch size one and a constant learning rate, the extra draws are taken
away from the ends without lowering the middle's error.

**FINDING: the seed decides more than the sampler.** Across the six uniform
seeds the velocity error ranges **0.276 to 0.709** -- wider than the **0.306**
gap between the two samplers' means -- and the best logit-normal seed
(**0.364**) beats the worst uniform one, so a single run could have returned
either verdict.

**CONTROL:** with the uniform draw the loop here and the lesson's own `train`
produce weights that differ by **0.0** after 500 steps.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "13-flow-matching-rectified-flows"
SEEDS, SPREAD, MODES, N = (31, 1, 2, 3, 4, 5), 0.3, (-2.0, 2.0), 600
ENDS, MIDDLE = (0.02, 0.05, 0.1, 0.9, 0.95, 0.98), (0.3, 0.4, 0.5, 0.6, 0.7)
METRICS = ("all", "ends", "middle", "sample")
ALL_T = tuple((i + 0.5) / 20 for i in range(20))


def uniform(rng):
    return rng.random()


def logit_normal(rng):
    return 1.0 / (1.0 + math.exp(-rng.gauss(0, 1)))


def train(ref, net, steps, rng, draw_t, lr=0.01):
    """The lesson's train(), with the t draw as a parameter and nothing else changed."""
    for _ in range(steps):
        x0 = ref.sample_data(rng)
        x1 = rng.gauss(0, 1)
        t = draw_t(rng)
        pred, cache = ref.forward(t * x1 + (1 - t) * x0, t, net)
        ref.apply(net, ref.backward(x1 - x0, pred, cache, net), lr)


def exact_velocity(x, t):
    """E[x_1 - x_0 | x_t = x] for the lesson's two-mode mixture."""
    num = den = 0.0
    for mode in MODES:
        mean, var = (1 - t) * mode, (1 - t) ** 2 * SPREAD**2 + t * t
        weight = math.exp(-((x - mean) ** 2) / (2 * var)) / math.sqrt(var)
        num += weight * (-mode + (t - (1 - t) * SPREAD**2) / var * (x - mean))
        den += weight
    return num / den


def velocity_error(ref, net, times, rng):
    errs = []
    for t in times:
        for _ in range(60):
            x = t * rng.gauss(0, 1) + (1 - t) * ref.sample_data(rng)
            errs.append((ref.forward(x, t, net)[0] - exact_velocity(x, t)) ** 2)
    return sum(errs) / len(errs)


def quantile(p, lo=-8.0, hi=8.0):
    """Bisection on the mixture's CDF."""
    for _ in range(50):
        mid = (lo + hi) / 2
        cdf = sum(1 + math.erf((mid - m) / (SPREAD * math.sqrt(2))) for m in MODES) / 4
        lo, hi = (mid, hi) if cdf < p else (lo, mid)
    return lo


TRUTH = [quantile((i + 0.5) / N) for i in range(N)]


def score(ref, random, draw_t, seed):
    """Velocity error by t band, and 20-step quantile MSE, for one trained network."""
    rng = random.Random(seed)
    net = ref.init_net(5, 24, 1, rng)
    train(ref, net, 6000, rng, draw_t)
    bands = {"all": ALL_T, "ends": ENDS, "middle": MIDDLE}
    row = {k: velocity_error(ref, net, ts, random.Random(5)) for k, ts in bands.items()}
    got = sorted(ref.sample(net, 20, random.Random(1000 + i)) for i in range(N))
    return {**row, "sample": sum((a - b) ** 2 for a, b in zip(got, TRUTH)) / N}


def summarise(draw_t, rows, rng, n=20000):
    """Means over seeds, the seed range, and where the t draws land."""
    ts = [draw_t(rng) for _ in range(n)]
    alls = [r["all"] for r in rows]
    means = {m: sum(r[m] for r in rows) / len(rows) for m in METRICS}
    return {
        **means,
        "lo": min(alls),
        "hi": max(alls),
        "end_share": sum(not 0.1 < t < 0.9 for t in ts) / n,
        "mid_share": sum(0.3 <= t <= 0.7 for t in ts) / n,
    }


def solve():
    import random

    ref = parity.load_reference(PHASE, LESSON, "main")
    a, b = random.Random(31), random.Random(31)
    mine, theirs = ref.init_net(5, 24, 1, a), ref.init_net(5, 24, 1, b)
    train(ref, mine, 500, a, uniform)
    ref.train(theirs, 500, 0.01, b)
    diff = parity.assert_close(list(mine.values()), list(theirs.values()), 0)
    out = {"parity": diff.worst}
    for f in (uniform, logit_normal):
        rows = [score(ref, random, f, seed) for seed in SEEDS]
        out[f.__name__] = summarise(f, rows, random.Random(0))
    return out


def verify(result):
    uni, ln = result["uniform"], result["logit_normal"]
    return [
        practice.Check(
            "ANSWER: no -- logit-normal is worse on every measure",
            all(ln[k] > uni[k] for k in METRICS),
            "six-seed means, uniform vs logit-normal -- "
            + ", ".join(f"{k}: {uni[k]:.3f} vs {ln[k]:.3f}" for k in METRICS)
            + " (velocity error vs the exact field by t band; 20-step quantile MSE)",
        ),
        practice.Check(
            "FINDING: the ends of the schedule are starved",
            ln["end_share"] < uni["end_share"] / 5 and ln["ends"] > 1.5 * uni["ends"],
            f"{ln['end_share']:.1%} of logit-normal draws land at t<0.1 or t>0.9, against "
            f"{uni['end_share']:.0%}, and end error rises {ln['ends'] / uni['ends']:.1f}x; "
            "every Euler sample's first query is at t=1, where that density is zero",
        ),
        practice.Check(
            "FINDING: more draws in the middle do not make the middle better",
            ln["mid_share"] > 1.4 * uni["mid_share"] and ln["middle"] >= uni["middle"],
            f"t in 0.3..0.7 gets {ln['mid_share'] / uni['mid_share']:.1f}x the draws, yet "
            f"its error moves {uni['middle']:.3f} -> {ln['middle']:.3f}",
        ),
        practice.Check(
            "FINDING: the seed decides more than the sampler",
            uni["hi"] - uni["lo"] > ln["all"] - uni["all"] and ln["lo"] < uni["hi"],
            f"uniform seeds span {uni['lo']:.3f} to {uni['hi']:.3f}, wider than the "
            f"{ln['all'] - uni['all']:.3f} gap in means; the best logit-normal seed "
            f"({ln['lo']:.3f}) beats the worst uniform one, so one run could say either",
        ),
        practice.Check(
            "CONTROL: the loop here is the lesson's train() when the draw is uniform",
            result["parity"] == 0.0,
            f"after 500 steps from seed 31 the weights differ by {result['parity']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
