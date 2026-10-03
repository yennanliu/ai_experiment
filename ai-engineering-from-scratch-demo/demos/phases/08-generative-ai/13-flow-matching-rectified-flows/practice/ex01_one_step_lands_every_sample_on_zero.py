"""Exercise 1 — one step lands every sample on zero, and twenty steps is not the floor.

    **Easy.** Run `code/main.py` and compare 1-step vs 20-step MSE vs the true data distribution.

Reading of the exercise: "MSE vs the true data distribution" is read as the
quantile MSE -- sort the samples and compare them with the exact quantiles of
the lesson's mixture `0.5 N(-2, 0.3^2) + 0.5 N(2, 0.3^2)`, which is the squared
1-D Wasserstein-2 distance and the only MSE that compares two distributions
rather than two paired lists. The network is trained exactly as `main()` does
it (seed 31, 24 hidden units, 6000 steps, lr 0.01) by the lesson's own
`init_net` and `train`, and sampled through the lesson's own `sample`, on 1000
fixed noise seeds. Alongside it runs the *exact* velocity field
`v*(x, t) = E[x_1 - x_0 | x_t = x]`, which is closed-form for a Gaussian
mixture, so the network's error can be separated from the sampler's.

**ANSWER: 1-step 1.249, 20-step 0.521.** Quantile MSE by Euler step count:

| steps | 1 | 2 | 4 | 8 | 20 |
|---|---:|---:|---:|---:|---:|
| lesson network | 1.249 | **0.236** | 0.338 | 0.434 | 0.521 |
| exact field | 4.090 | 0.473 | 0.057 | 0.024 | 0.017 |

**FINDING: one Euler step from the *perfect* field sends every sample to 0.0.**
At `t = 1` both mixture components sit at the same mean, so `v*(x, 1) = x` and `x - 1 * v*(x, 1) = 0` for every noise draw: the largest
|sample| is **4e-16** and the error is `E[x_0^2] = 4.09`. A single step returns
the data *mean*, which lies in neither mode. The network's 1-step row is
"better" than perfect only because it is wrong at `t = 1`.

**FINDING: the lesson's left/right count cannot see that collapse.** At 1 step
the lesson's own statistic reads **514 / 486**, a near-perfect split, while only
**32%** of samples land within three standard deviations of either mode.

**FINDING: more steps converge to the network, not to the data.** The network's
2-step row beats its 20-step row, **0.236** against **0.521**, and its 20-step
error is **31x** the exact field's. Past four steps the integration error is
gone; what remains is the model's.

**CONTROL:** the exact field reaches **0.017** at 20 steps, so the quantile
metric and the closed-form field agree with the data to the precision 1000
quantiles allow.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "13-flow-matching-rectified-flows"
N, STEPS, SPREAD, MODES = 1000, (1, 2, 4, 8, 20), 0.3, (-2.0, 2.0)


def cdf(x):
    """CDF of the lesson's mixture, as sample_data draws it."""
    return (
        sum(0.5 + 0.5 * math.erf((x - m) / (SPREAD * math.sqrt(2))) for m in MODES) / 2
    )


def quantile(p, lo=-8.0, hi=8.0):
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) < p else (lo, mid)
    return (lo + hi) / 2


TRUTH = [quantile((i + 0.5) / N) for i in range(N)]


def quantile_mse(samples):
    return sum((a - b) ** 2 for a, b in zip(sorted(samples), TRUTH)) / N


def exact_velocity(x, t):
    """E[x_1 - x_0 | x_t = x] for x_t = t x_1 + (1 - t) x_0, per component then mixed."""
    num = den = 0.0
    for mode in MODES:
        mean, var = (1 - t) * mode, (1 - t) ** 2 * SPREAD**2 + t * t
        weight = math.exp(-((x - mean) ** 2) / (2 * var)) / math.sqrt(var)
        num += weight * (-mode + (t - (1 - t) * SPREAD**2) / var * (x - mean))
        den += weight
    return num / den


def exact_sample(steps, rng):
    """The lesson's sample() loop, with the exact field in place of the network."""
    x, dt = rng.gauss(0, 1), 1.0 / steps
    for i in range(steps):
        x -= dt * exact_velocity(x, 1.0 - i * dt)
    return x


def sweep(draw, random):
    """{steps: samples} over the same 1000 noise seeds for every step count."""
    return {k: [draw(k, random.Random(1000 + i)) for i in range(N)] for k in STEPS}


def collapse(one):
    """main()'s own left/right count, and the share actually inside a mode."""
    left = sum(x <= 0 for x in one)
    return (left, N - left), sum(abs(abs(x) - 2.0) < 3 * SPREAD for x in one) / N


def solve():
    import random

    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = random.Random(31)
    net = ref.init_net(in_dim=5, hidden=24, out_dim=1, rng=rng)
    ref.train(net, steps=6000, lr=0.01, rng=rng)
    nets = sweep(lambda k, r: ref.sample(net, k, r), random)
    exact = sweep(exact_sample, random)
    left_right, in_mode = collapse(nets[1])
    return {
        "net": {k: quantile_mse(s) for k, s in nets.items()},
        "exact": {k: quantile_mse(s) for k, s in exact.items()},
        "exact_one_max": max(abs(x) for x in exact[1]),
        "left_right": left_right,
        "in_mode": in_mode,
        "second_moment": sum(t * t for t in TRUTH) / N,
    }


def verify(result):
    net, exact = result["net"], result["exact"]
    left, right = result["left_right"]
    ratio = net[20] / exact[20]
    return [
        practice.Check(
            "ANSWER: 1-step and 20-step quantile MSE against the true mixture",
            net[1] > net[20],
            "lesson network by steps -- "
            + ", ".join(f"{k}: {v:.3f}" for k, v in net.items())
            + "; exact field -- "
            + ", ".join(f"{k}: {v:.3f}" for k, v in exact.items()),
        ),
        practice.Check(
            "FINDING: one step from the perfect field lands every sample on 0.0",
            result["exact_one_max"] < 1e-12
            and abs(exact[1] - result["second_moment"]) < 1e-9,
            f"at t=1 both components share mean 0, so v*(x, 1) = x and one Euler step returns "
            f"x - x = 0: the largest |sample| is {result['exact_one_max']:.1g} and the error is "
            f"E[x0^2] = {exact[1]:.3f}. One step returns the data mean, which lies in neither "
            f"mode; the network's {net[1]:.3f} is better than perfect only because it is wrong",
        ),
        practice.Check(
            "FINDING: the lesson's left/right count cannot see the collapse",
            abs(left - right) < 0.1 * N and result["in_mode"] < 0.5,
            f"at 1 step main()'s statistic reads left {left} / right {right}, a near-even "
            f"split, while only {result['in_mode']:.0%} of samples land within three standard "
            "deviations of either mode",
        ),
        practice.Check(
            "FINDING: more steps converge to the network, not to the data",
            net[2] < net[20] and ratio > 10,
            f"the network's 2-step error {net[2]:.3f} beats its 20-step {net[20]:.3f}, and its "
            f"20-step error is {ratio:.0f}x the exact field's {exact[20]:.3f}: past four steps "
            "the integration error is gone and what remains is the model's",
        ),
        practice.Check(
            "CONTROL: the exact field reaches the data at 20 steps",
            exact[20] < 0.03 and exact[4] < 0.1,
            f"exact field at 4 steps {exact[4]:.3f}, at 20 steps {exact[20]:.3f}, so the metric "
            "and the closed-form field agree with the mixture to what 1000 quantiles resolve",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
