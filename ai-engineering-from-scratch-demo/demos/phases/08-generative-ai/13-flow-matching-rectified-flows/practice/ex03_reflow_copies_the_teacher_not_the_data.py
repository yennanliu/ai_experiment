"""Exercise 3 — reflow straightens the teacher, mistakes included.

    **Hard.** Implement one reflow iteration: generate paired (x_0, x_1) by integrating the first model, train a second model on the pairs, and compare 1-step sample quality.

Reading of the exercise: the first model is `main()`'s own (seed 31, 24
hidden units, 6000 steps, lr 0.01, via the lesson's `init_net` and `train`).
1000 pairs are made by drawing `x_1` and integrating that model with the
lesson's own `sample` at 20 steps, which reads its noise from the same seeded
`Random` the pair records. The second model trains on the pairs with the same
loss, network and budget -- a loop around the lesson's `forward`, `backward`
and `apply` that draws a stored pair instead of an independent one. Quality is
the quantile MSE of 1000 samples against the mixture's exact quantiles. As a
ceiling, the same reflow is also run with the *exact* velocity field
`E[x_1 - x_0 | x_t]` as the teacher.

**ANSWER: one reflow cuts the 1-step error from 1.211 to 0.877.** The
teacher's own 20-step samples score 0.663.

**FINDING: the student learns the teacher, not the data.** Its 1-step samples
are **0.266** from the teacher's 20-step distribution and **0.877** from the
data. Given the exact field as teacher instead, the identical student code
reaches **0.179** at one step -- against the exact field's own 1-step error of
4.09. Reflow transfers whatever the teacher integrates to; its ceiling is the
teacher.

**FINDING: the doc's "paths cross" mechanism, made countable.** Under the
independent pairing **52.2%** of pair-lines cross each other (discordant
`(x_1, x_0)` orderings); under the reflow pairing **0.0%** do: an exact 1-D ODE
flow cannot reorder points, and 20 Euler steps did not either. With nothing crossing, the regression target is
a function of `x_t` and the learned field straightens: the mean squared gap
between a sample's 1-step and 20-step landing falls from **2.149** for the first
model to **0.122** for the second.

**CONTROL: the gain comes from the pairing, not the loop.** The same
`train_on_pairs`, seed and budget, given the 1000 noise draws matched to
*independent* data draws, scores **2.465** at one step against reflow's 0.877.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "13-flow-matching-rectified-flows"
N, SPREAD, MODES = 1000, 0.3, (-2.0, 2.0)


def cdf(x):
    return sum(1 + math.erf((x - m) / (SPREAD * math.sqrt(2))) for m in MODES) / 4


def quantile(p, lo=-8.0, hi=8.0):
    for _ in range(50):
        lo, hi = ((lo + hi) / 2, hi) if cdf((lo + hi) / 2) < p else (lo, (lo + hi) / 2)
    return lo


TRUTH = [quantile((i + 0.5) / N) for i in range(N)]


def gap(a, b):
    """Quantile MSE between two sample sets (or a set and TRUTH)."""
    return sum((x - y) ** 2 for x, y in zip(sorted(a), sorted(b))) / len(a)


def exact_velocity(x, t):
    num = den = 0.0
    for mode in MODES:
        mean, var = (1 - t) * mode, (1 - t) ** 2 * SPREAD**2 + t * t
        weight = math.exp(-((x - mean) ** 2) / (2 * var)) / math.sqrt(var)
        num += weight * (-mode + (t - (1 - t) * SPREAD**2) / var * (x - mean))
        den += weight
    return num / den


def exact_sample(steps, rng):
    x, dt = rng.gauss(0, 1), 1.0 / steps
    for i in range(steps):
        x -= dt * exact_velocity(x, 1.0 - i * dt)
    return x


def train_on_pairs(ref, pairs, rng, steps=6000, lr=0.01):
    """The lesson's train(), with (x_1, x_0) drawn from fixed pairs."""
    net = ref.init_net(5, 24, 1, rng)
    for _ in range(steps):
        x1, x0 = pairs[rng.randrange(len(pairs))]
        t = rng.random()
        pred, cache = ref.forward(t * x1 + (1 - t) * x0, t, net)
        ref.apply(net, ref.backward(x1 - x0, pred, cache, net), lr)
    return net


def crossing(pairs):
    """Share of pair-lines that cross: x_1 order and x_0 order disagree."""
    x0 = [b for _, b in sorted(pairs)]
    bad = sum(x0[j] < x0[i] for i in range(len(x0)) for j in range(i + 1, len(x0)))
    return bad / (len(x0) * (len(x0) - 1) / 2)


def draws(fn, offset):
    import random

    return [fn(random.Random(offset + i)) for i in range(N)]


def solve():
    import random

    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = random.Random(31)
    first = ref.init_net(5, 24, 1, rng)
    ref.train(first, 6000, 0.01, rng)
    noise = draws(lambda r: r.gauss(0, 1), 9000)
    pairs = list(zip(noise, draws(lambda r: ref.sample(first, 20, r), 9000)))
    second = train_on_pairs(ref, pairs, random.Random(77))
    independent = list(zip(noise, draws(ref.sample_data, 7000)))
    shuffled = train_on_pairs(ref, independent, random.Random(77))
    oracle = train_on_pairs(
        ref,
        list(zip(noise, draws(lambda r: exact_sample(100, r), 9000))),
        random.Random(77),
    )
    s = {
        name: {
            k: draws(lambda r, n=net, k=k: ref.sample(n, k, r), 50000) for k in (1, 20)
        }
        for name, net in (("first", first), ("second", second), ("oracle", oracle))
    }
    return {
        "one_step": {name: gap(v[1], TRUTH) for name, v in s.items()},
        "teacher": gap(s["first"][20], TRUTH),
        "to_teacher": gap(s["second"][1], s["first"][20]),
        "exact_one": gap(draws(lambda r: exact_sample(1, r), 50000), TRUTH),
        "control": gap(draws(lambda r: ref.sample(shuffled, 1, r), 50000), TRUTH),
        "cross": {"independent": crossing(independent), "reflow": crossing(pairs)},
        "bend": {
            n: sum((a - b) ** 2 for a, b in zip(s[n][1], s[n][20])) / N for n in s
        },
    }


def verify(result):
    one, cross, bend = result["one_step"], result["cross"], result["bend"]
    return [
        practice.Check(
            "ANSWER: one reflow improves 1-step quality",
            one["second"] < one["first"],
            f"1-step quantile MSE against the mixture -- first model {one['first']:.3f}, after "
            f"one reflow {one['second']:.3f}; the teacher's own 20-step samples score "
            f"{result['teacher']:.3f}",
        ),
        practice.Check(
            "FINDING: the student learns the teacher, not the data",
            result["to_teacher"] < one["second"] / 2
            and one["oracle"] < one["second"] / 2,
            f"the reflowed 1-step samples are {result['to_teacher']:.3f} from the teacher's "
            f"20-step distribution and {one['second']:.3f} from the data. With the exact field as "
            f"teacher the identical student code reaches {one['oracle']:.3f} at one step, against "
            f"that field's own 1-step {result['exact_one']:.2f}: reflow's ceiling is the teacher",
        ),
        practice.Check(
            "FINDING: reflow pairs never cross, so the field straightens",
            cross["reflow"] == 0.0
            and 0.4 < cross["independent"] < 0.6
            and bend["second"] < bend["first"] / 10,
            f"crossing pair-lines -- independent pairing {cross['independent']:.1%}, reflow "
            f"pairing {cross['reflow']:.1%} (an exact 1-D ODE flow cannot reorder points, and "
            f"20 Euler steps did not). Mean squared gap between a sample's 1-step and 20-step landing: first model "
            f"{bend['first']:.3f}, second {bend['second']:.3f}",
        ),
        practice.Check(
            "CONTROL: the same loop on independent pairs does not improve",
            result["control"] > one["second"] * 1.2,
            f"train_on_pairs given 1000 noise draws matched to independent data draws, same seed "
            f"and budget, scores {result['control']:.3f} at one step against reflow's "
            f"{one['second']:.3f}: the gain comes from the pairing, not the loop",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
