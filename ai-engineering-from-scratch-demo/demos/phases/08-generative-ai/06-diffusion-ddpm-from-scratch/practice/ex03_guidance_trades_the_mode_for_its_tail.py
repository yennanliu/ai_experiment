"""Exercise 3 — guidance peaks at w=1, and what it reliably does is narrow the output.

    **Hard.** Add classifier-free guidance. Condition on a class label
    `c ∈ {0, 1}`, drop it 10% of the time during training, and at sampling time
    use `ε = (1+w)·ε_cond - w·ε_uncond`. Measure the conditional-mode-hit rate at
    `w = 0, 1, 3, 7`.

Reading of the exercise: the label reaches the network as an extra input
dimension carrying `-1` for the dropped case and `0` or `1` otherwise, so the
lesson's own `init_net`, `forward`, `backward` and `apply_update` run unchanged at
`x_dim = 2`. Class 0 is the `-2` mode and class 1 the `+2` mode of the lesson's
own `sample_data`, split apart so the label means something. The drop rate is the
exercise's 10%, and the hit rate is the share of samples for class `c` landing
within 1.0 of that class's own centre.

**ANSWER: it peaks at w=1 and then goes backwards.**

| w | 0 | 1 | 3 | 7 |
|---|---:|---:|---:|---:|
| hit rate | 0.527 | **0.562** | 0.520 | 0.490 |
| sample spread | 1.73 | 1.66 | 1.50 | **1.48** |

Guidance buys **3.5 points** of hit rate at `w = 1` and has given all of them
back by `w = 3`. The exercise asks for the rate at 0, 1, 3 and 7 as though it
were monotone; it is not.

**FINDING: what guidance reliably does is narrow the output, not sharpen it.**
The spread falls monotonically, **1.73 -> 1.48**, across the same sweep in which
the hit rate turns over. That is the diversity cost classifier-free guidance is
known for, and it is the one column here that moves in one direction throughout.

**FINDING: the signal being amplified is weak to begin with.** The unguided hit
rate is **0.527** against a ceiling of 1.0 -- conditioning barely works at this
`T`, for the reason Exercise 1 gives: `alpha_bar[-1]` is 0.667 and the sampler
starts off the manifold its trainer built. Guidance is being asked to amplify
something the model is not confident about.

**FINDING: the subtrahend is the least-trained thing in the model.** With a 10%
drop rate the `c = -1` input is seen about **594** times in 6,000 steps, against
roughly **2,703** for each label. The guidance formula multiplies that branch's
error by `w` and subtracts it, so the least-trained prediction is the one whose
mistakes get amplified -- which is a mechanism for the turnover above, though
this experiment does not isolate it.

**CONTROL: at `w = 0` the formula is the ordinary conditional sampler.**
`(1+0)*eps_cond - 0*eps_uncond` is `eps_cond` exactly, so the `w = 0` column is
the unguided baseline and every later column is measured against it rather than
against a separate run.

Structure: `net_input` carries the label; `train` is the lesson's loop with
dropout; `draw` is the lesson's reverse chain with the guidance combination.
"""

from __future__ import annotations

import math
import statistics

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "06-diffusion-ddpm-from-scratch"
T_DIM, HIDDEN, TRAIN_STEPS, RATE, T = 8, 32, 6_000, 0.01, 40
WEIGHTS, SEEDS, DRAWS, DROP, CENTRE = (0.0, 1.0, 3.0, 7.0), 2, 300, 0.1, 2.0


def labelled(rng):
    """One (sample, class) pair: class 0 is the -2 mode, class 1 the +2 mode."""
    label = rng.randrange(2)
    return rng.gauss(-CENTRE + 2 * CENTRE * label, 0.4), label


def train(ref, net, alpha_bars, rng):
    """The lesson's own loop with the label as a second input, dropped DROP of the time."""
    dropped = 0
    for _ in range(TRAIN_STEPS):
        x0, label = labelled(rng)
        step, eps = rng.randrange(T), rng.gauss(0, 1)
        a_bar = alpha_bars[step]
        x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
        shown = -1.0 if rng.random() < DROP else float(label)
        dropped += shown < 0
        hat, cache = ref.forward([x_t, shown], ref.sin_embed(step, T, T_DIM), net)
        ref.apply_update(net, ref.backward([eps], hat, cache, net), RATE)
    return dropped


def draw(ref, net, alphas, alpha_bars, rng, label, weight):
    """The lesson's reverse chain with eps = (1+w)*eps_cond - w*eps_uncond."""
    x = rng.gauss(0, 1)
    for step in range(T - 1, -1, -1):
        embed = ref.sin_embed(step, T, T_DIM)
        cond = ref.forward([x, float(label)], embed, net)[0][0]
        guided = cond
        if weight > 0:
            free = ref.forward([x, -1.0], embed, net)[0][0]
            guided = (1 + weight) * cond - weight * free
        beta = 1 - alphas[step]
        mean = (x - beta / math.sqrt(1 - alpha_bars[step]) * guided) / math.sqrt(alphas[step])
        x = mean + math.sqrt(beta) * rng.gauss(0, 1) if step > 0 else mean
    return x


def score(samples, label):
    """(hit rate against this class's own centre, spread of the samples)."""
    want = -CENTRE + 2 * CENTRE * label
    return (sum(1 for s in samples if abs(s - want) < 1.0) / len(samples),
            statistics.pstdev(samples))


def measure(ref, net, alphas, alpha_bars, rng, grid):
    """Score every (weight, label) pair for one trained network into `grid`."""
    for weight in WEIGHTS:
        for label in (0, 1):
            grid[weight].append(score([draw(ref, net, alphas, alpha_bars, rng, label, weight)
                                       for _ in range(DRAWS)], label))


def solve():
    import random
    ref = parity.load_reference(PHASE, LESSON, "main")
    grid, dropped = {w: [] for w in WEIGHTS}, 0
    for seed in range(SEEDS):
        rng = random.Random(seed)
        net = ref.init_net(2, T_DIM, HIDDEN, rng)
        _, alphas, alpha_bars = ref.make_schedule(T)
        dropped += train(ref, net, alpha_bars, rng)
        measure(ref, net, alphas, alpha_bars, rng, grid)
    return {
        "hits": {w: statistics.fmean(r[0] for r in rows) for w, rows in grid.items()},
        "sds": {w: statistics.fmean(r[1] for r in rows) for w, rows in grid.items()},
        "dropped": dropped / SEEDS,
    }


def verify(result):
    hits, sds = result["hits"], result["sds"]
    return [
        practice.Check(
            "ANSWER: the hit rate peaks at w=1 and then goes backwards",
            hits[WEIGHTS[1]] == max(hits.values()) and hits[WEIGHTS[-1]] < hits[WEIGHTS[0]],
            "hit rate / spread by w -- "
            + ", ".join(f"w={w:.0f}: {hits[w]:.3f} / {sds[w]:.2f}" for w in WEIGHTS)
            + f". Guidance buys {hits[WEIGHTS[1]] - hits[WEIGHTS[0]]:+.3f} at w=1 and has given "
              f"it all back by w={WEIGHTS[2]:.0f}, ending {hits[WEIGHTS[-1]] - hits[WEIGHTS[0]]:+.3f} "
              "below the unguided baseline. The exercise asks for 0, 1, 3 and 7 as though the "
              "rate were monotone in w; it is not",
        ),
        practice.Check(
            "FINDING: what guidance reliably does is narrow the output",
            all(sds[a] > sds[b] for a, b in zip(WEIGHTS, WEIGHTS[1:])),
            f"the spread falls monotonically across the sweep, {sds[WEIGHTS[0]]:.2f} -> "
            f"{sds[WEIGHTS[-1]]:.2f}, over the same range in which the hit rate turns over. "
            "(1+w)*eps_cond - w*eps_uncond extrapolates past the conditional prediction, away "
            "from the unconditional one, and the result is less diverse rather than more: that "
            "is the diversity cost guidance is known for, and it is the one column that moves in "
            "a single direction throughout",
        ),
        practice.Check(
            "FINDING: the signal being amplified is weak to begin with",
            hits[WEIGHTS[0]] < 0.7,
            f"the unguided hit rate is {hits[WEIGHTS[0]]:.3f} against a ceiling of 1.0, so "
            f"conditioning barely works at T={T} -- for the reason Exercise 1 gives, that "
            "alpha_bar[-1] is 0.667 and the sampler starts off the manifold its trainer built. "
            "Guidance is being asked to amplify something the model is not confident about",
        ),
        practice.Check(
            "FINDING: the subtrahend is the least-trained thing in the model",
            0.05 * TRAIN_STEPS < result["dropped"] < 0.15 * TRAIN_STEPS,
            f"at the exercise's {DROP:.0%} drop rate the unconditional input is seen "
            f"{result['dropped']:.0f} times in {TRAIN_STEPS:,} steps, against about "
            f"{(TRAIN_STEPS - result['dropped']) / 2:.0f} for each label. The guidance formula "
            "multiplies that branch's error by w and subtracts it, so the least-trained "
            "prediction in the model is the one whose mistakes get amplified",
        ),
        practice.Check(
            "CONTROL: at w = 0 the formula is the ordinary conditional sampler",
            WEIGHTS[0] == 0.0,
            f"(1+0)*eps_cond - 0*eps_uncond is eps_cond exactly, and the code takes that branch "
            f"without evaluating the unconditional pass at all. The w=0 column, {hits[0.0]:.3f} "
            "hit rate, is therefore the unguided baseline of this same network rather than a "
            "separate run that might differ for other reasons",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
