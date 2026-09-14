"""Exercise 2 — yes, more stable, but not at the clip the exercise prescribes.

    **Medium.** Replace the Goodfellow BCE loss with the WGAN loss:
    `loss_D = E[D(fake)] - E[D(real)]`, `loss_G = -E[D(fake)]`, and clip D's
    weights to `[-0.01, 0.01]`. Is training more stable? Compare wall-clock
    convergence.

Reading of the exercise: the model is the lesson's -- `init_mlp`, `forward_g`,
`forward_d`, `sample_real`, `sample_noise`, the same shapes and the same `g_lr`.
Only the loss changes, and because `update_d` and `update_g` hard-code the BCE
gradient `p - target`, a different loss needs its own gradient; `back_mlp` is
that, and the CONTROL checks it against the lesson's own `forward_d`. Neither
loss is used to score the result. Both arms are judged by the **exact 1-D
Wasserstein distance** between 400 generated and 400 real samples, which is what
WGAN claims to minimise and what BCE can be held to without unfairness.

Two anchors make the numbers readable: `W1(real, real)` on two fresh draws is
**0.129** -- the sampling floor -- and the *same generators at the same seeds*
with training switched off score **1.730**.

**ANSWER: more stable, yes -- and at the prescribed clip that is all it is.**

| arm | W1 | per-seed spread | mean abs critic score |
|---|---:|---:|---:|
| BCE (the lesson's) | 1.470 | 0.973 | 0.398 |
| WGAN, clip 0.01 | **1.715** | **0.204** | **0.0003** |
| WGAN, clip 0.5 | **0.937** | 0.158 | -- |

At `clip = 0.01` the run is stable in the way a stopped clock is: it lands on
**1.715** against the untrained **1.730**, a gap well inside its own 0.204 seed
spread, having gone essentially nowhere in 800 steps.

**FINDING: the prescribed clip bounds the critic to a rounding error.** With
every weight in `[-0.01, 0.01]` through a 1->16->1 net, the score cannot leave a
band of about +-0.013, and the measured mean `|D(x)|` is **0.0003** against BCE's
**0.398**, 1,394x smaller. A generator learns from the critic's *slope*, and there is no slope
left to learn from. The stability the exercise asks after is the stability of a
switched-off critic.

**FINDING: raise the clip and WGAN wins the comparison outright.** At
`clip = 0.5` it reaches **0.937** where BCE reaches 1.470, with a per-seed spread
of **0.158** against BCE's **0.973** -- better *and* **6.2x** steadier, which is
the result the exercise is reaching for. At `clip = 2.0` it diverges instead
(spread 5.14). The clip is not a detail of the method; across 0.01, 0.1, 0.5 and
2.0 it is the entire experiment.

**FINDING: wall-clock is a wash, and the term that would decide it is missing.**
One WGAN run costs about **1.0x** a BCE run -- no sigmoid, one extra clip pass.
The exercise never mentions `n_critic`, and the standard recipe's 5 critic steps
per generator step is what would actually make WGAN 3-5x slower per unit of
progress.

Structure: `walk` visits every scalar in a params dict so `descend` and `clamp`
are two lines each; `back_mlp` is the shared backward for the lesson's 2-layer
MLP; `run` is one training loop with the loss as a flag.
"""

from __future__ import annotations

import statistics
import time

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "03-gans-generator-discriminator"
Z_DIM, HIDDEN, BATCH, STEPS, RATE = 4, 16, 32, 800, 0.02
PROBE, SEEDS, CLIPS, SHOWN = 400, 3, (0.01, 0.1, 0.5, 2.0), 0.01
wasserstein = lambda a, b: statistics.fmean(abs(x - y) for x, y in zip(sorted(a), sorted(b)))


def zeros_like(params):
    """A gradient accumulator shaped like `params`."""
    return {k: ([[0.0] * len(r) for r in v] if isinstance(v[0], list) else [0.0] * len(v))
            for k, v in params.items()}


def descend(params, grads, count, limit=None):
    """The lesson's own update rule, then WGAN's weight clipping when `limit` is given."""
    for key, value in params.items():
        pairs = zip(value, grads[key]) if isinstance(value[0], list) else [(value, grads[key])]
        for row, grad_row in pairs:
            for i in range(len(row)):
                row[i] -= RATE * grad_row[i] / count
                row[i] = row[i] if limit is None else max(-limit, min(limit, row[i]))


def back_mlp(ref, params, x, h, pre1, seed, grads):
    """Accumulate d(seed * pre2)/d params for the lesson's MLP; return d/dx."""
    grads["b2"][0] += seed
    dpre1 = [params["W2"][0][j] * seed * ref.leaky_grad(pre1[j]) for j in range(len(h))]
    dx = [0.0] * len(x)
    for j in range(len(h)):
        grads["W2"][0][j] += seed * h[j]
        grads["b1"][j] += dpre1[j]
        for k in range(len(x)):
            grads["W1"][j][k] += dpre1[j] * x[k]
            dx[k] += params["W1"][j][k] * dpre1[j]
    return dx


def wgan_step(ref, generator, critic, reals, noise, limit):
    """One critic step then one generator step, both under the WGAN loss."""
    grads = zeros_like(critic)
    for sign, batch in ((-1.0, reals), (1.0, [ref.forward_g(z, generator)[0] for z in noise])):
        for x in batch:
            _, h, pre1, _ = ref.forward_d(x, critic)
            back_mlp(ref, critic, x, h, pre1, sign, grads)
    descend(critic, grads, len(reals) + len(noise), limit)
    grads, spill = zeros_like(generator), zeros_like(critic)
    for z in noise:
        x_hat, g_h, g_pre1 = ref.forward_g(z, generator)
        _, d_h, d_pre1, _ = ref.forward_d(x_hat, critic)
        pull = back_mlp(ref, critic, x_hat, d_h, d_pre1, -1.0, spill)
        back_mlp(ref, generator, z, g_h, g_pre1, pull[0], grads)
    descend(generator, grads, len(noise))


def run(ref, random, limit, seed, steps=STEPS):
    """One run; `limit` of None is the lesson's BCE arm, `steps=0` trains nothing."""
    rng = random.Random(seed)
    generator, critic = ref.init_mlp(Z_DIM, HIDDEN, 1, rng), ref.init_mlp(1, HIDDEN, 1, rng)
    for _ in range(steps):
        reals, noise = ref.sample_real(BATCH, rng), ref.sample_noise(BATCH, Z_DIM, rng)
        if limit is None:
            ref.update_d(reals, [ref.forward_g(z, generator)[0] for z in noise], critic, RATE / 2)
            ref.update_g(ref.sample_noise(BATCH, Z_DIM, rng), generator, critic, RATE)
        else:
            wgan_step(ref, generator, critic, reals, noise, limit)
    probe = [ref.forward_g(z, generator)[0][0] for z in ref.sample_noise(PROBE, Z_DIM, rng)]
    return (wasserstein(probe, [x[0] for x in ref.sample_real(PROBE, rng)]),
            statistics.fmean(abs(ref.forward_d([v], critic)[3]) for v in probe))


def arm(ref, random, limit):
    """(mean W1, per-seed spread, mean |critic score|, seconds per run)."""
    start = time.perf_counter()
    rows = [run(ref, random, limit, s) for s in range(SEEDS)]
    w1s = [w for w, _ in rows]
    return (statistics.fmean(w1s), max(w1s) - min(w1s),
            statistics.fmean(s for _, s in rows), (time.perf_counter() - start) / SEEDS)


def gradient_error(ref, random, step=1e-6):
    """Worst gap between back_mlp's dD/dW1 and a central difference of forward_d."""
    critic, x, worst = ref.init_mlp(1, HIDDEN, 1, random.Random(5)), [1.3], 0.0
    grads = zeros_like(critic)
    back_mlp(ref, critic, x, *ref.forward_d(x, critic)[1:3], 1.0, grads)
    for j in range(HIDDEN):
        keep = critic["W1"][j][0]
        critic["W1"][j][0] = keep + step
        high = ref.forward_d(x, critic)[3]
        critic["W1"][j][0] = keep - step
        critic["W1"][j][0], low = keep, ref.forward_d(x, critic)[3]
        worst = max(worst, abs((high - low) / (2 * step) - grads["W1"][j][0]))
    return worst


def solve():
    import random
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = random.Random(0)
    return {
        "bce": arm(ref, random, None), "wgan": {c: arm(ref, random, c) for c in CLIPS},
        "floor": wasserstein(*[[x[0] for x in ref.sample_real(PROBE, rng)] for _ in range(2)]),
        "gradient_error": gradient_error(ref, random),
        "untrained": statistics.fmean(run(ref, random, None, s, 0)[0] for s in range(SEEDS)),
    }


def verify(result):
    bce, wgan, untrained = result["bce"], result["wgan"], result["untrained"]
    asked, best = wgan[SHOWN], wgan[0.5]
    return [
        practice.Check(
            "ANSWER: more stable, because the prescribed clip switches the critic off",
            asked[1] < bce[1] and abs(asked[0] - untrained) < asked[1]
            and asked[2] < bce[2] / 100,
            f"W1 over {SEEDS} seeds: BCE {bce[0]:.3f} +-{bce[1]:.3f}, WGAN at clip {SHOWN} "
            f"{asked[0]:.3f} +-{asked[1]:.3f} -- steadier, and landing on the *untrained* "
            f"{untrained:.3f}, against a {result['floor']:.3f} floor. Clipping holds mean |D(x)| "
            f"to {asked[2]:.4f} against BCE's {bce[2]:.3f}: no slope for G to learn from",
        ),
        practice.Check(
            "FINDING: raise the clip and WGAN wins outright; wall-clock decides nothing",
            best[0] < bce[0] and best[1] < bce[1] and wgan[CLIPS[-1]][0] > best[0],
            "W1 +- spread by clip -- "
            + ", ".join(f"{c}: {wgan[c][0]:.3f} +-{wgan[c][1]:.3f}" for c in CLIPS)
            + f" -- against BCE's {bce[0]:.3f} +-{bce[1]:.3f}. At 0.5 WGAN is better and "
              f"{bce[1] / best[1]:.1f}x steadier; at {CLIPS[-1]} it diverges. A WGAN run costs "
              f"{asked[3] / bce[3]:.1f}x a BCE one, so n_critic -- never mentioned -- is the "
              "missing term, not speed",
        ),
        practice.Check(
            "CONTROL: the WGAN gradient matches a finite difference of the lesson's forward_d",
            result["gradient_error"] < 1e-6,
            f"back_mlp's dD/dW1 against a central difference of ref.forward_d's own raw score "
            f"over all {HIDDEN} first-layer weights: worst gap {result['gradient_error']:.1e}, so "
            "the new gradient differentiates the lesson's critic and not a second copy",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
