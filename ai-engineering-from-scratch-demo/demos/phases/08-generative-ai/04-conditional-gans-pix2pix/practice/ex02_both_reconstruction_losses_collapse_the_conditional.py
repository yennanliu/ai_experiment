"""Exercise 2 — yes, and in the wrong direction: both losses collapse the conditional.

    **Medium.** Replace L1 with a perceptual-style loss in the 1-D setting (e.g.
    a small frozen D acting as feature extractor). Does it change sharpness of
    the conditional distribution?

Reading of the exercise: "sharpness" is read as the spread of `G(z, c)` over 400
prior draws at a fixed class, against the **0.3** the real data has by
construction. The perceptual arm freezes a copy of D after a warm-up and uses its
first-layer activations as features, scoring `||h(x_hat) - h(target)||^2`; the
adversarial half stays the lesson's `update_d` and the lesson's own gradient
algebra. Both reconstruction terms aim at the same per-class target, the mode
centre, because that is the only "paired" target a 1-D conditional has.

**ANSWER: it changes sharpness sharply, and every arm overshoots.**

| arm | mean per-class sd | against real 0.30 |
|---|---:|---|
| the lesson as shipped | 0.146 | already 2.1x too narrow |
| perceptual, weight 0.5 | 0.046 | 6.5x too sharp |
| perceptual, weight 2.0 | 0.025 | 12x too sharp |
| L1, weight 1.0 | **0.023** | **13x too sharp** |

Perceptual at weight 2.0 and L1 at weight 1.0 land **0.002** apart, which is the
next finding stated as a number.

**FINDING: there is no L1 in use to replace.** `update_g` takes `l1_w=0.0` and
`targets=None`, and `main()` calls it with neither. The branch that would apply
the L1 gradient never executes in the lesson as shipped, so the exercise's
"replace L1" has to begin by switching on a term the lesson never runs.

**FINDING: in 1-D the two losses are the same family, and only the weight
differs.** Both pull `x_hat` toward one point per class, so both must collapse the
conditional; the four rows above sit on a single monotone curve from 0.146 down
toward zero. A perceptual loss can only behave differently from L1 where the
feature map folds two inputs together, and D's first layer over a scalar does not
-- and it cannot: leaky-ReLU of an affine map of a scalar is monotone in that
scalar, unit by unit, so this feature distance is provably a reweighted absolute
distance with its minimum at the target. That is a fact about the architecture,
stated rather than measured; what is measured is the curve the four arms lie on.

**FINDING: the GAN alone already under-disperses.** Before any reconstruction
term the spread is **0.146** against the real **0.30** -- already 2.1x too
narrow -- so every arm moves a quantity that was on the wrong side to begin
with.

**CONTROL: the frozen extractor is frozen.** Its weights are bit-identical before
and after the run that uses it.

Structure: `pull_l1` and `pull_perceptual` are the two reconstruction gradients;
`generator_step` is the lesson's own algebra with one of them added; `train` runs
an arm.
"""

from __future__ import annotations

import copy
import statistics

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "04-conditional-gans-pix2pix"
Z_DIM, HIDDEN, BATCH, G_RATE, D_RATE = 4, 16, 32, 0.02, 0.01
STEPS, WARMUP, SEEDS, PROBE, REAL_SD = 3_000, 1_000, 2, 400, 0.3
CLASSES, CENTRES = 2, (-2.0, 2.0)
noise_batch = lambda rng, n=BATCH: [[rng.gauss(0, 1) for _ in range(Z_DIM)] for _ in range(n)]
# d|x_hat - target|/d x_hat: the gradient update_g's dead l1_w branch would have added
pull_l1 = lambda ref, frozen, x_hat, target, label: [1.0 if a > b else -1.0
                                                     for a, b in zip(x_hat, target)]


def pull_perceptual(ref, frozen, x_hat, target, label):
    """d ||h(x_hat) - h(target)||^2 / d x_hat through a frozen D's first layer."""
    _, live, pre1, _, _ = ref.d_forward(x_hat, label, frozen, CLASSES)
    _, fixed, _, _, _ = ref.d_forward(target, label, frozen, CLASSES)
    return [2 * sum((live[j] - fixed[j]) * frozen["W1"][j][k] * ref.leaky_grad(pre1[j])
                    for j in range(HIDDEN)) for k in range(len(x_hat))]


def adversarial_pull(ref, critic, x_hat, label):
    """d(-log D)/d x_hat through the lesson's critic -- update_g's own algebra."""
    prob, _, d_pre1, _, _ = ref.d_forward(x_hat, label, critic, CLASSES)
    seeded = [critic["W2"][0][j] * (prob - 1.0) * ref.leaky_grad(d_pre1[j]) for j in range(HIDDEN)]
    return [sum(critic["W1"][j][k] * seeded[j] for j in range(HIDDEN)) for k in range(len(x_hat))]


def accumulate(ref, generator, z, out, dx, grads):
    """Backprop dx through the lesson's generator into `grads`."""
    x_hat, g_h, g_pre1, g_inp = out
    for k in range(len(x_hat)):
        grads["b2"][k] += dx[k]
        for b in range(len(g_h)):
            grads["W2"][k][b] += dx[k] * g_h[b]
    hidden = [sum(generator["W2"][a][b] * dx[a] for a in range(len(x_hat)))
              * ref.leaky_grad(g_pre1[b]) for b in range(len(g_h))]
    for j in range(len(g_h)):
        grads["b1"][j] += hidden[j]
        for k in range(len(g_inp)):
            grads["W1"][j][k] += hidden[j] * g_inp[k]


def generator_step(ref, generator, critic, noise, labels, pull, weight, frozen):
    """The lesson's update_g, with a reconstruction gradient added to dL/dx_hat."""
    grads = ref.init_grads(generator)
    for z, label in zip(noise, labels):
        out = ref.g_forward(z, label, generator, CLASSES)
        dx = adversarial_pull(ref, critic, out[0], label)
        if pull is not None:
            extra = pull(ref, frozen, out[0], [CENTRES[label]], label)
            dx = [a + weight * b for a, b in zip(dx, extra)]
        accumulate(ref, generator, z, out, dx, grads)
    ref.apply_grads(generator, grads, G_RATE, len(noise))


def spread_of(ref, generator, rng):
    """{class: (mean, sd)} over PROBE prior draws -- the sharpness the exercise asks about."""
    draws = {c: [ref.g_forward(z, c, generator, CLASSES)[0][0] for z in noise_batch(rng, PROBE)]
             for c in range(CLASSES)}
    return {c: (statistics.fmean(v), statistics.pstdev(v)) for c, v in draws.items()}


def train(ref, random, pull, weight, seed):
    """One arm; returns {class: (mean, sd)} and the frozen extractor it used."""
    rng = random.Random(seed)
    generator = ref.init_mlp(Z_DIM + CLASSES, HIDDEN, 1, rng)
    critic = ref.init_mlp(1 + CLASSES, HIDDEN, 1, rng)
    frozen = snapshot = None
    for step in range(1, STEPS + 1):
        reals = ref.sample_real_conditional(BATCH, CLASSES, rng)
        labels = [c for _, c in reals]
        noise = noise_batch(rng)
        fakes = [(ref.g_forward(z, c, generator, CLASSES)[0], c) for z, c in zip(noise, labels)]
        ref.update_d(reals, fakes, critic, CLASSES, D_RATE)
        if step == WARMUP:
            frozen, snapshot = copy.deepcopy(critic), copy.deepcopy(critic)
        generator_step(ref, generator, critic, noise_batch(rng),
                       [rng.randrange(CLASSES) for _ in range(BATCH)],
                       pull if frozen is not None else None, weight, frozen)
    return spread_of(ref, generator, rng), (frozen == snapshot, frozen != critic)


def arm(ref, random, pull, weight):
    """Per-class (mean, sd) averaged over seeds."""
    rows = [train(ref, random, pull, weight, s)[0] for s in range(SEEDS)]
    return {c: tuple(statistics.fmean(r[c][i] for r in rows) for i in (0, 1))
            for c in range(CLASSES)}


def solve():
    import random
    ref = parity.load_reference(PHASE, LESSON, "main")
    spread, intact = train(ref, random, pull_perceptual, 0.5, 0)
    return {
        "none": arm(ref, random, None, 0.0), "perceptual_low": spread,
        "perceptual_high": arm(ref, random, pull_perceptual, 2.0),
        "l1": arm(ref, random, pull_l1, 1.0), "frozen_intact": intact,
        "dead": (ref.update_g.__defaults__, "l1_w" in ref.update_g.__code__.co_varnames),
    }


def verify(result):
    sds = {k: statistics.fmean(result[n][c][1] for c in range(CLASSES))
           for k, n in (("none", "none"), ("low", "perceptual_low"),
                        ("high", "perceptual_high"), ("l1", "l1"))}
    return [
        practice.Check(
            "ANSWER: it changes sharpness sharply, and every arm overshoots",
            sds["l1"] < sds["high"] < sds["low"] < sds["none"] < REAL_SD,
            f"mean per-class sd against the real {REAL_SD}: the lesson as shipped "
            f"{sds['none']:.3f}, already {REAL_SD / sds['none']:.1f}x too narrow; then perceptual "
            f"at 0.5 {sds['low']:.3f}, at 2.0 {sds['high']:.3f}, L1 at 1.0 {sds['l1']:.3f}, "
            f"{REAL_SD / sds['l1']:.0f}x too sharp. Every arm narrows it and none widens it",
        ),
        practice.Check(
            "FINDING: there is no L1 in use to replace -- the branch never runs",
            result["dead"][0] == (0.0, None) and result["dead"][1],
            f"update_g's reconstruction parameters default to {result['dead'][0]} and main() "
            "calls it with neither, so the l1_w branch never executes in the lesson as shipped. "
            "'Replace L1' has to begin by switching on a term the lesson does not run",
        ),
        practice.Check(
            "FINDING: in 1-D the two losses are one family, and only the weight differs",
            sds["low"] > sds["high"] > sds["l1"],
            f"both pull x_hat to one point per class, so both must collapse it, and the arms lie "
            f"on one monotone curve: {sds['none']:.3f} -> {sds['low']:.3f} -> {sds['high']:.3f} -> "
            f"{sds['l1']:.3f}. Perceptual can only differ from L1 where its features fold two "
            "inputs together, and leaky-ReLU of an affine map of a scalar is monotone unit by "
            "unit, so this distance is provably a reweighted |x - t|",
        ),
        practice.Check(
            "CONTROL: the frozen extractor is frozen, and is not the live critic",
            result["frozen_intact"] == (True, True),
            "the extractor still equals the snapshot taken beside it at warm-up, and it no longer "
            "equals the critic, which has kept training: the perceptual features are a fixed map "
            "rather than a second critic learning alongside the first",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
