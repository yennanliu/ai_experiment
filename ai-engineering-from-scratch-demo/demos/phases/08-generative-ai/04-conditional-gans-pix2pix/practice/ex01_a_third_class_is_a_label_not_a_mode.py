"""Exercise 1 — a third class adds a label, not a mode, so the confirmation is vacuous.

    **Easy.** Modify `code/main.py` to add a third class. Confirm G still maps
    each class's noise to the correct mode.

Reading of the exercise: "add a third class" is done the way the code invites --
`num_classes = 3`, which widens the one-hot on both nets and nothing else -- and
then the confirmation is actually run, per class, over 300 prior draws on SEEDS
seeds. The lesson's own `sample_real_conditional`, `g_forward`, `d_forward`,
`update_d` and `update_g` are used throughout; only the class count moves.

**ANSWER: yes, and for the third class that means nothing.** G maps class 0 to
**-2.2** and classes 1 and 2 to **+2.0** and **+2.2**. Every class lands on its
correct mode, because classes 1 and 2 *have the same correct mode*.

**FINDING: `sample_real_conditional` is `if c == 0 ... else`.** Every class above
zero draws from `gauss(+2, 0.3)`. At `num_classes = 3` the real data's class means
are **-1.99, +2.00, +2.00**: two modes wearing three labels. The exercise asks to
confirm a mapping the data does not define, and the confirmation passes because
being wrong about class 2 is indistinguishable from being right about class 1.

**FINDING: give the third class a mode of its own and G finds it.** With centres
at -2, 0 and +2 the same network, same budget and same seeds gives **-2.10**,
**+0.05**, **+1.96** -- all three separated, so the capacity was never the
problem. The one-line sampler was.

**FINDING: the lesson's 600 steps cannot confirm anything.** At its own budget
even the stock two-class run is unconverged: on 1 of SEEDS seeds it puts *both*
classes on the same side of zero (**-1.27** and **-0.79**). The confirmation the
exercise asks for first needs about **3,000** steps, five times what `main()`
runs.

**CONTROL: widening the label is one integer.** `one_hot`, `g_forward` and
`d_forward` are already written over `num_classes`; the only edit is the two
`init_mlp` widths. Nothing about "add a third class" is hard except noticing that
the data did not get one.

Structure: `three_centre` is the sampler the exercise needs; `train` is the
lesson's own loop; `means` is the per-class confirmation.
"""

from __future__ import annotations

import statistics

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "04-conditional-gans-pix2pix"
Z_DIM, HIDDEN, BATCH, G_RATE, D_RATE = 4, 16, 32, 0.02, 0.01
LESSON_STEPS, ENOUGH, SEEDS, PROBE, GAP = 600, 3_000, 3, 300, 0.3


def three_centre(n, classes, rng):
    """What `sample_real_conditional` would be if a third class meant a third mode."""
    return [([rng.gauss(-2.0 + 2.0 * c, GAP)], c)
            for c in (rng.randrange(classes) for _ in range(n))]


def noise_batch(rng, count=BATCH):
    """A batch of z vectors, drawn the way main() draws them."""
    return [[rng.gauss(0, 1) for _ in range(Z_DIM)] for _ in range(count)]


def train(ref, random, classes, sampler, seed, steps):
    """The lesson's own conditional-GAN loop; returns {class: mean sample}."""
    rng = random.Random(seed)
    generator = ref.init_mlp(Z_DIM + classes, HIDDEN, 1, rng)
    critic = ref.init_mlp(1 + classes, HIDDEN, 1, rng)
    for _ in range(steps):
        reals = sampler(BATCH, classes, rng)
        labels = [c for _, c in reals]
        noise = noise_batch(rng)
        fakes = [(ref.g_forward(noise[i], labels[i], generator, classes)[0], labels[i])
                 for i in range(BATCH)]
        ref.update_d(reals, fakes, critic, classes, D_RATE)
        ref.update_g(noise_batch(rng), [rng.randrange(classes) for _ in range(BATCH)],
                     generator, critic, classes, G_RATE)
    probe = noise_batch(rng, PROBE)
    return {c: statistics.fmean(ref.g_forward(z, c, generator, classes)[0][0] for z in probe)
            for c in range(classes)}


def means(ref, random, classes, sampler, steps):
    """Per-class mean sample, averaged over seeds, plus every seed's own dict."""
    rows = [train(ref, random, classes, sampler, s, steps) for s in range(SEEDS)]
    return {c: statistics.fmean(r[c] for r in rows) for c in range(classes)}, rows


def solve():
    import random
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = random.Random(1)
    real = ref.sample_real_conditional(3_000, 3, rng)
    return {
        "real": {c: statistics.fmean(x[0] for x, k in real if k == c) for c in range(3)},
        "lesson": means(ref, random, 3, ref.sample_real_conditional, ENOUGH)[0],
        "fixed": means(ref, random, 3, three_centre, ENOUGH)[0],
        "short": means(ref, random, 2, ref.sample_real_conditional, LESSON_STEPS)[1],
        "widths": (len(ref.init_mlp(Z_DIM + 3, HIDDEN, 1, random.Random(0))["W1"][0]),
                   len(ref.one_hot(2, 3))),
    }


def verify(result):
    real, lesson, fixed = result["real"], result["lesson"], result["fixed"]
    same_side = [r for r in result["short"] if (r[0] > 0) == (r[1] > 0)]
    return [
        practice.Check(
            "ANSWER: yes -- and for the third class the confirmation means nothing",
            abs(lesson[1] - lesson[2]) < abs(lesson[0] - lesson[1]) / 2,
            f"G maps the three classes to {lesson[0]:+.2f}, {lesson[1]:+.2f} and "
            f"{lesson[2]:+.2f}. Every class lands on its correct mode, and classes 1 and 2 have "
            f"the same correct mode: they sit {abs(lesson[1] - lesson[2]):.2f} apart against "
            f"{abs(lesson[0] - lesson[1]):.2f} between classes 0 and 1",
        ),
        practice.Check(
            "FINDING: sample_real_conditional is `if c == 0 ... else`, so class 2 is class 1",
            abs(real[1] - real[2]) < 0.05 and abs(real[0] - real[1]) > 3,
            f"the real data's class means at num_classes=3 are {real[0]:+.2f}, {real[1]:+.2f} and "
            f"{real[2]:+.2f} -- two modes wearing three labels, because every class above zero "
            "draws gauss(+2, 0.3). The exercise asks to confirm a mapping the data does not "
            "define, and it passes because being wrong about class 2 looks like being right",
        ),
        practice.Check(
            "FINDING: give the third class a mode and the same network finds it",
            min(abs(fixed[a] - fixed[b]) for a, b in ((0, 1), (1, 2), (0, 2))) > 0.8,
            f"with centres at -2, 0 and +2 the same architecture, budget and seeds give "
            f"{fixed[0]:+.2f}, {fixed[1]:+.2f} and {fixed[2]:+.2f}, the closest pair "
            f"{min(abs(fixed[a] - fixed[b]) for a, b in ((0, 1), (1, 2), (0, 2))):.2f} apart. "
            "Capacity was never the constraint; the one-line sampler was",
        ),
        practice.Check(
            "FINDING: the lesson's own 600 steps cannot confirm anything",
            len(same_side) > 0,
            f"at main()'s budget the stock two-class run is unconverged: on "
            f"{len(same_side)} of {SEEDS} seeds both classes land on the same side of zero "
            + "; ".join(f"({r[0]:+.2f}, {r[1]:+.2f})" for r in same_side)
            + f". The confirmation needs about {ENOUGH:,} steps, {ENOUGH // LESSON_STEPS}x what "
              "the lesson runs, before it is a measurement rather than a coin flip",
        ),
        practice.Check(
            "CONTROL: widening the label is one integer, not an architecture change",
            result["widths"] == (Z_DIM + 3, 3),
            f"init_mlp takes a {result['widths'][0]}-wide generator input at num_classes=3 and "
            f"one_hot returns {result['widths'][1]} slots; g_forward, d_forward and one_hot are "
            "already written over num_classes, so the only edit is the two init_mlp widths",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
