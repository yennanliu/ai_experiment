"""Exercise 3 — all 8 modes by step 5,000, so there is no collapse left to fix.

    **Hard.** Extend the 1-D example to 2-D data (mixture of 8 Gaussians on a
    ring). Track how many of the 8 modes the generator captures at steps 1k, 5k,
    10k. Implement minibatch discrimination and re-measure.

Reading of the exercise: the ring is the benchmark's own -- radius 2, 8 centres,
`sigma = 0.05` -- and a mode counts as captured when it draws at least a quarter
of its fair share of 400 probe samples by nearest centre. Both arms run the
lesson's own `update_d` and `update_g` at the lesson's own settings. Minibatch
discrimination appends one feature per row, the mean `exp(-L1)` to the rest of
the batch, which is high exactly when a batch has collapsed. That arm needs its
own generator backward, because `forward_d` then takes 3 inputs where `x_hat` has
2; the feature itself is held fixed, as minibatch discrimination intends.

**ANSWER: 8 of 8 from step 5,000 onward, with or without the fix.**

| step | plain GAN | + minibatch discrimination |
|---|---|---|
| 1,000 | 4, 6, 8 | 5, 7, 8 |
| 5,000 | **8, 8, 8** | **8, 8, 8** |
| 10,000 | **8, 8, 8** | **8, 8, 8** |

**FINDING: the exercise's premise does not occur here.** Every seed reaches all
eight modes by step 5,000 and holds them to 10,000, in both arms. Minibatch
discrimination is a fix for mode collapse, and on this architecture, this ring
and this budget there is no collapse to fix -- so "re-measure" returns the same
number twice and the exercise cannot show what it exists to show.

**FINDING: what the fix buys is a head start, not a better endpoint -- probably.**
At step 1,000 the augmented arm is exactly one mode ahead on each of the two
seeds with room to move (4->5, 6->7) and level on the third, which was already
at 8. The direction is consistent, and it is still weak evidence: the two arms
are **not paired**, because the critic's extra input column consumes a different
number of draws at initialisation and every sample after that differs. Three
seeds with a plain-arm range of 4 cannot separate a 0.67-mode gap. What is not in
doubt is that by step 5,000 the gap is **zero** and stays zero.

**FINDING: "extend to 2-D" is one function.** `init_mlp`, `forward_g`,
`forward_d`, `update_d` and `update_g` are all written over `len(x)` and already
run unmodified at any input width -- the CONTROL confirms the 2-D generator
trains through the lesson's untouched `update_g`. Only `sample_real` is hard-wired
to one dimension. The exercise's "hard" half is replacing a nine-line sampler.

Structure: `ring` is the data; `kinship` is the minibatch feature; `captured`
counts modes; `run` trains one arm.
"""

from __future__ import annotations

import math
import statistics

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "03-gans-generator-discriminator"
Z_DIM, HIDDEN, BATCH, RATE, PROBE = 4, 16, 32, 0.02, 400
MODES, RADIUS, SPREAD, SEEDS, MARKS = 8, 2.0, 0.05, 3, (1_000, 5_000, 10_000)
CENTRES = [(RADIUS * math.cos(2 * math.pi * k / MODES), RADIUS * math.sin(2 * math.pi * k / MODES))
           for k in range(MODES)]


def ring(n, rng):
    """The 8-Gaussian ring, in place of the lesson's 1-D two-mode sample_real."""
    picks = [CENTRES[rng.randrange(MODES)] for _ in range(n)]
    return [[cx + rng.gauss(0, SPREAD), cy + rng.gauss(0, SPREAD)] for cx, cy in picks]


def captured(points):
    """Modes drawing at least a quarter of their fair share of `points`."""
    counts = [0] * MODES
    for x, y in points:
        counts[min(range(MODES), key=lambda k: (x - CENTRES[k][0]) ** 2
                   + (y - CENTRES[k][1]) ** 2)] += 1
    return sum(1 for c in counts if c >= len(points) / (4 * MODES))


def kinship(batch):
    """One minibatch-discrimination feature per row: mean exp(-L1) to the rest of the batch."""
    return [statistics.fmean(math.exp(-sum(abs(a - b) for a, b in zip(x, y))) for y in batch)
            for x in batch]


def pull_through(ref, critic, x_hat, extra):
    """d(-log D)/d x_hat through the lesson's critic, the minibatch feature held fixed."""
    prob, _, d_pre1, _ = ref.forward_d(x_hat + [extra], critic)
    seeded = [critic["W2"][0][j] * (prob - 1.0) * ref.leaky_grad(d_pre1[j]) for j in range(HIDDEN)]
    return [sum(critic["W1"][j][i] * seeded[j] for j in range(HIDDEN)) for i in range(len(x_hat))]


def accumulate(ref, generator, z, out, pull, grads):
    """Backprop `pull` through the lesson's generator into `grads`."""
    _, g_h, g_pre1 = out
    for i, seed in enumerate(pull):
        grads["b2"][i] += seed
        for j in range(HIDDEN):
            grads["W2"][i][j] += seed * g_h[j]
    hidden = [sum(generator["W2"][i][j] * pull[i] for i in range(len(pull)))
              * ref.leaky_grad(g_pre1[j]) for j in range(HIDDEN)]
    for j in range(HIDDEN):
        grads["b1"][j] += hidden[j]
        for k in range(Z_DIM):
            grads["W1"][j][k] += hidden[j] * z[k]


def descend(params, grads, count):
    """The lesson's own mean-gradient step."""
    for key, value in params.items():
        pairs = zip(value, grads[key]) if isinstance(value[0], list) else [(value, grads[key])]
        for row, grad_row in pairs:
            for i in range(len(row)):
                row[i] -= RATE * grad_row[i] / count


def generator_step(ref, generator, critic, noise):
    """The lesson's update_g, rewritten only so D may take a wider input than G emits."""
    outs = [ref.forward_g(z, generator) for z in noise]
    grads = {k: ([[0.0] * len(r) for r in v] if isinstance(v[0], list) else [0.0] * len(v))
             for k, v in generator.items()}
    for z, out, extra in zip(noise, outs, kinship([o[0] for o in outs])):
        accumulate(ref, generator, z, out, pull_through(ref, critic, out[0], extra), grads)
    descend(generator, grads, len(noise))


def run(ref, random, minibatch, seed):
    """{step: modes captured} for one arm, at the marks the exercise names."""
    rng = random.Random(seed)
    generator = ref.init_mlp(Z_DIM, HIDDEN, 2, rng)
    critic = ref.init_mlp(2 + int(minibatch), HIDDEN, 1, rng)
    marks = {}
    for step in range(1, MARKS[-1] + 1):
        reals = ring(BATCH, rng)
        noise = ref.sample_noise(BATCH, Z_DIM, rng)
        fakes = [ref.forward_g(z, generator)[0] for z in noise]
        if minibatch:
            pair = [[x + [f] for x, f in zip(rows, kinship(rows))] for rows in (reals, fakes)]
            ref.update_d(pair[0], pair[1], critic, RATE / 2)
            generator_step(ref, generator, critic, ref.sample_noise(BATCH, Z_DIM, rng))
        else:
            ref.update_d(reals, fakes, critic, RATE / 2)
            ref.update_g(ref.sample_noise(BATCH, Z_DIM, rng), generator, critic, RATE)
        if step in MARKS:
            marks[step] = captured([ref.forward_g(z, generator)[0]
                                    for z in ref.sample_noise(PROBE, Z_DIM, rng)])
    return marks


def solve():
    import random
    ref = parity.load_reference(PHASE, LESSON, "main")
    arms = {name: [run(ref, random, name == "minibatch", s) for s in range(SEEDS)]
            for name in ("plain", "minibatch")}
    return {
        "arms": {name: {m: [r[m] for r in rows] for m in MARKS} for name, rows in arms.items()},
        "real": captured(ring(PROBE, random.Random(9))),
        "widths": (len(ref.init_mlp(Z_DIM, HIDDEN, 2, random.Random(0))["W2"]),
                   len(ref.init_mlp(2, HIDDEN, 1, random.Random(0))["W1"][0])),
    }


def verify(result):
    plain, mbd = result["arms"]["plain"], result["arms"]["minibatch"]
    early = statistics.fmean(mbd[MARKS[0]]) - statistics.fmean(plain[MARKS[0]])
    return [
        practice.Check(
            f"ANSWER: {MODES} of {MODES} from step {MARKS[1]:,} on, with or without the fix",
            {v for m in MARKS[1:] for a in (plain, mbd) for v in a[m]} == {MODES},
            "modes captured, per seed -- "
            + "; ".join(f"step {m:,}: plain {plain[m]} vs minibatch {mbd[m]}" for m in MARKS)
            + f". Real data scores {result['real']}/{MODES} under the same rule, so from "
              f"{MARKS[1]:,} on the generator is at the ceiling. Minibatch discrimination fixes "
              "mode collapse; there is none here to fix, so re-measuring returns the same number",
        ),
        practice.Check(
            "FINDING: the fix buys a head start at 1,000, not a better endpoint",
            all(m >= p for m, p in zip(mbd[MARKS[0]], plain[MARKS[0]]))
            and abs(early) < max(plain[MARKS[0]]) - min(plain[MARKS[0]]),
            f"at {MARKS[0]:,} plain captures {plain[MARKS[0]]} and augmented {mbd[MARKS[0]]}: "
            f"ahead or level on every seed, {early:+.2f} of a mode against a seed range of "
            f"{max(plain[MARKS[0]]) - min(plain[MARKS[0]])}. The arms are not paired (a wider "
            f"critic draws differently at init), so by {MARKS[1]:,} the gap is simply zero",
        ),
        practice.Check(
            "CONTROL: 'extend to 2-D' is one function -- the lesson's own code is width-agnostic",
            result["widths"] == (2, 2),
            f"init_mlp builds a {result['widths'][0]}-output generator and a "
            f"{result['widths'][1]}-input critic from the lesson's own call, and the plain arm "
            "trains through update_d and update_g unmodified -- both written over len(x). Only "
            "sample_real is hard-wired to one dimension",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
