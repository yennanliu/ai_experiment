"""Exercise 1 — it does not collapse, because this is the non-saturating loss.

    **Easy.** Run `code/main.py` with the stock settings. Then set
    `D_LR = 5 * G_LR` and rerun. How fast does G's loss collapse to a constant?

Reading of the exercise: the lesson's own `update_d` and `update_g` are driven at
the lesson's own settings -- `z_dim=4`, hidden 16, batch 32, 800 steps, `g_lr`
0.02 -- and only the discriminator's learning rate changes. Neither update
function returns a loss, so G's loss is computed the way `update_g` defines it,
`-log D(G(z))`, on the same batch it just stepped on. SEEDS seeds per setting,
because a single run's tail wanders by more than the effect being looked for.

**ANSWER: it does not collapse, and the two settings are indistinguishable.**

| d_lr | G loss, last 100 steps | within-run sd | D(real) | D(fake) |
|---|---:|---:|---:|---:|
| 0.5x g_lr (stock) | 0.791 | 0.025 | 0.545 | 0.451 |
| 5x g_lr | 0.787 | 0.029 | 0.549 | 0.457 |

The means differ by **0.004**, while the same quantity spans **0.425** across the
six seeds of a single setting -- **115x** the gap. Three seeds are not enough to
see this: on seeds 0-2 alone the two settings differ by 0.085 and look separated,
which is why the count is six. There is no collapse to time.

**FINDING: the exercise asks how fast the non-saturating loss saturates.**
`update_g`'s own docstring says "Non-saturating G loss: maximize log D(G(z))".
The loss that goes flat when D wins is the *saturating* one, `log(1 - D(G(z)))`,
whose gradient vanishes as `D(G(z)) -> 0`. Goodfellow 2014 introduced the form
this lesson implements specifically to remove that failure, so the exercise is
asking for a symptom the code was written to not have.

**FINDING: D never wins here, at any ratio tried.** Across `d_lr` from 0.5x to
**50x** `g_lr`, `D(fake)` stays between 0.43 and 0.46 and `D(real)` between 0.54
and 0.60 -- a discriminator at chance, which is equilibrium rather than
dominance. What a 50x ratio buys is **variance**: the within-run tail deviation
of G's loss rises to **0.136** against the stock **0.025**, 5.5x noisier.
Over-training D makes this run jumpier, not deader.

**CONTROL: `D_LR` and `G_LR` do not exist.** They are locals named `d_lr` and
`g_lr` inside `main()`, so the edit the exercise describes -- setting a constant
-- cannot be made the way it is written.

Structure: `g_loss` is the loss `update_g` implies; `run` is the lesson's own
training loop with one knob; `sweep` runs it over ratios and seeds.
"""

from __future__ import annotations

import math
import statistics

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "03-gans-generator-discriminator"
Z_DIM, HIDDEN, BATCH, STEPS, G_RATE = 4, 16, 32, 800, 0.02
RATIOS, SEEDS, TAIL = (0.5, 5.0, 50.0), 6, 100


def g_loss(ref, noise, generator, discriminator):
    """-log D(G(z)) over a batch: the loss update_g takes its gradient from."""
    probs = [ref.forward_d(ref.forward_g(z, generator)[0], discriminator)[0] for z in noise]
    return statistics.fmean(-math.log(max(p, 1e-12)) for p in probs)


def run(ref, random, ratio, seed):
    """The lesson's own loop with d_lr = ratio * g_lr; returns its loss trace and D's verdicts."""
    rng = random.Random(seed)
    generator = ref.init_mlp(Z_DIM, HIDDEN, 1, rng)
    discriminator = ref.init_mlp(1, HIDDEN, 1, rng)
    trace = []
    for _ in range(STEPS):
        noise = ref.sample_noise(BATCH, Z_DIM, rng)
        fakes = [ref.forward_g(z, generator)[0] for z in noise]
        ref.update_d(ref.sample_real(BATCH, rng), fakes, discriminator, G_RATE * ratio)
        noise = ref.sample_noise(BATCH, Z_DIM, rng)
        ref.update_g(noise, generator, discriminator, G_RATE)
        trace.append(g_loss(ref, noise, generator, discriminator))
    probe = [ref.forward_g(z, generator)[0][0] for z in ref.sample_noise(400, Z_DIM, rng)]
    return (statistics.fmean(trace[-TAIL:]), statistics.pstdev(trace[-TAIL:]),
            statistics.fmean(ref.forward_d(x, discriminator)[0]
                             for x in ref.sample_real(200, rng)),
            statistics.fmean(ref.forward_d([v], discriminator)[0] for v in probe))


def sweep(ref, random):
    """{ratio: (per-seed tail means, mean, tail sd, D(real), D(fake))}."""
    grid = {}
    for ratio in RATIOS:
        rows = list(zip(*(run(ref, random, ratio, s) for s in range(SEEDS))))
        grid[ratio] = (list(rows[0]),) + tuple(statistics.fmean(col) for col in rows)
    return grid


def solve():
    import random
    ref = parity.load_reference(PHASE, LESSON, "main")
    grid = sweep(ref, random)
    return {
        "grid": grid,
        "named": [n for n in ("D_LR", "G_LR") if hasattr(ref, n)],
        "docstring": (ref.update_g.__doc__ or "").strip(),
    }


def verify(result):
    grid = result["grid"]
    stock, faster, extreme = (grid[r] for r in RATIOS)
    spread = max(max(g[0]) - min(g[0]) for g in (stock, faster))
    return [
        practice.Check(
            "ANSWER: G's loss does not collapse -- the two settings are indistinguishable",
            abs(faster[1] - stock[1]) < spread / 4,
            f"G's loss over the last {TAIL} steps, averaged over {SEEDS} seeds: {stock[1]:.3f} at "
            f"the stock d_lr = {RATIOS[0]}x g_lr against {faster[1]:.3f} at {RATIOS[1]}x, a gap of "
            f"{abs(faster[1] - stock[1]):.3f}. The same quantity spans {spread:.3f} across seeds "
            f"within a single setting -- {spread / max(abs(faster[1] - stock[1]), 1e-9):.0f}x the "
            "gap -- so the experiment cannot separate them. There is no collapse to time",
        ),
        practice.Check(
            "FINDING: the exercise asks how fast the non-saturating loss saturates",
            "Non-saturating" in result["docstring"],
            f"update_g's own docstring reads {result['docstring']!r}. The loss that goes flat when "
            "D wins is the saturating log(1 - D(G(z))), whose gradient vanishes as D(G(z)) -> 0; "
            "the non-saturating -log D(G(z)) this lesson implements was introduced precisely to "
            "remove that failure. The symptom asked about was engineered out of the code",
        ),
        practice.Check(
            "FINDING: D never wins, and a 50x ratio buys variance rather than collapse",
            all(0.35 < grid[r][4] < 0.55 for r in RATIOS) and extreme[2] > 3 * stock[2],
            "D(real) / D(fake) at each ratio -- "
            + ", ".join(f"{r}x: {grid[r][3]:.3f} / {grid[r][4]:.3f}" for r in RATIOS)
            + f". D stays at chance throughout, which is equilibrium and not dominance. What "
              f"{RATIOS[-1]}x changes is the within-run tail deviation of G's loss: "
              f"{extreme[2]:.3f} against {stock[2]:.3f}, {extreme[2] / stock[2]:.1f}x noisier "
              "rather than flatter",
        ),
        practice.Check(
            "CONTROL: D_LR and G_LR are not constants, so the edit cannot be made as written",
            result["named"] == [],
            f"the module exposes {result['named']} of the two names the exercise says to set: they "
            "are locals d_lr and g_lr inside main(), assigned on one line with the batch size. "
            "The instruction describes editing a constant that is not there",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
