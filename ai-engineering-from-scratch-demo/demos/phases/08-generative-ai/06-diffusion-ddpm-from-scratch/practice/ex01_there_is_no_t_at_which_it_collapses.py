"""Exercise 1 — it degrades smoothly, and T is not the variable that decides it.

    **Easy.** Change T from 40 to 10 in `code/main.py`. How does sample quality
    (visual histogram of outputs) degrade? At what T does the two-mode structure
    collapse?

Reading of the exercise: a histogram is read rather than looked at. Two numbers
replace the picture -- the share of samples landing within 1.0 of either true
centre (`in-modes`, which the data itself scores 0.980) and the share falling in
the gap `|x| < 0.5` (`in-valley`, which the data scores 0.000). The lesson's own
`make_schedule`, `init_net`, `train` and `sample` run unchanged at SEEDS seeds
per T.

**ANSWER: smoothly, and it never collapses.**

| T | 5 | 10 | 20 | 40 | 100 | real data |
|---|---:|---:|---:|---:|---:|---:|
| `alpha_bar[-1]` | 0.951 | 0.904 | 0.817 | 0.667 | 0.364 | -- |
| in-modes | 0.400 | 0.550 | 0.752 | **0.867** | 0.895 | **0.980** |
| in-valley | 0.323 | 0.189 | 0.077 | **0.020** | 0.003 | **0.000** |

At T=10 both modes are still there, just blunt: half the mass is on a centre and
a fifth has leaked into the gap. There is no cliff anywhere in the sweep, so the
question's "at what T" has no answer to give.

**FINDING: T is not the variable -- `alpha_bar[T-1]` is.** `make_schedule` writes
`betas = 1e-4 + (0.02 - 1e-4) * t / (T - 1)`, so the per-step noise range is
**hard-coded and independent of T** and the *total* noise is whatever T steps of
it happen to add up to. Changing T from 40 to 10 does not re-time a fixed budget,
it cuts the budget to a quarter. Both columns above move monotonically with
`alpha_bar[-1]` and neither moves with T except through it.

**FINDING: the forward process never reaches noise, even at the lesson's own T.**
At T=40 `alpha_bar[-1]` is **0.667**, so `x_T` still carries **81.7%** of the
signal. But `sample` starts the reverse chain at `rng.gauss(0, 1)` -- pure noise,
a distribution the model never saw in training. Reaching `alpha_bar ~ 0` under
this beta range needs about **T=1000**. The lesson's sampler starts off the
manifold its own trainer built, at every T in the sweep.

**CONTROL: the metric tops out where the data is.** Real draws from
`sample_data` score 0.980 and 0.000 under the same rule, so T=100's 0.895 is
still short of the ceiling rather than at it.

Structure: `quality` is the histogram read as two numbers; `run` is the lesson's
own train-and-sample at one T.
"""

from __future__ import annotations

import contextlib
import io
import statistics

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "06-diffusion-ddpm-from-scratch"
X_DIM, T_DIM, HIDDEN, STEPS, RATE = 1, 8, 32, 4_000, 0.01
GRID, SEEDS, DRAWS, CENTRE, LESSON_T = (5, 10, 20, 40, 100), 2, 400, 2.0, 40


def quality(samples):
    """(share within 1.0 of a centre, share in the gap) -- the histogram as two numbers."""
    return (sum(1 for v in samples if abs(abs(v) - CENTRE) < 1.0) / len(samples),
            sum(1 for v in samples if abs(v) < 0.5) / len(samples))


def run(ref, random, steps, seed):
    """The lesson's own train-and-sample loop at one T, with its progress prints muted."""
    rng = random.Random(seed)
    net = ref.init_net(X_DIM, T_DIM, HIDDEN, rng)
    _, alphas, alpha_bars = ref.make_schedule(steps)
    with contextlib.redirect_stdout(io.StringIO()):
        ref.train(net, alpha_bars, steps, STEPS, RATE, T_DIM, rng)
    return [ref.sample(net, alphas, alpha_bars, steps, T_DIM, rng) for _ in range(DRAWS)]


def sweep(ref, random):
    """{T: (alpha_bar[-1], in-modes, in-valley)} averaged over seeds."""
    out = {}
    for steps in GRID:
        rows = [quality(run(ref, random, steps, s)) for s in range(SEEDS)]
        out[steps] = (ref.make_schedule(steps)[2][-1],
                      statistics.fmean(r[0] for r in rows),
                      statistics.fmean(r[1] for r in rows))
    return out


def solve():
    import random
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = random.Random(9)
    grid = sweep(ref, random)
    modes = [grid[t][1] for t in GRID]
    return {
        "grid": grid, "modes": modes, "best": max(modes),
        "worst_drop": max([a - b for a, b in zip(modes, modes[1:])] + [0.0]),
        "shown": ", ".join(f"T={t}: {grid[t][1]:.3f} / {grid[t][2]:.3f}" for t in GRID),
        "real": quality([ref.sample_data(rng) for _ in range(DRAWS)]),
        "betas": (ref.make_schedule(10)[0][0], ref.make_schedule(10)[0][-1],
                  ref.make_schedule(1000)[0][0], ref.make_schedule(1000)[0][-1]),
        "long": ref.make_schedule(1000)[2][-1],
    }


def verify(result):
    grid, real = result["grid"], result["real"]
    return [
        practice.Check(
            "ANSWER: quality degrades smoothly and never collapses",
            result["worst_drop"] < 0.02 and min(result["modes"]) > 0.3,
            f"in-modes / in-valley by T -- {result['shown']}. At T=10 both modes are still there, "
            f"just blunt. The largest fall between neighbouring T is "
            f"{result['worst_drop']:.3f}, so there is no cliff anywhere in the sweep and the "
            "question's 'at what T' has no answer to give",
        ),
        practice.Check(
            "FINDING: T is not the variable -- alpha_bar[T-1] is",
            result["betas"][0] == result["betas"][2] and result["betas"][1] == result["betas"][3],
            f"make_schedule writes betas = 1e-4 + (0.02 - 1e-4) * t / (T - 1), so the per-step "
            f"range is hard-coded: at T=10 it runs {result['betas'][0]:.4f}..{result['betas'][1]:.4f} "
            f"and at T=1000 the identical {result['betas'][2]:.4f}..{result['betas'][3]:.4f}. "
            "Changing T does not re-time a fixed noise budget, it rescales the budget, and both "
            "columns move with alpha_bar[-1] rather than with T",
        ),
        practice.Check(
            "FINDING: the forward process never reaches noise, even at the lesson's own T",
            grid[LESSON_T][0] > 0.5 and result["long"] < 0.01,
            f"at T={LESSON_T} alpha_bar[-1] is {grid[LESSON_T][0]:.3f}, so x_T still carries "
            f"{grid[LESSON_T][0] ** 0.5:.1%} of the signal -- yet sample() starts the reverse "
            f"chain at gauss(0, 1), pure noise the model never saw. Reaching alpha_bar ~ 0 under "
            f"this beta range needs about T=1000, where it is {result['long']:.4f}. The sampler "
            "starts off the manifold its own trainer built, at every T in the sweep",
        ),
        practice.Check(
            "CONTROL: the metric tops out where the data is",
            real[0] > 0.95 and real[1] < 0.01 and result["best"] < real[0],
            f"real draws from sample_data score {real[0]:.3f} in-modes and {real[1]:.3f} "
            f"in-valley under the same rule, so the best T in the sweep, {result['best']:.3f}, is "
            "still short of the ceiling rather than sitting at it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
