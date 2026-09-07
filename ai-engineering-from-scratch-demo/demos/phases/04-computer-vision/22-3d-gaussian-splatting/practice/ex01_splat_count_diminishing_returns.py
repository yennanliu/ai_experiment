"""Exercise 1 — splat count diminishing returns.

    **(Easy)** Run the 2D splat trainer above on a different synthetic image. Vary `num_splats` in `[16, 64, 256]` and plot MSE vs step for each. Identify the point of diminishing returns.

Reading of the exercise: "identify the point of diminishing returns" presumes
there is one on the axis it asks you to plot, and over `[16, 64, 256]` there is
not. Every 4x in `num_splats` buys *more* than 4x in best MSE, so the
step-indexed curve bends the wrong way for the question. A step is also not a
unit of work: `rasterise_2d` composites in a Python `for` loop over splats, so
one 256-splat step costs the wall clock about 25x what one 16-splat step costs.
Re-plot against splat-steps -- count times steps, a number no clock enters --
and the knee appears where the exercise expects one, near 12,800:
below that budget 256 splats is the worst of the three arms, above it the best.
Two properties of the lesson's own loop surface while running it. Half the six
runs do not reach step 300 in a usable state -- one ends NaN, two end several
times above their own best -- so "MSE vs step" is not the monotone curve the
exercise pictures, and reading the *final* MSE off such a plot ranks the arms
differently from reading the best. And `rasterise_2d` accumulates into
`torch.zeros` with no background term, so any pixel the splats miss renders
black; a quarter to a third of the frame never accumulates half a unit of
weight at any splat count, which is the floor under every MSE here.

Structure: `target` is the different synthetic image the exercise asks for --
yellow diagonal bands, a blue ring, a dark-green ground; `coverage` reports how
much of the frame the trained splats actually paint, read off the same clamped
alphas `rasterise_2d` composites; `fit` runs the lesson's own `Splats2D` under
Adam for 300 steps at one (seed, count) and records the loss history, the wall
clock and that coverage. `best_at` is the running best over a prefix of a
history, which is what lets one set of runs answer the equal-compute question
as well as the step-indexed one; `survives` is "finite, and within 1.5x of its
own best" -- the weakest reading of "the run finished" that the plot needs;
`tables` and the module-level lambdas above it exist to keep `verify` inside
D14's complexity ceiling. At 143 code lines this is over D14's 120-line target
and seven clear of its ceiling: the overrun is the second seed, without which
"half the runs do not survive" would rest on a single draw.
"""

from __future__ import annotations

import math
import time

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "22-3d-gaussian-splatting"

SIZE, STEPS, LR = 40, 300, 0.08
COUNTS, SEEDS = (16, 64, 256), (0, 1)
BUDGETS = (1600, 4800, 12800, 19200)

best_at = lambda hist, upto: min(v for v in hist[:upto] if math.isfinite(v))     # noqa: E731
survives = lambda hist: math.isfinite(hist[-1]) and hist[-1] <= 1.5 * best_at(hist, STEPS)  # noqa: E731
compute = lambda runs: {(s, b): {n: best_at(runs[(s, n)]["history"], b // n)     # noqa: E731
                                 for n in COUNTS if 1 <= b // n <= STEPS}
                        for s in SEEDS for b in BUDGETS}
crosses = lambda t: (all(t[(s, 4800)][64] * 2 < t[(s, 4800)][256] for s in SEEDS)  # noqa: E731
                     and all(t[(s, 19200)][256] < t[(s, 19200)][64] for s in SEEDS))
best_row = lambda best: "  ".join(f"{n}->{best[(0, n)]:.5f}/{best[(1, n)]:.5f}" for n in COUNTS)  # noqa: E731
budget_row = lambda t: "; ".join(f"{b:,}: " + " ".join(                         # noqa: E731
    f"{n}={t[(0, b)][n]:.5f}/{t[(1, b)][n]:.5f}" for n in sorted(t[(0, b)])) for b in BUDGETS)
alive_row = lambda runs, best, alive: "  ".join(                                # noqa: E731
    f"s{s}/{n} {runs[(s, n)]['history'][-1]:.5f} vs {best[(s, n)]:.5f} "
    f"{'ok' if alive[(s, n)] else 'LOST'}" for s in SEEDS for n in COUNTS)
paint_row = lambda p: "  ".join(f"{n} splats {p[n][0]:.3f}, {p[n][1]:.1%}" for n in COUNTS)  # noqa: E731


def target(torch):
    rows, cols = torch.meshgrid(torch.arange(SIZE, dtype=torch.float32),
                                torch.arange(SIZE, dtype=torch.float32), indexing="ij")
    image = torch.zeros(SIZE, SIZE, 3)
    image[..., 1] = 0.25
    image[((cols + rows) // 7) % 2 == 0] = torch.tensor([0.9, 0.85, 0.1])
    image[((cols - 20) ** 2 + (rows - 20) ** 2 - 144).abs() < 50] = torch.tensor([0.1, 0.2, 0.85])
    return image


def coverage(torch, ref, model) -> tuple:
    rows, cols = torch.meshgrid(torch.arange(SIZE, dtype=torch.float32),
                                torch.arange(SIZE, dtype=torch.float32), indexing="ij")
    with torch.no_grad():
        density = ref.eval_2d_gaussian(model.means, model.covs(),
                                       torch.stack([cols, rows], dim=-1))
        alpha = (torch.sigmoid(model.opacity_logit)[:, None, None] * density).clamp(0.0, 0.99)
        weight = 1.0 - torch.prod(1.0 - alpha, dim=0)
    return float(weight.mean()), float((weight < 0.5).to(torch.float32).mean())


def fit(torch, ref, count, seed, goal) -> dict:
    torch.manual_seed(seed)
    model = ref.Splats2D(num_splats=count, image_size=SIZE, seed=seed)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    history, clock = [], time.perf_counter()
    for _ in range(STEPS):
        loss = ((model((SIZE, SIZE)) - goal) ** 2).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        history.append(loss.item())
    return {"history": history, "seconds": time.perf_counter() - clock,
            "coverage": coverage(torch, ref, model)}


def tables(runs) -> dict:
    best = {key: best_at(run["history"], STEPS) for key, run in runs.items()}
    return {"best": best,
            "gain": {s: (best[(s, 16)] / best[(s, 64)], best[(s, 64)] / best[(s, 256)])
                     for s in SEEDS},
            "alive": {key: survives(run["history"]) for key, run in runs.items()},
            "seconds": {n: sum(runs[(s, n)]["seconds"] for s in SEEDS) / len(SEEDS)
                        for n in COUNTS},
            "paint": {n: runs[(0, n)]["coverage"] for n in COUNTS}}


def solve():
    try:
        import torch
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    goal = target(torch)
    return {"runs": {(seed, count): fit(torch, ref, count, seed, goal)
                     for seed in SEEDS for count in COUNTS},
            "black": float((goal ** 2).mean()),
            "constant": float(((goal - goal.mean(dim=(0, 1))) ** 2).mean())}


def verify(result):
    runs, view = result["runs"], tables(result["runs"])
    best, gain, alive = view["best"], view["gain"], view["alive"]
    seconds, paint, budget = view["seconds"], view["paint"], compute(runs)
    return [
        practice.Check(
            "ANSWER: on the axis the exercise plots there is no knee — returns accelerate",
            all(min(pair) > 4.0 for pair in gain.values()),
            f"best MSE over 300 steps, seed 0 / seed 1: {best_row(best)}"
            + f". Each 4x in num_splats divides MSE by {gain[0][0]:.1f}x then {gain[0][1]:.1f}x "
            f"(seed 0) and {gain[1][0]:.1f}x then {gain[1][1]:.1f}x (seed 1) — super-linear both "
            "times, so the step-indexed curve has no point of diminishing returns to identify"),
        practice.Check(
            "FINDING: the knee is on the compute axis, and it sits near 12,800 splat-steps",
            crosses(budget),
            f"best MSE at equal count x steps (seed 0 / seed 1): {budget_row(budget)}"
            + ". 256 splats is the worst arm below ~12,800 and the best above it; the two curves "
            "cross there on both seeds, which is the answer the exercise wants"),
        practice.Check(
            "FINDING: half the runs never reach step 300 in a usable state",
            sum(alive.values()) <= len(alive) // 2 and not any(alive[(s, 256)] for s in SEEDS),
            "final vs best MSE, and whether the run stayed within 1.5x of its own best: "
            + alive_row(runs, best, alive)
            + f". {sum(alive.values())} of {len(alive)} survive; neither 256-splat run does. Ranking "
            "the arms by *final* MSE instead of best would therefore report a different winner"),
        practice.Check(
            "MECHANISM: a step is not a unit — `rasterise_2d` loops over splats in Python",
            seconds[256] > 8 * seconds[16],
            f"mean wall clock for 300 steps: 16 splats {seconds[16]:.2f}s, 64 {seconds[64]:.2f}s, "
            f"256 {seconds[256]:.2f}s — {seconds[256] / seconds[16]:.0f}x for 16x the splats, "
            f"and {seconds[64] / seconds[16]:.1f}x then {seconds[256] / seconds[64]:.1f}x per 4x. "
            "The compositing `for i in range(means.size(0))` is sequential and cannot vectorise: "
            "each iteration needs the transmittance the previous one left behind"),
        practice.Check(
            "CONTROL: the renderer has no background, so a third of the frame is floored at black",
            all(0.2 < paint[n][1] < 0.4 for n in COUNTS)
            and result["black"] > result["constant"] > best[(0, 16)],
            "`rasterise_2d` accumulates into `torch.zeros(H, W, 3)`, so a pixel is only as bright as "
            "the weight it collects. Trained (seed 0), mean accumulated weight and the share of "
            f"pixels below 0.5: {paint_row(paint)}"
            + f" — a quarter to a third of the frame renders dark at *every* count, so 16x more "
            f"splats does not buy coverage. For scale, an all-black image scores "
            f"{result['black']:.3f} on this target and the best constant image "
            f"{result['constant']:.3f}, so 16 splats at {best[(0, 16)]:.5f} beat a flat colour by "
            f"only {result['constant'] / best[(0, 16)]:.1f}x"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
