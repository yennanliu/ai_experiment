"""Exercise 1 -- samples per second flattens by effective batch 16-32 and is flat past 64; the loss never leaves ln 16.

    Re-run the sweep with `--num-steps 100` and plot samples per second against effective batch. Where does the curve flatten?

Reading of the exercise: the lesson's own arguments are parsed with
`--num-steps 100` (so micro-batch 4, lr 0.05, seed 0, grid 1,2,4,8,16) and
`sweep_effective_batches` is run on that grid extended to accum 32 and 64,
so the flat part is visible past the lesson's default top of 64. Nothing is
written to `outputs/`. The "plot" is an ASCII bar per point. Throughput is
wall-clock, so the sweep is run 3 times and each point's rate is effective
batch / its best median step time (the median over 100 steps, and the best
of 3 sweeps, shrug off stalls from a loaded machine). The plateau is the
median rate at effective batch 64, 128 and 256, and "flattened" means within
10% of it. Timing is only asserted as a shape with wide margins.

**ANSWER: the curve flattens by effective batch 16-32 (accum 4-8), and it is
flat everywhere past 64.** At effective batch 4 the rate is 69-72% of the
plateau (30,000-37,000 samples/s here, over ten runs). Every step pays a fixed
cost: `zero_grads`, a `global_grad_norm` that calls `.item()` on each of the
6 parameters, and `optimizer.step()`. That cost is spread over more
micro-batches as accum grows. A fit of median step time = a + b * accum puts
the fixed cost at about half of one micro-batch's forward+backward
(a = 0.05-0.07 ms, b = 0.105 ms). By accum 4-8 the fixed cost is 6-12% of
the step, so the rate sits near the forward+backward ceiling of 4 / b
samples per ms. Past that point doubling accum only doubles the wall time
per step, which is the lesson's own point.

**FINDING: the loss curve has nothing to smooth.** `synthetic_batch` draws
the targets independently of the inputs, so no setting can learn anything.
The average loss at every one of the 7 points, identical over the 3
repeats, is within 0.01 of
ln 16 = 2.7726, the loss of a uniform guess over 16 classes. The lesson's
"smooth loss" at large accum cannot show up in this sweep.

**FINDING: the shipped `outputs/accum-curve.json` did not come from the
documented command.** It holds 3 points (accum 1, 2, 4) at 8 steps each.
`python3 code/main.py` with its defaults writes 5 points (accum 1-16) at
25 steps each.

Expected output: three PASS checks and a 7-row samples-per-second table.
"""

from __future__ import annotations

import json
import math
import sys

from harness import parity, practice

try:
    import torch  # noqa: F401  (the lesson's code needs it)
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "46-gradient-accumulation"
GRID, REPEATS = (1, 2, 4, 8, 16, 32, 64), 3


def lesson_args(ref, argv):
    saved, sys.argv = sys.argv, ["main.py", *argv]
    try:
        return ref.parse_args()
    finally:
        sys.argv = saved


def fit(points):
    """Least squares median_step_ms = a + b * accum."""
    xs = [p.accum_steps for p in points]
    ys = [p.median_step_ms for p in points]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def best_of(ref, args, grid):
    """REPEATS sweeps; per point, the one with the fastest median step."""
    runs = [
        ref.sweep_effective_batches(
            micro_batch=args.micro_batch, accum_grid=grid, num_steps=args.num_steps, lr=args.lr, seed=args.seed
        )
        for _ in range(REPEATS)
    ]
    points = [min(same, key=lambda p: p.median_step_ms) for same in zip(*runs)]
    return points, len({tuple(p.avg_loss for p in run) for run in runs}) == 1


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    args = lesson_args(ref, ["--num-steps", "100"])
    grid = sorted({*(int(s) for s in args.accum_grid.split(",")), *GRID})
    points, repeatable = best_of(ref, args, grid)
    rate = {p.effective_batch: p.effective_batch / p.median_step_ms * 1000 for p in points}
    best = sorted([rate[64], rate[128], rate[256]])[1]  # the plateau: median of the top of the grid
    shipped = json.loads((parity.lesson_dir(PHASE, LESSON) / "outputs" / "accum-curve.json").read_text())
    default = lesson_args(ref, [])
    return {
        "args": (args.micro_batch, args.num_steps, args.lr, args.seed, args.accum_grid),
        "rate": rate, "best": best, "fit": fit(points),
        "flat_at": min(e for e, r in rate.items() if r >= 0.9 * best),
        "counts": {(p.steps, p.sync_calls) for p in points},
        "losses": [p.avg_loss for p in points],
        "repeatable": repeatable,
        "shipped": [(p["accum_steps"], p["steps"]) for p in shipped["points"]],
        "default": (default.accum_grid, default.num_steps),
    }


def plot(result):
    return "\n".join(
        f"    eff {e:>4}  {r:>8.0f}/s  " + "#" * round(40 * r / result["best"]) for e, r in result["rate"].items()
    )


def flattens(r, slope):
    """Shape only, wide margins: 69-72% measured vs < 85%, flat from 16-32 vs <= 64."""
    rate, best = r["rate"], r["best"]
    setup = r["args"] == (4, 100, 0.05, 0, "1,2,4,8,16") and r["counts"] == {(100, 100)}
    shape = rate[4] < 0.85 * best and r["flat_at"] <= 64
    return setup and list(rate) == [4 * g for g in GRID] and shape and slope > 0


def verify(result):
    r, rate = result, result["rate"]
    a, b = r["fit"]
    print(plot(r))
    return [
        practice.Check(
            "ANSWER: the curve flattens by effective batch 16-32 (accum 4-8) and is flat past 64",
            flattens(r, b),
            f"flat (>= 90% of the plateau {r['best']:.0f}/s) from eff {r['flat_at']}; eff 4 at "
            f"{rate[4] / r['best']:.0%}; fixed cost {a:.3f} ms = {a / b:.1f} micro-batches of {b:.3f} ms",
        ),
        practice.Check(
            "FINDING: the loss curve has nothing to smooth -- every point sits at ln 16",
            r["repeatable"] and all(abs(x - math.log(16)) < 0.01 for x in r["losses"]),
            f"avg loss {[round(x, 4) for x in r['losses']]} vs ln 16 = {math.log(16):.4f}",
        ),
        practice.Check(
            "FINDING: the shipped accum-curve.json did not come from the documented command",
            r["shipped"] == [(1, 8), (2, 8), (4, 8)] and r["default"] == ("1,2,4,8,16", 25),
            f"shipped (accum, steps) {r['shipped']}; defaults grid {r['default'][0]} at {r['default'][1]} steps",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
