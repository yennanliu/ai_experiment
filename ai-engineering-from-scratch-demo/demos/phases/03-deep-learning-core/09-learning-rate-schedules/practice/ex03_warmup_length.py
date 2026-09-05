"""Exercise 3 — warmup length swept over 0, 1, 5, 10 and 20 percent of the run.

    Train with warmup + cosine but vary the warmup length: 0%, 1%, 5%, 10%, 20%
    of total steps. Find the sweet spot where training is most stable.

Reading of the exercise: "most stable" has to be measured, not eyeballed, so
check 2 counts every epoch-over-epoch loss increase across the grid — the thing a
sweet spot would trade against. Check 3 extends the grid past 20% to see whether
the winner is a sweet spot or an edge, check 4 prices what warmup costs, and
check 5 asks whether this trainer can fail the way warmup prevents.
"""

from __future__ import annotations

import inspect
from operator import itemgetter

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "09-learning-rate-schedules"
EPOCHS, N, PEAK, LR_MIN = 300, 200, 0.05, 1e-5
TOTAL = EPOCHS * N
FRACS, EXTRA, SEEDS = (0.0, 0.01, 0.05, 0.10, 0.20), (0.50, 0.75), (42, 7, 99)
ADAPTIVE = ("momentum", "velocity", "beta", "m_hat", "v_hat", "exp_avg", "adam")


def one_run(ref, data, frac):        # end loss, monotonicity and time-to-target
    losses = ref.train_with_schedule(ref.warmup_cosine_schedule, "wc", data, epochs=EPOCHS,
                                     base_lr=PEAK, lr_min=LR_MIN,
                                     warmup_steps=int(TOTAL * frac))
    return {"end": losses[-1],
            "ups": sum(1 for i in range(1, EPOCHS) if losses[i] > losses[i - 1]),
            "reach": next((i for i, v in enumerate(losses) if v < 0.05), EPOCHS)}


def budgets(ref):
    """Total learning rate spent, per warmup fraction."""
    wc = lambda s, f: ref.warmup_cosine_schedule(  # noqa: E731
        s, lr=PEAK, total_steps=TOTAL, lr_min=LR_MIN, warmup_steps=int(TOTAL * f))
    return [sum(wc(s, f) for s in range(TOTAL)) for f in FRACS]


def cosine_gap(ref):
    """How far warmup_steps = 0 is from plain cosine annealing."""
    kw = {"lr": PEAK, "total_steps": TOTAL, "lr_min": LR_MIN}
    return max(abs(ref.warmup_cosine_schedule(s, warmup_steps=0, **kw)
                   - ref.cosine_schedule(s, **kw)) for s in range(0, TOTAL, 7))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    grid = FRACS + EXTRA
    runs = {seed: [one_run(ref, ref.make_circle_data(seed=seed), f) for f in grid]
            for seed in SEEDS}
    source = inspect.getsource(ref.train_with_schedule)
    return {
        "grid": grid, "runs": runs, "budgets": budgets(ref), "gap": cosine_gap(ref),
        "adaptive": [w for w in ADAPTIVE if w in source],
        "update": next(ln.strip() for ln in source.splitlines() if "w2[i] -=" in ln),
    }


def digest(runs, grid):        # everything the checks quote, so `verify` stays inside D14
    ends = {s: list(map(itemgetter("end"), runs[s])) for s in SEEDS}
    pick = lambda s, upto: min(range(upto), key=lambda i: ends[s][i])  # noqa: E731
    delays = [list(map(itemgetter("reach"), runs[s][:len(FRACS)])) for s in SEEDS]
    return {"ends": ends, "delays": delays, "reach": delays[0],
            "narrow": list(map(lambda s: pick(s, len(FRACS)), SEEDS)),
            "wide": list(map(lambda s: pick(s, len(grid)), SEEDS)),
            "ups": sum(map(itemgetter("ups"), [r for s in SEEDS for r in runs[s][:len(FRACS)]])),
            "spread": max(map(lambda s: max(ends[s][:5]) / min(ends[s][:5]), SEEDS))}


def verify(result):
    grid, spend, d = result["grid"], result["budgets"], digest(result["runs"], result["grid"])
    ends, narrow, wide, ups = d["ends"], d["narrow"], d["wide"], d["ups"]
    spread, reach = d["spread"], d["reach"]
    return [
        practice.Check("ANSWER: 20% wins the named grid, and the whole grid is within 6%",
                       set(narrow) == {len(FRACS) - 1} and spread < 1.06,
                       "end loss at 0/1/5/10/20% warmup — "
                       + " | ".join(map(lambda s: f"seed {s}: "
                                        + ", ".join(map("{:.6f}".format, ends[s][:5])), SEEDS))
                       + f". 20% is best at all three seeds, but the worst-to-best "
                       f"ratio is {spread:.3f}. (The 0% row is plain cosine: "
                       f"warmup_steps = 0 matches `cosine_schedule` to {result['gap']:.1e})"),
        practice.Check("FINDING: no stability to trade against, and no spend to trade either",
                       ups == 0 and max(spend) / min(spend) < 1.001,
                       f"{ups} epoch-over-epoch loss increases across all {len(FRACS) * len(SEEDS)} "
                       f"grid runs ({len(FRACS) * len(SEEDS) * (EPOCHS - 1)} transitions) — every "
                       f"curve is monotone non-increasing, so 'most stable' cannot separate the "
                       f"five. MECHANISM: the integral of lr at 0/1/5/10/20% is "
                       + ", ".join(map("{:.1f}".format, spend))
                       + f", a range of {(max(spend) / min(spend) - 1) * 100:.3f}%: the ramp "
                       f"gives back half the peak rate over W steps but compresses cosine"),
        practice.Check("FINDING: 20% wins only because the grid stops there",
                       len(set(wide)) > 1,
                       "extending to 50% and 75%, the best fraction per seed becomes "
                       + ", ".join(map(lambda si: f"seed {si[0]}: {grid[si[1]] * 100:.0f}% "
                                       f"({ends[si[0]][si[1]]:.6f})", zip(SEEDS, wide)))
                       + ". Three seeds, three winners, all within 1%: the ranking is noise"),
        practice.Check("…what warmup does buy is measurable, and it is a delay",
                       all(map(lambda x: x == sorted(x), d["delays"])) and reach[-1] > 2 * reach[0],
                       f"epochs to reach loss < 0.05 at seed {SEEDS[0]}, by warmup fraction: "
                       + ", ".join(map(lambda p: f"{p[0] * 100:.0f}%: {p[1]}", zip(FRACS, reach)))
                       + ". Monotone in warmup length at every seed, and the end loss is flat: "
                       "the whole measurable effect is that training happens later"),
        practice.Check("MECHANISM: the trainer has no optimizer state for warmup to stabilise",
                       not result["adaptive"],
                       f"the lesson motivates warmup by Adam's zero-initialised moment estimates, "
                       f"but `train_with_schedule` is plain SGD: its whole update is "
                       f"`{result['update']}`, and the source mentions none of {ADAPTIVE}. Nothing "
                       f"carries between steps, so no running statistic can be wrong at step 0"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
