"""Exercise 4 — SGDR, cosine annealing with warm restarts, on a longer run.

    Implement cosine annealing with warm restarts (SGDR): reset the learning rate
    to lr_max every T steps and decay again. Compare to standard cosine on a
    longer training run.

Reading of the exercise: a restart schedule is the lesson's `cosine_schedule`
called on `step % period`, so check 1 builds it that way and shows why the
comparison cannot say anything — restarts leave the integral untouched. Checks 3
and 4 let `step` run past `total_steps`, the other way a reader might make a
restart happen, which two of the three schedules do not survive.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "09-learning-rate-schedules"
EPOCHS, N, PEAK, LR_MIN = 600, 200, 0.05, 1e-5
TOTAL, HALF = EPOCHS * N, EPOCHS * N // 2
PERIODS, SEEDS = (5000, 20000), (42, 7, 99)


def sgdr(ref):
    """SGDR: the lesson's cosine, restarted every `period` steps."""
    def fn(step, lr=0.01, period=20000, lr_min=LR_MIN, **kwargs):
        return ref.cosine_schedule(step % period, lr=lr, total_steps=period, lr_min=lr_min)
    return fn


def spend(fn, **kwargs):        # integral of the schedule over the run
    return sum(fn(s, lr=PEAK, total_steps=TOTAL, **kwargs) for s in range(TOTAL))


def restart_rises(losses, period):
    """Post-restart loss peak over the loss just before each restart."""
    epochs = [period * k // N for k in range(1, TOTAL // period)]
    return [max(losses[e:e + 3]) / losses[e - 1] for e in epochs], epochs


def schedules(ref):
    """The lesson's three annealing schedules, with the kwargs each needs."""
    warm = {"warmup_steps": 6000, "lr_min": LR_MIN}
    return [("cosine", ref.cosine_schedule, {"lr_min": LR_MIN}),
            ("warmup_cosine", ref.warmup_cosine_schedule, warm),
            ("one_cycle", ref.one_cycle_schedule, {})]

def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    restart, ends, rises = sgdr(ref), {}, None
    run = lambda d, f, **k: ref.train_with_schedule(f, "x", d, epochs=EPOCHS,  # noqa: E731
                                                    base_lr=PEAK, **k)
    for seed in SEEDS:
        data = ref.make_circle_data(seed=seed)
        ends[seed] = {"cosine": run(data, ref.cosine_schedule, lr_min=LR_MIN)[-1]}
        for period in PERIODS:
            losses = run(data, restart, period=period)
            ends[seed][period] = losses[-1]
            if seed == SEEDS[0] and period == PERIODS[-1]:
                rises = restart_rises(losses, period)
    data = ref.make_circle_data(seed=SEEDS[0])
    deltas = list(map(lambda s: ends[s][PERIODS[-1]] / ends[s]["cosine"], SEEDS))
    # the same three schedules, driven past a stale total_steps
    stale = lambda fn, kw: (lambda s, lr=PEAK, **_i: fn(s, lr=lr, total_steps=HALF, **kw))
    return {"ends": ends, "rises": rises,
            "budgets": {"cosine": spend(ref.cosine_schedule, lr_min=LR_MIN),
                        **{p: spend(restart, period=p) for p in PERIODS}},
            "late": {n: run(data, stale(f, k))[EPOCHS // 2 - 1::EPOCHS // 2]
                     for n, f, k in schedules(ref)},
            "lr_late": {n: f(TOTAL - 1, lr=PEAK, total_steps=HALF, **k)
                        for n, f, k in schedules(ref)},
            "deltas": deltas, "agree": max(map(lambda d: abs(d - 1), deltas)) < 0.02}


def verify(result):
    ends = result["ends"]
    late, lr_late, (rises, epochs) = result["late"], result["lr_late"], result["rises"]
    deltas, budgets = result["deltas"], result["budgets"]
    return [
        practice.Check("ANSWER: over 600 epochs SGDR and cosine finish in the same place — "
                       "MECHANISM: restarts cannot change what a schedule spends",
                       result["agree"] and max(budgets.values()) / min(budgets.values()) < 1.001,
                       " | ".join(map(lambda s: f"seed {s}: cosine {ends[s]['cosine']:.6f}, "
                                      + ", ".join(map(lambda p: f"restart/{p} {ends[s][p]:.6f}",
                                                      PERIODS)), SEEDS))
                       + ". Integral of lr over the run — "
                       + ", ".join(map(lambda kv: f"{kv[0]}: {kv[1]:.1f}", budgets.items()))
                       + ": a cycle integrates to 0.5*(lr_max + lr_min)*T_i and the cycles sum "
                       "to the run length whatever the period"),
        practice.Check("FINDING: the residual flips sign with the seed; only the spike repeats",
                       min(deltas) < 1.0 < max(deltas) and min(rises) > 1.05,
                       "SGDR / cosine end-loss ratio at period 20000: "
                       + ", ".join(map(lambda sd: f"seed {sd[0]} {sd[1]:.4f}", zip(SEEDS, deltas)))
                       + f" — two win, one loses, all inside 1.6%. At restart epochs {epochs} the "
                       f"loss instead rises " + ", ".join(map(lambda r: f"{(r - 1) * 100:.1f}%", rises))
                       + f", the rate jumping {LR_MIN} -> {PEAK} in one step"),
        practice.Check("FINDING: the three schedules disagree about what happens past "
                       "total_steps, and one restarts by accident",
                       lr_late["cosine"] == LR_MIN and lr_late["warmup_cosine"] > 0.9 * PEAK,
                       f"driven to step {TOTAL - 1} with total_steps left at {HALF}: "
                       f"`cosine_schedule` clamps to {lr_late['cosine']:.6f} (guarding on `step "
                       f">= total_steps`); `warmup_cosine_schedule` has none, its cosine argument "
                       f"passes 2*pi, and it climbs back to "
                       f"{lr_late['warmup_cosine'] / PEAK * 100:.0f}% of peak — second-half loss "
                       + " -> ".join(map("{:.5f}".format, late["warmup_cosine"]))
                       + ": an unasked-for warm restart, and it helps"),
        practice.Check("CONTROL: the third runs the rate negative and undoes the training",
                       lr_late["one_cycle"] < -PEAK and late["one_cycle"][-1] > 0.5,
                       f"`one_cycle_schedule` computes lr*(1 - progress), and progress > 1 past "
                       f"total_steps gives {lr_late['one_cycle']:+.6f} — twice the peak rate, "
                       f"wrong sign, unbounded. Gradient ascent: loss at epochs 299, 599 "
                       + " -> ".join(map("{:.5f}".format, late["one_cycle"]))
                       + ", the net pushed onto the constant-1 predictor"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
