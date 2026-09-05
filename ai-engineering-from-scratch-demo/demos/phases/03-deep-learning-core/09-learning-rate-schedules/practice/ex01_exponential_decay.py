"""Exercise 1 — exponential decay against cosine annealing on the circle dataset.

    Implement exponential decay: lr(t) = lr_0 * gamma^t where gamma = 0.999.
    Compare to cosine annealing on the circle dataset.

Reading of the exercise: `t` is unqualified and the lesson's trainer counts
steps, so check 1 takes it literally (t = step) and check 3 takes the other
reading (t = epoch) — they give opposite verdicts, which is the answer. Checks 4
and 5 ask whether "compare" can mean anything on this fixture at all, by
matching total learning-rate spend across shapes and by re-running every seed.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "09-learning-rate-schedules"
GAMMA, EPOCHS, N, PEAK = 0.999, 300, 200, 0.05
TOTAL, COS = EPOCHS * N, {"lr_min": 1e-5}
SEEDS = (42, 7, 99)


def expo_step(step, lr=0.01, **kwargs):
    """t = the step counter the lesson's trainer passes — the literal reading."""
    return lr * GAMMA**step


def expo_epoch(step, lr=0.01, **kwargs):
    """t = one decay per epoch of the 200-point dataset — the other reading."""
    return lr * GAMMA ** (step // N)


def budget(fn, **kwargs):
    """Total learning rate spent over the run — the integral of the schedule."""
    return sum(fn(s, lr=PEAK, total_steps=TOTAL, **kwargs) for s in range(TOTAL))


def sweep(ref):
    """End loss per schedule per data seed, plus one full loss curve."""
    runs = {"expo_step": (expo_step, {}), "expo_epoch": (expo_epoch, {}),
            "cosine": (ref.cosine_schedule, COS), "constant": (ref.constant_schedule, {})}
    ends, curves = {name: [] for name in runs}, {}
    for seed in SEEDS:
        data = ref.make_circle_data(seed=seed)
        for name, (fn, kw) in runs.items():
            losses = ref.train_with_schedule(fn, name, data, epochs=EPOCHS, base_lr=PEAK, **kw)
            ends[name].append(losses[-1])
            curves.setdefault(name, losses)
    return ends, curves


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ends, curves = sweep(ref)
    data = ref.make_circle_data(seed=SEEDS[0])
    budgets = {"expo_step": budget(expo_step), "expo_epoch": budget(expo_epoch),
               "cosine": budget(ref.cosine_schedule, **COS),
               "constant": budget(ref.constant_schedule)}
    flat = lambda lr: ref.train_with_schedule(  # noqa: E731
        ref.constant_schedule, "flat", data, epochs=EPOCHS, base_lr=lr)[-1]
    flats = {k: (budgets[k] / TOTAL, flat(budgets[k] / TOTAL)) for k in ("cosine", "expo_epoch")}
    frozen = next(i for i, v in enumerate(curves["expo_step"]) if v == curves["expo_step"][-1])
    return {"ends": ends, "budgets": budgets, "flats": flats, "frozen": frozen,
            "half_life": math.log(0.5) / math.log(GAMMA),
            "lr_at_freeze": PEAK * GAMMA ** (frozen * N),
            "matched_gamma": math.exp(math.log(COS["lr_min"] / PEAK) / TOTAL)}


def pairs(ends, a, b):
    return ", ".join(f"{x:.6f} vs {y:.6f}" for x, y in zip(ends[a], ends[b]))


def verify(result):
    ends, flats, budgets = result["ends"], result["flats"], result["budgets"]
    ratio = [e / c for e, c in zip(ends["expo_step"], ends["cosine"])]
    shape = [flats[k][1] / ends[k][0] for k in ("cosine", "expo_epoch")]
    return [
        practice.Check("ANSWER: read literally (t = step), exponential decay loses to cosine "
                       "by an order of magnitude",
                       min(ratio) > 8.0,
                       f"end loss after 300 epochs at lr_0 = 0.05, seeds {SEEDS}: "
                       f"{pairs(ends, 'expo_step', 'cosine')} — {min(ratio):.1f}x worse at best"),
        practice.Check("FINDING: it does not converge slowly, it stops — and MECHANISM: gamma "
                       "fixes an absolute time constant where cosine fixes a relative one",
                       result["frozen"] < 200 and result["matched_gamma"] > 0.9998,
                       f"the epoch loss is bit-identical from epoch {result['frozen']} to 299: "
                       f"0.999^(200*{result['frozen']}) puts the rate at {result['lr_at_freeze']:.2e}, "
                       f"so every update rounds away. gamma halves the rate every "
                       f"{result['half_life']:.0f} steps whatever the run length, where cosine "
                       f"halves it at total_steps/2 = {TOTAL // 2}; reaching lr_min = 1e-05 over "
                       f"{TOTAL} steps needs gamma = {result['matched_gamma']:.8f}"),
        practice.Check("…and read as t = epoch the same schedule beats cosine, at every seed",
                       all(e < c for e, c in zip(ends["expo_epoch"], ends["cosine"])),
                       f"one decay per epoch: {pairs(ends, 'expo_epoch', 'cosine')}. The "
                       f"exercise's comparison has opposite answers under the two readings of t"),
        practice.Check("MECHANISM: the comparison is decided by total rate spent, not by shape",
                       max(shape) < 1.15,
                       "integral of lr over the run: "
                       + ", ".join(f"{k} {v:.0f}" for k, v in budgets.items())
                       + f". A *constant* schedule matched to cosine's integral (lr = "
                       f"{flats['cosine'][0]:.5f}) ends at {flats['cosine'][1]:.6f} vs cosine's "
                       f"{ends['cosine'][0]:.6f}; one matched to expo_epoch, "
                       f"{flats['expo_epoch'][1]:.6f} vs {ends['expo_epoch'][0]:.6f} — "
                       f"{(max(shape) - 1) * 100:.0f}% apart at most, while the 1.73x spread in "
                       f"integral moves the loss "
                       f"{(1 - ends['expo_epoch'][0] / ends['cosine'][0]) * 100:.0f}%"),
        practice.Check("CONTROL: a constant rate beats cosine on this fixture, at every seed",
                       all(k < c for k, c in zip(ends["constant"], ends["cosine"])),
                       f"constant lr = 0.05 vs cosine: {pairs(ends, 'constant', 'cosine')}. The "
                       f"circle task never needs the fine steps a decay buys, so the schedule that "
                       f"spends most wins — this comparison measures integral, not shape"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
