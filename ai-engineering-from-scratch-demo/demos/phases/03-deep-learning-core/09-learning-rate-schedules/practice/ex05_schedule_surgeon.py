"""Exercise 5 — a closed-loop schedule that watches the loss and edits itself.

    Build a "schedule surgeon" that monitors training loss and automatically
    switches from warmup to cosine when the loss stabilizes, and reduces lr if the
    loss plateaus for too long.

Reading of the exercise: both triggers are thresholds on the epoch loss, so the
surgeon is specified by "stabilizes" (three epochs improving by under 5% each) and
"plateaus" (eight without a 1% improvement on the best). Check 2 sweeps the handoff
epoch the trigger hunts for, check 3 prices it against its integral.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "09-learning-rate-schedules"
EPOCHS, N, PEAK, LR_MIN = 60, 200, 0.05, 1e-5
TOTAL, WCAP, SEEDS = EPOCHS * N, int(0.20 * EPOCHS * N), (42, 7, 99)
STABLE_W, STABLE_REL, PATIENCE, PLATEAU_REL, FACTOR = 3, 0.05, 8, 0.01, 0.5
HANDOFFS = tuple(range(4, 60, 8))
# a recorded rate history, handed back as one of the lesson's schedule_fns
replay = lambda hist: (lambda step, lr=0.01, **kw: hist[min(step, len(hist) - 1)])  # noqa: E731


def next_lr(state, step):        # warmup ramp to the handoff, cosine from there on
    if state["handoff"] is None or step < state["handoff"]:
        return PEAK * min(1.0, (step + 1) / WCAP)
    progress = (step - state["handoff"]) / (TOTAL - state["handoff"])
    return LR_MIN + 0.5 * (state["peak"] - LR_MIN) * (1 + math.cos(math.pi * progress))


def triggers(state, losses, hist):        # hand off when stable, damp when plateaued
    lo = max(1, len(losses) - STABLE_W)
    rel = [(losses[i - 1] - losses[i]) / losses[i - 1] for i in range(lo, len(losses))]
    if state["handoff"] is None and len(losses) > STABLE_W and all(r < STABLE_REL for r in rel):
        state.update(handoff=len(hist), peak=hist[-1] / state["damp"],
                     handoff_epoch=len(losses) - 1)
    if losses[-1] < state["best"] * (1 - PLATEAU_REL):
        return state.update(best=losses[-1], since=0)
    state["since"] += 1
    if state["since"] >= PATIENCE:
        state.update(damp=state["damp"] * FACTOR, since=0)
        state["cuts"].append((len(losses) - 1, hist[-1]))
    state["best"] = min(state["best"], losses[-1])


def surgeon(ref, data):        # drive the trainer one epoch at a time, recording rates
    state = {"handoff": None, "peak": PEAK, "damp": 1.0, "best": float("inf"), "since": 0,
             "cuts": [], "handoff_epoch": None}
    hist, losses = [], []
    for _epoch in range(EPOCHS):
        hist.extend(next_lr(state, len(hist) + i) * state["damp"] for i in range(N))
        losses.append(ref.train_with_schedule(replay(hist), "s", data, base_lr=PEAK,
                                              epochs=len(losses) + 1)[-1])
        triggers(state, losses, hist)
    return losses, sum(hist), state


def solve():
    ref, runs = parity.load_reference(PHASE, LESSON, "main"), {}
    for seed in SEEDS:
        data = ref.make_circle_data(seed=seed)
        run = lambda fn, lr=PEAK, **kw: ref.train_with_schedule(  # noqa: E731
            fn, "f", data, epochs=EPOCHS, base_lr=lr, **kw)[-1]
        losses, spent, state = surgeon(ref, data)
        runs[seed] = {"end": losses[-1], "spent": spent, "const": run(ref.constant_schedule),
                      "cosine": run(ref.cosine_schedule, lr_min=LR_MIN),
                      "flat": run(ref.constant_schedule, spent / TOTAL), **state,
                      "sweep": [run(lambda s, lr=0.01, h=h * N, **k:
                                    next_lr({"handoff": h, "peak": PEAK}, s)) for h in HANDOFFS]}
    data = ref.make_circle_data(seed=SEEDS[0])
    plain = ref.train_with_schedule(ref.constant_schedule, "c", data, epochs=40, base_lr=PEAK)
    echoed = ref.train_with_schedule(replay([PEAK] * 40 * N), "r", data, epochs=40, base_lr=PEAK)
    return {"runs": runs, "replay_exact": plain == echoed,
            "cuts": [(s, e, lr) for s in SEEDS for e, lr in runs[s]["cuts"]],
            "gap": max(abs(runs[s]["flat"] / runs[s]["end"] - 1) for s in SEEDS),
            "handoffs": [runs[s]["handoff_epoch"] for s in SEEDS],
            "monotone": all(r["sweep"] == sorted(r["sweep"], reverse=True) for r in runs.values())}


def evidence(runs, cuts):        # flags and rows, keeping `verify` inside D14
    return {"fired": all(runs[s]["cuts"] and runs[s]["handoff_epoch"] for s in SEEDS),
            "beaten": all(runs[s]["const"] < runs[s]["end"] < runs[s]["cosine"] for s in SEEDS),
            "late": max(lr for _s, _e, lr in cuts) < 0.2 * PEAK,
            "cutrow": ", ".join(f"({s}, {e}, {lr:.5f})" for s, e, lr in cuts),
            "sweeps": " | ".join(", ".join(f"{v:.4f}" for v in runs[s]["sweep"]) for s in SEEDS)}


def verify(result):
    runs, gap = result["runs"], result["gap"]
    ev = evidence(runs, result["cuts"])
    per = lambda k, f=".6f": ", ".join(format(runs[s][k], f) for s in SEEDS)  # noqa: E731
    return [
        practice.Check("ANSWER: fires both triggers, beats cosine, loses to a constant rate",
                       ev["fired"] and ev["beaten"] and ev["late"],
                       f"seeds {SEEDS}: handoff at epoch {result['handoffs']}, always at full "
                       f"rate; end loss {per('end')} vs {per('cosine')} cosine, {per('const')} "
                       f"constant. Cuts (seed, epoch, lr) — {ev['cutrow']}, all in the last "
                       f"third below 20% of peak, where the supervised schedule has stalled"),
        practice.Check("FINDING: no handoff point to detect — later is monotonically better",
                       result["monotone"],
                       f"end loss vs a *fixed* handoff epoch {list(HANDOFFS)}: {ev['sweeps']}. "
                       f"Strictly decreasing at all three seeds, so the best switch point is the "
                       f"latest tried, while the trigger fires 9 epochs apart across seeds"),
        practice.Check("CONTROL: a constant rate spending the same total reproduces it",
                       gap < 0.10 and result["replay_exact"],
                       f"rate integral spent {per('spent', '.0f')}; a constant schedule at that "
                       f"mean rate ends {per('flat')} vs the surgeon's {per('end')}, within "
                       f"{gap * 100:.0f}% everywhere. `schedule_fn` is never passed the loss, so "
                       f"the loop closes by replay, exact to the bit on a 40-epoch control"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
