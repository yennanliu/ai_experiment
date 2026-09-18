"""Exercise 3 — the floor is doing the balancing, not the cosine.

    Cosine schedule vs linear schedule: trace the number of unmasked tokens per
    step for T=8. Which is more balanced?

Reading of the exercise: both schedules are run through the lesson's own
integer arithmetic -- `max(1, int(remaining * keep_ratio))` against a shrinking
pool -- rather than being compared as continuous curves, because the rounding
and the `max(0.15, ...)` floor change the answer and both live in `sample`.

**ANSWER: cosine, at a standard deviation of 0.707 against 1.030.** Cosine
unmasks **2, 2, 2, 2, 3, 3, 1, 1** and linear **2, 3, 4, 3, 2, 1, 1**. Both
finish all 16 tokens; cosine uses all 8 steps and linear finishes in **7**,
having spent a quarter of its budget on one step.

**FINDING: the first two cosine steps are the floor, not the cosine.** Raw
`1 - cos` gives keep ratios of **0.0192** and **0.0761**, both below the
`max(0.15, ...)` floor in `sample`. Without it those steps would unmask
`int(16 * 0.0192) = 0` tokens, and `max(1, ...)` would make it one -- so the
schedule the exercise is comparing does not control its own opening.

**FINDING: linear is front-loaded exactly where the model knows least.** Its
peak is step 2, with **4** tokens unmasked while 11 are still hidden; cosine's
peak is step 4, with 3 unmasked and 8 hidden. The schedule that commits hardest
does so with the least context, which is the argument for cosine and is visible
in the traces rather than in the curves.

**FINDING: the trailing while-loop never fires for either.** Both schedules
reach a keep ratio of 1.0 at step 7 and clear the pool, so `sample`'s
`while any(t == MASK ...)` block is unreachable at T=8 -- it exists for a
schedule that does not end at 1.0.

Structure: `keep_ratios` builds each schedule through the lesson's own floor,
`trace` applies its integer arithmetic to a shrinking pool, and `spread` is the
balance measure.
"""

from __future__ import annotations

import statistics

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "14-show-o-discrete-diffusion-unified"
STEPS, LENGTH, FLOOR = 8, 16, 0.15


def keep_ratios(ref, kind, steps=STEPS):
    """The lesson's own keep ratio, including its max(FLOOR, ...) clamp."""
    if kind == "cosine":
        raw = [1 - ratio for ratio in ref.cosine_schedule(steps)]
    else:
        raw = [(t + 1) / steps for t in range(steps)]
    return raw, [max(FLOOR, value) for value in raw]


def trace(ratios, length=LENGTH):
    """sample()'s integer arithmetic: max(1, int(remaining * ratio)) per step."""
    remaining, per_step = length, []
    for ratio in ratios:
        if remaining == 0:
            break
        taken = min(remaining, max(1, int(remaining * ratio)))
        per_step.append(taken)
        remaining -= taken
    return per_step, remaining


def spread(per_step):
    return round(statistics.pstdev(per_step), 3)


def peak(per_step, length=LENGTH):
    """Which step unmasks most, how many, and how many were still hidden then."""
    where = per_step.index(max(per_step))
    return where, max(per_step), length - sum(per_step[:where])


def summarise(traces, key):
    return {kind: key(row) for kind, row in traces.items()}


def clamped_of(schedules):
    return {kind: [round(v, 4) for v in clamped]
            for kind, (_, clamped) in schedules.items()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    schedules = {kind: keep_ratios(ref, kind) for kind in ("cosine", "linear")}
    traces = {kind: trace(clamped) for kind, (_, clamped) in schedules.items()}
    peaks = {kind: peak(row[0]) for kind, row in traces.items()}
    clamped = clamped_of(schedules)
    raw_cosine = [round(v, 4) for v in schedules["cosine"][0]]
    return {
        "keep": clamped,
        "per_step": summarise(traces, lambda row: row[0]),
        "left": summarise(traces, lambda row: row[1]),
        "steps_used": summarise(traces, lambda row: len(row[0])),
        "spread": summarise(traces, lambda row: spread(row[0])),
        "balanced": min(traces, key=lambda kind: spread(traces[kind][0])),
        "raw_cosine": raw_cosine,
        "floored_steps": [i for i, value in enumerate(raw_cosine) if value < FLOOR],
        "unfloored_first": int(LENGTH * raw_cosine[0]),
        "peaks": summarise(peaks, lambda row: row[0]),
        "peak_size": summarise(peaks, lambda row: row[1]),
        "hidden_at_peak": summarise(peaks, lambda row: row[2]),
        "final_ratio": summarise(clamped, lambda row: round(row[-1], 12)),
    }


def verify(result):
    per_step, spreads = result["per_step"], result["spread"]
    return [
        practice.Check(
            "ANSWER: cosine, at a standard deviation of 0.707 against 1.030",
            all([per_step["cosine"] == [2, 2, 2, 2, 3, 3, 1, 1],
                 per_step["linear"] == [2, 3, 4, 3, 2, 1, 1],
                 spreads == {"cosine": 0.707, "linear": 1.03},
                 result["balanced"] == "cosine",
                 result["steps_used"] == {"cosine": 8, "linear": 7}]),
            f"cosine unmasks {per_step['cosine']} and linear {per_step['linear']} -- "
            f"standard deviations {spreads}. Both clear all {LENGTH} tokens; cosine uses all "
            f"{result['steps_used']['cosine']} steps and linear finishes in "
            f"{result['steps_used']['linear']}, having spent a quarter of its budget on one "
            "step",
        ),
        practice.Check(
            "FINDING: the first two cosine steps are the floor, not the cosine",
            all([result["raw_cosine"][:2] == [0.0192, 0.0761],
                 result["floored_steps"] == [0, 1],
                 result["unfloored_first"] == 0]),
            f"raw 1 - cos gives keep ratios {result['raw_cosine'][:3]}..., of which steps "
            f"{result['floored_steps']} sit below the {FLOOR} floor in sample(). Without it "
            f"the first step would unmask int({LENGTH} x {result['raw_cosine'][0]}) = "
            f"{result['unfloored_first']} tokens and max(1, ...) would make it one -- so the "
            "schedule being compared does not control its own opening",
        ),
        practice.Check(
            "FINDING: linear is front-loaded exactly where the model knows least",
            all([result["peaks"] == {"cosine": 4, "linear": 2},
                 result["peak_size"] == {"cosine": 3, "linear": 4},
                 result["hidden_at_peak"] == {"cosine": 8, "linear": 11}]),
            f"linear's peak is step {result['peaks']['linear']}, unmasking "
            f"{result['peak_size']['linear']} tokens with "
            f"{result['hidden_at_peak']['linear']} still hidden; cosine's is step "
            f"{result['peaks']['cosine']}, {result['peak_size']['cosine']} tokens with "
            f"{result['hidden_at_peak']['cosine']} hidden. The schedule that commits hardest "
            "does so with the least context",
        ),
        practice.Check(
            "FINDING: the trailing while-loop never fires for either",
            all([result["left"] == {"cosine": 0, "linear": 0},
                 all(value >= 1.0 - 1e-12
                     for value in result["final_ratio"].values())]),
            f"both schedules reach a final keep ratio of {result['final_ratio']} and clear "
            f"the pool ({result['left']} masked left), so sample()'s while any(t == MASK) "
            f"block is unreachable at T={STEPS}. It exists for a schedule that does not end "
            "at 1.0 -- and cosine only just does, because cos(pi/2) is 6.1e-17 rather than "
            "zero",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
