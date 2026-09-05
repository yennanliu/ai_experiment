"""Exercise 2 — Leslie Smith's learning rate range test on the circle dataset.

    Implement the learning rate range test (Leslie Smith): train for a few
    hundred steps while exponentially increasing the LR from 1e-7 to 1. Plot loss
    vs LR. The optimal max LR is just before the loss starts increasing.

Reading of the exercise: the last sentence is a claim, not an instruction, so the
test is worth running only if the turning point is looked for rather than assumed.
Check 1 runs the sweep as specified; check 2 widens it by four decades to find the
turning point the window misses; check 3 asks what "increasing" can mean for this
loss; check 4 swaps the continuous run for independent runs of the same budget.
"""

from __future__ import annotations

import contextlib
import io

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "09-learning-rate-schedules"
LO, N, BINS = 1e-7, 20, 25
STEPS = N * BINS
GRID = [10 ** (-7 + 0.5 * i) for i in range(25)]      # 1e-7 .. 1e5


def ramp(hi):
    """A schedule that sweeps lr geometrically from LO to `hi` across STEPS."""
    def fn(step, lr=0.01, **kwargs):
        return LO * (hi / LO) ** (min(step, STEPS - 1) / (STEPS - 1))
    return fn


def range_test(ref, hi):
    """One continuous run; each epoch's mean loss labelled by its mid-bin lr."""
    data = ref.make_circle_data(n=N, seed=42)
    losses = ref.train_with_schedule(ramp(hi), "range", data, epochs=BINS, base_lr=0.05)
    lrs = [LO * (hi / LO) ** ((b * N + N // 2) / (STEPS - 1)) for b in range(BINS)]
    rises = [b for b in range(1, BINS) if losses[b] > losses[b - 1]]  # bin-to-bin increases
    return {"lrs": lrs, "losses": losses, "rises": rises,
            "best": min(range(BINS), key=lambda b: losses[b])}


def independent(ref, data, epochs):
    """Fresh network per learning rate — the honest version of loss-vs-lr."""
    ends = [ref.train_with_schedule(ref.constant_schedule, "c", data, epochs=epochs,
                                    base_lr=lr)[-1] for lr in GRID]
    return {"ends": ends, "best": min(range(len(GRID)), key=lambda i: ends[i]), "worst": max(ends)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    data = ref.make_circle_data(seed=42)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ref.lr_sensitivity(data)
    return {
        "narrow": range_test(ref, 1.0), "wide": range_test(ref, 1e4),
        "short": independent(ref, data, 3), "long": independent(ref, data, 100),
        "sensitivity": buf.getvalue(),
        "positives": sum(1 for _x, t in data if t == 1.0) / len(data),
    }


def verify(result):
    narrow, wide = result["narrow"], result["wide"]
    short, long_ = result["short"], result["long"]
    pos = result["positives"]
    pinned = [f"{g:.0e}" for g, e in zip(GRID, short["ends"]) if e == pos]
    return [
        practice.Check("ANSWER: over 1e-7 to 1 the loss never starts increasing — no answer",
                       not narrow["rises"] and narrow["best"] == BINS - 1,
                       f"{BINS} bins over {STEPS} steps, 7 decades: loss falls monotonically from "
                       f"{narrow['losses'][0]:.6f} at lr = {narrow['lrs'][0]:.2e} to "
                       f"{narrow['losses'][-1]:.6f} at lr = {narrow['lrs'][-1]:.2e}, with "
                       f"{len(narrow['rises'])} rises. The minimum is the last bin, so the rule "
                       f"selects the top of whatever window you chose"),
        practice.Check("FINDING: the turning point exists, above the specified window",
                       wide["lrs"][wide["best"]] > 1.0 and bool(wide["rises"]),
                       f"the identical sweep widened to 1e+04 bottoms out at lr = "
                       f"{wide['lrs'][wide['best']]:.2e} (loss {wide['losses'][wide['best']]:.6f}) "
                       f"and first rises at {wide['lrs'][wide['rises'][0]]:.2e} — both above the "
                       f"stated ceiling of 1.0, so the window ends "
                       f"{wide['lrs'][wide['best']]:.1f}x short of the answer it should bracket"),
        practice.Check("MECHANISM: nothing here diverges — the upper edge is a collapse onto "
                       "the majority class, not a blow-up",
                       long_["worst"] < 1.0 and "DIVERGED" not in result["sensitivity"]
                       and len(pinned) >= 3,
                       f"MSE on a sigmoid output is bounded above by 1 by construction; the worst "
                       f"value over lr in [1e-07, 1e+05] is {long_['worst']:.6f}. The lesson's "
                       f"`lr_sensitivity` gates DIVERGED on `end > 1.0`, which no run can satisfy "
                       f"— it prints CONVERGED for lr = 1.0 though the text says lr = 0.1 makes "
                       f"'loss jump to infinity in 3 steps'. Instead, at lr >= 1e+01 the "
                       f"3-epoch loss pins at exactly {pos:.6f}, the positive-class fraction "
                       f"({pos * 200:.0f}/200), for bins {', '.join(pinned)} — the output "
                       f"saturates to 0, so the gradient vanishes rather than explodes"),
        practice.Check("CONTROL: with independent runs the curve turns inside that window",
                       0 < short["best"] < len(GRID) - 1,
                       f"a fresh network per rate, 3 epochs ({3 * 200} steps, the same order as "
                       f"the sweep): minimum {short['ends'][short['best']]:.6f} at lr = "
                       f"{GRID[short['best']]:.2e}, rising to "
                       f"{short['ends'][short['best'] + 1]:.6f} and "
                       f"{short['ends'][short['best'] + 2]:.6f} at the next two decades. "
                       f"MECHANISM: the range test evaluates high rates on a network the low "
                       f"ones already trained, so 'loss at lr' is confounded with 'loss after "
                       f"k steps'"),
        practice.Check("FINDING: the lr_max it reports depends on the budget it was run for",
                       GRID[long_["best"]] < GRID[short["best"]],
                       f"the same independent sweep at 100 epochs puts the optimum at lr = "
                       f"{GRID[long_['best']]:.2e} (loss {long_['ends'][long_['best']]:.6f}), "
                       f"{GRID[short['best']] / GRID[long_['best']]:.0f}x below the 3-epoch answer "
                       f"of {GRID[short['best']]:.2e}; the lesson then runs at 0.05, another 2x "
                       f"down. An lr_max is defined only alongside its step count"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
