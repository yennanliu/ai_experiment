"""Exercise 2 -- a restart helps the under-trained run only by adding 27% more LR, and hurts the memorised one.

    Add a `--restart` flag that adds a second warmup at `total_steps / 2`.
    Defend whether warm restarts improve or hurt on the toy run.

Reading of the exercise: `schedule_from_args` parses `--restart` (with
`--lr-max`, `--warmup` and `--total`). Without the flag it returns the
lesson's `CosineWithWarmup(20, 200, lr_max, lr_max / 100)`. With the flag
the first half is unchanged, and at step 100 a second `CosineWithWarmup(20,
100, ...)` cycle begins. That cycle warms up from zero to lr_max, which is
the lesson's definition of warmup, and then decays to lr_min at step 200.
The toy run is the lesson's `build_toy_model` batch, trained for 200
`TrainState` steps, at the demo peak 1e-2 and at 1e-3. Two controls
separate the restart from the budget: an SGDR-style schedule of two
100-step cycles, which keeps the LR area, and a no-restart cosine whose peak
is raised to match the restart's area.

**ANSWER: the restart helps the under-trained run and hurts the memorised
one.** At lr_max 1e-3 the final loss falls from 0.0956 to 0.0233. At 1e-2
the model has already memorised the batch, and the restart nearly doubles the
residual loss, from 5.2e-10 to 9.9e-10. The defence is that on this toy the
restart helps only because it adds learning rate.

**FINDING: most of the gain is the extra LR budget, not the restart.** The
added cycle raises the summed LR from 100.9 to 128.4 x lr_max, +27.3%. A
plain cosine with its peak raised by that factor reaches 0.0273 at 1e-3, so
it recovers 95% of the restart's gain. SGDR's two cycles keep the area at
100.8 and reach only 0.0915, 4% better than no restart.

**FINDING: the "warm" restart starts cold.** The second warmup ramps from 0,
so the LR at step 100 falls from 5.99e-3 to 0.0. When the restart follows a
full decay, as in SGDR at 1e-2, the loss climbs 46% within 10 steps, from
5.07e-5 to 7.41e-5. There is also nothing for a restart to escape to: one
fixed batch has no held-out set, which is where SGDR's claimed benefit lies.

Structure: `schedule_from_args()` is the flag; `restarted()` builds the schedule;
`run()` trains the toy model under any object with `lr(step)`.
"""

from __future__ import annotations

import argparse
import types

from harness import parity, practice

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None
torch.set_num_threads(1)

PHASE, LESSON = "19-capstone-projects", "44-cosine-lr-warmup"
REF = parity.load_reference(PHASE, LESSON, "main")


def restarted(first, second, at):
    """A schedule object: `first` until step `at`, then `second` from its own step 0."""
    return types.SimpleNamespace(lr=lambda step: first.lr(step) if step < at else second.lr(step - at))


def schedule_from_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--restart", action="store_true")
    p.add_argument("--lr-max", type=float, default=1e-2)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--total", type=int, default=200)
    a = p.parse_args(argv)
    base = REF.CosineWithWarmup(a.warmup, a.total, a.lr_max, a.lr_max / 100)
    half = a.total // 2
    cycle = REF.CosineWithWarmup(a.warmup, a.total - half, a.lr_max, a.lr_max / 100)
    return restarted(base, cycle, half) if a.restart else base


def run(schedule, steps=200):
    model, x, y = REF.build_toy_model()
    state = REF.TrainState(model, schedule, nn.functional.mse_loss)
    for _ in range(steps):
        state.step(x, y)
    with torch.no_grad():
        final = float(nn.functional.mse_loss(model(x), y))
    return state.log, final


def compare(lr_max):
    flag = ["--lr-max", str(lr_max)]
    base, restart = schedule_from_args(flag), schedule_from_args(flag + ["--restart"])
    area = [sum(s.lr(k) for k in range(200)) / lr_max for s in (base, restart)]
    k = area[1] / area[0]
    cycle = REF.CosineWithWarmup(20, 100, lr_max, lr_max / 100)
    runs = {"base": base, "restart": restart, "sgdr": restarted(cycle, cycle, 100),
            "matched": REF.CosineWithWarmup(20, 200, lr_max * k, lr_max * k / 100)}
    logs = {name: run(s) for name, s in runs.items()}
    sgdr = logs["sgdr"][0]
    return {"final": {n: v[1] for n, v in logs.items()}, "area": area + [sum(r.lr for r in sgdr) / lr_max],
            "lr_99_100": [restart.lr(99), restart.lr(100)],
            "sgdr_bump": [sgdr[100].loss, max(r.loss for r in sgdr[100:111])]}


def solve():
    return {"hi": compare(1e-2), "lo": compare(1e-3)}


def verify(result):
    hi, lo = result["hi"], result["lo"]
    f, g = lo["final"], hi["final"]
    recovered = (f["base"] - f["matched"]) / (f["base"] - f["restart"])
    bump = hi["sgdr_bump"][1] / hi["sgdr_bump"][0]
    return [
        practice.Check(
            "ANSWER: the restart helps the under-trained run and hurts the memorised one",
            [round(f["base"], 4), round(f["restart"], 4)] == [0.0956, 0.0233]
            and g["base"] < g["restart"] < 1e-8,
            f"lr_max 1e-3: {f['base']:.4f} -> {f['restart']:.4f}; lr_max 1e-2: "
            f"{g['base']:.1e} -> {g['restart']:.1e}",
        ),
        practice.Check(
            "FINDING: most of the gain is the extra LR budget, not the restart",
            [round(a, 1) for a in lo["area"]] == [100.9, 128.4, 100.8]
            and round(f["matched"], 4) == 0.0273 and round(f["sgdr"], 4) == 0.0915
            and 0.9 < recovered < 1.0,
            f"summed LR / lr_max: base {lo['area'][0]:.1f}, restart {lo['area'][1]:.1f}, SGDR "
            f"{lo['area'][2]:.1f}; area-matched cosine {f['matched']:.4f} recovers {recovered:.0%}; "
            f"SGDR {f['sgdr']:.4f}",
        ),
        practice.Check(
            "FINDING: the warm restart starts cold",
            round(hi["lr_99_100"][0], 5) == 0.00599 and hi["lr_99_100"][1] == 0.0 and bump > 1.3,
            f"LR at steps 99/100: {hi['lr_99_100'][0]:.2e} -> {hi['lr_99_100'][1]}; SGDR at 1e-2: "
            f"loss {hi['sgdr_bump'][0]:.2e} -> {hi['sgdr_bump'][1]:.2e} (+{bump - 1:.0%}) by step 110",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
