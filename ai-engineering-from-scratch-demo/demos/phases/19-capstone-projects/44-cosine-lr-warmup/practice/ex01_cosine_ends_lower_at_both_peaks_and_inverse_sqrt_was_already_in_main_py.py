"""Exercise 1 -- cosine ends lower in both regimes, and inverse-sqrt was already in main.py.

    Add an inverse-square-root variant of the schedule and compare it on a
    200-step toy training run. Which curve produces the lower final loss?

Reading of the exercise: the toy run is the lesson's own: `build_toy_model`
(seed 7, a 16-32-4 MLP on one fixed batch of 8), driven by `TrainState` with
AdamW and MSE for 200 steps. Both schedules use the lesson's demo peak and
floor ratio (lr_min = lr_max / 100) with a 20-step warmup. The inverse-sqrt
curve is the lesson's `InverseSqrtWarmup`: linear warmup to lr_max, then
lr_max * sqrt(warmup / step). "Final loss" is the MSE on the batch after the
200th update. The run is repeated at lr_max = 1e-2, the demo's peak, and at
1e-3, because at 1e-2 the model memorises the batch.

**ANSWER: cosine gives the lower final loss at both peaks.** At lr_max 1e-3
it ends at 0.0956 and inverse-sqrt at 0.1062. At 1e-2 both memorise the
batch: cosine ends at about 5e-10 and inverse-sqrt at about 1e-6.

**FINDING: the variant already exists.** `main.py` ships
`InverseSqrtWarmup`, and `code/tests/test_schedule.py` already tests it.
It takes no `total_steps` and no `lr_min`, so at step 199 it is still at
31.7% of peak, while cosine is at 1.0%.

**FINDING: the answer depends on when the run stops.** The two logs are
bit-identical for steps 0-20, because the warmups match. At 1e-2,
inverse-sqrt has the lower logged loss on 66 of the 200 steps, all between
steps 26 and 145. Cosine overtakes once it anneals. At 1e-3 inverse-sqrt
never leads. A final-loss comparison measures where each curve ends more
than what shape it has.

Structure: `run()` trains the toy model under one schedule; `race()` runs
both schedules at one peak.
"""

from __future__ import annotations

from harness import parity, practice

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None
torch.set_num_threads(1)

PHASE, LESSON = "19-capstone-projects", "44-cosine-lr-warmup"
STEPS, WARMUP = 200, 20


def run(ref, schedule):
    model, x, y = ref.build_toy_model()
    state = ref.TrainState(model, schedule, nn.functional.mse_loss)
    for _ in range(STEPS):
        state.step(x, y)
    with torch.no_grad():
        final = float(nn.functional.mse_loss(model(x), y))
    return state.log, final


def race(ref, lr_max):
    cos_log, cos_final = run(ref, ref.CosineWithWarmup(WARMUP, STEPS, lr_max, lr_max / 100))
    isq_log, isq_final = run(ref, ref.InverseSqrtWarmup(WARMUP, lr_max))
    pairs = list(zip(cos_log, isq_log))
    return {
        "final": [cos_final, isq_final],
        "same_prefix": next(k for k, (a, b) in enumerate(pairs) if (a.loss, a.lr) != (b.loss, b.lr)),
        "isq_leads": [k for k, (a, b) in enumerate(pairs) if b.loss < a.loss],
        "last_lr": [cos_log[-1].lr / lr_max, isq_log[-1].lr / lr_max],
        "final4": [round(cos_final, 4), round(isq_final, 4)],
        "last_lr3": [round(cos_log[-1].lr / lr_max, 3), round(isq_log[-1].lr / lr_max, 3)],
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    tests = (parity.lesson_dir(PHASE, LESSON) / "code" / "tests" / "test_schedule.py").read_text()
    return {
        "hi": race(ref, 1e-2), "lo": race(ref, 1e-3),
        "shipped": hasattr(ref, "InverseSqrtWarmup"),
        "tested": "InverseSqrtWarmup(warmup_steps=4" in tests,
        "isq_fields": sorted(f.name for f in ref.dataclasses.fields(ref.InverseSqrtWarmup)),
    }


def verify(result):
    hi, lo = result["hi"], result["lo"]
    leads = hi["isq_leads"]
    return [
        practice.Check(
            "ANSWER: cosine gives the lower final loss at both peaks",
            (lo["final4"], hi["final"][0] < 1e-8 < hi["final"][1] < 1e-5) == ([0.0956, 0.1062], True),
            f"lr_max 1e-3: cosine {lo['final'][0]:.4f} vs inverse-sqrt {lo['final'][1]:.4f}; "
            f"lr_max 1e-2: {hi['final'][0]:.1e} vs {hi['final'][1]:.1e}",
        ),
        practice.Check(
            "FINDING: the variant already exists in main.py and its tests",
            (result["shipped"], result["tested"], result["isq_fields"], hi["last_lr3"])
            == (True, True, ["lr_max", "warmup_steps"], [0.01, 0.317]),
            f"InverseSqrtWarmup fields {result['isq_fields']}; LR at step 199 as a share of "
            f"peak: cosine {hi['last_lr'][0]:.1%}, inverse-sqrt {hi['last_lr'][1]:.1%}",
        ),
        practice.Check(
            "FINDING: the answer depends on when the run stops",
            (hi["same_prefix"], lo["same_prefix"], 55 <= len(leads) <= 75, leads[0] > WARMUP,
             leads[-1] < 160, lo["isq_leads"]) == (21, 21, True, True, True, []),
            f"logs identical for steps 0-{hi['same_prefix'] - 1}; at 1e-2 inverse-sqrt leads on "
            f"{len(leads)}/200 steps ({leads[0]}-{leads[-1]}), at 1e-3 on {len(lo['isq_leads'])}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
