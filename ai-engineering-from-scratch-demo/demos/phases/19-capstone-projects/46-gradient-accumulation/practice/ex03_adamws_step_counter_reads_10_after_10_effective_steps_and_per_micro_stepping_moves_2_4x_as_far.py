"""Exercise 3 -- AdamW's step counter reads 10 after 10 effective steps at accum 1, 4 and 16; stepping per micro-batch moves 2.4x as far.

    Swap SGD for AdamW and confirm the optimizer state advances once per effective step, not once per micro-batch.

Reading of the exercise: `run_config` hard-codes `torch.optim.SGD` and takes
no optimizer argument, so the swap is done where the lesson builds it: for
the length of the run, `torch.optim.SGD` is replaced by a factory that
returns `torch.optim.AdamW(params, lr=lr)` and keeps a handle on it. The
lesson's own sweep then runs unchanged: micro-batch 4, accum 1, 4 and 16,
10 optimizer steps each. "Optimizer state" means AdamW's per-parameter
`step` counter, which also drives its bias correction. A second check
confirms the state is also the right state: 5 accumulated AdamW steps (4
chunks of 4) are compared with 5 full-batch AdamW steps on the same
16-sample batches. The counterfactual steps once per micro-batch.

**ANSWER: the state advances once per effective step.** After 10 effective
steps the `step` counter of all 6 parameters reads 10 at accum 1, 4 and 16,
while the model saw 10, 40 and 160 micro-batches. The accumulated run also
matches the full-batch run. After 5 steps the parameters differ by at most
6.3e-8 and `exp_avg_sq` by 5.5e-12, at float32 rounding.

**FINDING: stepping per micro-batch is not a small error under AdamW.** The
counter reads 20 after 5 effective steps (4x), so the bias correction
1 - beta2^t has moved on to 1 - 0.999^20 instead of 1 - 0.999^5, and every
one of the 20 steps is a full lr-sized Adam step. The largest parameter is
then 0.116 away from the full-batch run, 2.4x the full-batch run's own
largest move from its initial weights (0.049). This is the lesson's "burn
through the schedule", measured.

Expected output: two PASS checks.
"""

from __future__ import annotations

from harness import parity, practice

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "46-gradient-accumulation"
GRID, STEPS, N = (1, 4, 16), 10, 4


def sweep_with_adamw(ref):
    """The lesson's sweep, with its hard-coded SGD built as AdamW instead."""
    made, saved = [], torch.optim.SGD

    def adamw(params, lr):
        made.append(torch.optim.AdamW(params, lr=lr))
        return made[-1]

    torch.optim.SGD = adamw
    try:
        points = ref.sweep_effective_batches(micro_batch=4, accum_grid=GRID, num_steps=STEPS)
    finally:
        torch.optim.SGD = saved
    counts = [sorted({int(s["step"]) for s in opt.state.values()}) for opt in made]
    return [(p.accum_steps, p.steps * p.accum_steps, c, len(o.state)) for p, c, o in zip(points, counts, made)]


def train(ref, mode, steps=5):
    """'full' batch, 'accum' over N chunks, or 'micro' (a step per chunk)."""
    gen = torch.Generator()
    gen.manual_seed(7)
    ref.seed_everything(7)
    model = ref.make_model(32, 48, 8)
    start = [p.detach().clone() for p in model.parameters()]
    opt, loss_fn = torch.optim.AdamW(model.parameters(), lr=0.01), nn.CrossEntropyLoss()
    for _ in range(steps):
        x, y = ref.synthetic_batch(16, 32, 8, gen)
        chunks = list(zip(torch.split(x, 16 // N), torch.split(y, 16 // N)))
        groups = {"full": [[(x, y)]], "accum": [chunks], "micro": [[c] for c in chunks]}[mode]
        for group in groups:
            ref.train_one_optimizer_step(model, opt, group, loss_fn, no_sync_until_last=True, sync_counter=[0])
    state = [opt.state[p] for p in model.parameters()]
    return start, [p.detach().clone() for p in model.parameters()], state


def gap(a, b):
    return max(float((u - v).abs().max()) for u, v in zip(a, b))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    start, full, full_state = train(ref, "full")
    _, accum, accum_state = train(ref, "accum")
    _, micro, micro_state = train(ref, "micro")
    return {
        "sweep": sweep_with_adamw(ref),
        "steps": [sorted({int(s["step"]) for s in st}) for st in (full_state, accum_state, micro_state)],
        "accum_gap": gap(accum, full),
        "sq_gap": gap([s["exp_avg_sq"] for s in accum_state], [s["exp_avg_sq"] for s in full_state]),
        "micro_gap": gap(micro, full),
        "moved": gap(full, start),
    }


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: AdamW's step counter reads 10 after 10 effective steps at accum 1, 4 and 16",
            r["sweep"] == [(1, 10, [10], 6), (4, 40, [10], 6), (16, 160, [10], 6)]
            and r["steps"][:2] == [[5], [5]] and r["accum_gap"] < 1e-6 and r["sq_gap"] < 1e-10,
            f"(accum, micro-batches seen, step counters, params) {r['sweep']}; accumulated vs full "
            f"after 5 steps: params {r['accum_gap']:.2g}, exp_avg_sq {r['sq_gap']:.2g}",
        ),
        practice.Check(
            "FINDING: stepping per micro-batch runs AdamW's counter 4x ahead and moves 2.4x as far",
            r["steps"][2] == [20] and abs(r["micro_gap"] - 0.116) < 0.005 and abs(r["moved"] - 0.049) < 0.002,
            f"per-micro counter {r['steps'][2]}; params {r['micro_gap']:.3f} from the full-batch run, "
            f"which itself moved {r['moved']:.3f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
