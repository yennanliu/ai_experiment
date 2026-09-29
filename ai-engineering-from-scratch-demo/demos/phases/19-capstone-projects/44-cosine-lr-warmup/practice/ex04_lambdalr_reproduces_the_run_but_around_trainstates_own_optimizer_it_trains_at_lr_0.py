"""Exercise 4 -- LambdaLR reproduces the run exactly, but wrapped around TrainState's own optimizer it trains at LR 0.

    Wire the schedule into a `torch.optim.lr_scheduler.LambdaLR` so it
    composes with framework code. The lesson uses a plain step function; what
    does the wrapper change?

Reading of the exercise: `wrap()` builds AdamW with the same hyperparameters
as `TrainState` (betas 0.9/0.95, eps 1e-8, weight decay 0.01), but at lr =
lr_max, and gives it `LambdaLR(opt, lambda k: schedule.lr(k) / lr_max)`.
It then runs the lesson's 20-step demo (warmup 4, total 20, 1e-2 -> 1e-4,
`build_toy_model`) in the framework order: optimizer.step(), then
scheduler.step(). Its LR, loss and final weights are compared with the
lesson's `TrainState`. Two natural mistakes are also run: keeping
`TrainState`'s order, and wrapping `TrainState`'s own optimizer.

**ANSWER: the numbers do not change; where the schedule lives does.** The
wrapped run applies the same LR on 18 of the 20 steps bit for bit. The
other two differ by 1 ulp, because lr(k) / lr_max * lr_max is a round trip
through a division. All 20 losses and the final weights are identical. What
changes is the representation and the owner. The schedule becomes a
multiplier on each param group's `initial_lr`, so one lambda scales the
decay and no-decay groups alike. The step counter moves from
`TrainState.global_step` into the scheduler's `last_epoch`, which is part of
`state_dict()` and so survives a checkpoint. The lambda itself is not
saved: `lr_lambdas` is `[None]`.

**FINDING: wrapped around TrainState's own optimizer, the scheduler trains at
LR 0.** `TrainState` builds AdamW at `lr=schedule.lr(0)`, which is 0.0.
LambdaLR takes that as its base, so every multiplier gives 0 and 20 steps
leave the loss unchanged. The multiplier form needs the optimizer built at
lr_max.

**FINDING: keeping the lesson's order shifts the schedule by one step.**
`TrainState` sets the LR and then steps the optimizer. Written with
LambdaLR, that becomes scheduler.step() before optimizer.step(). PyTorch
then warns, the first update runs at 2.5e-3 instead of 0, and the peak lands
at step 3 instead of step 4. The final logged loss is 0.1768 instead of
0.1857. The lesson has nothing to checkpoint either way: `TrainState` has no
`state_dict`, although the doc says the schedule reads `global_step` "from
the trainer's checkpoint".

Structure: `lesson()` runs `TrainState`; `wrap()` runs the LambdaLR loop in
either order; `frozen()` wraps `TrainState`'s own optimizer.
"""

from __future__ import annotations

import math
import warnings

from harness import parity, practice

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None
torch.set_num_threads(1)

PHASE, LESSON = "19-capstone-projects", "44-cosine-lr-warmup"
REF = parity.load_reference(PHASE, LESSON, "main")
SCHED = REF.CosineWithWarmup(warmup_steps=4, total_steps=20, lr_max=1e-2, lr_min=1e-4)
LOSS = nn.functional.mse_loss


def lesson():
    model, x, y = REF.build_toy_model()
    state = REF.TrainState(model, SCHED, LOSS)
    for _ in range(20):
        state.step(x, y)
    return [r.lr for r in state.log], [r.loss for r in state.log], model, state


def loop(model, x, y, opt, sched, before):
    lrs, losses = [], []
    for _ in range(20):
        if before:
            sched.step()
        opt.zero_grad(set_to_none=True)
        loss = LOSS(model(x), y)
        loss.backward()
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        losses.append(loss.item())
        if not before:
            sched.step()
    return lrs, losses


def wrap(before=False):
    model, x, y = REF.build_toy_model()
    opt = torch.optim.AdamW(model.parameters(), lr=SCHED.lr_max, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: SCHED.lr(k) / SCHED.lr_max)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lrs, losses = loop(model, x, y, opt, sched, before)
    return lrs, losses, model, sched.state_dict(), [str(w.message)[:64] for w in caught]


def frozen():
    model, x, y = REF.build_toy_model()
    state = REF.TrainState(model, SCHED, LOSS)
    sched = torch.optim.lr_scheduler.LambdaLR(state.optimizer, lambda k: SCHED.lr(k) / SCHED.lr_max)
    lrs, losses = loop(model, x, y, state.optimizer, sched, False)
    return sched.base_lrs, set(lrs), losses[0] - losses[-1]


def solve():
    ref_lrs, ref_losses, ref_model, state = lesson()
    lrs, losses, model, sd, caught = wrap()
    early_lrs, early_losses, _, _, early_caught = wrap(before=True)
    return {
        "lr_same": sum(a == b for a, b in zip(ref_lrs, lrs)),
        "lr_ulps": sorted({round(abs(a - b) / math.ulp(a)) for a, b in zip(ref_lrs, lrs) if a != b}),
        "loss_same": sum(a == b for a, b in zip(ref_losses, losses)),
        "weights_diff": max(float((p - q).detach().abs().max())
                            for p, q in zip(ref_model.parameters(), model.parameters())),
        "sd": (sd["last_epoch"], sd["lr_lambdas"]), "warned": caught,
        "frozen": frozen(), "train_state_sd": hasattr(state, "state_dict"),
        "early": (early_lrs[0], early_lrs.index(max(early_lrs)), early_losses[-1], ref_losses[-1]),
        "early_warned": early_caught,
        "doc": "reads `global_step` from the trainer's checkpoint" in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    r = result
    e = r["early"]
    return [
        practice.Check(
            "ANSWER: the numbers do not change; where the schedule lives does",
            (r["lr_same"], r["lr_ulps"], r["loss_same"], r["sd"], r["warned"]) == (18, [1], 20, (20, [None]), [])
            and r["weights_diff"] < 1e-7,
            f"LR bit-equal on {r['lr_same']}/20 (others off by {r['lr_ulps']} ulp), losses "
            f"{r['loss_same']}/20, max weight diff {r['weights_diff']}; state_dict last_epoch/"
            f"lr_lambdas {r['sd']}",
        ),
        practice.Check(
            "FINDING: wrapped around TrainState's own optimizer, the scheduler trains at LR 0",
            r["frozen"] == ([0.0], {0.0}, 0.0),
            f"base_lrs {r['frozen'][0]}, LRs applied {r['frozen'][1]}, loss drop {r['frozen'][2]}",
        ),
        practice.Check(
            "FINDING: keeping the lesson's order shifts the schedule by one step",
            e[:2] == (0.0025, 3) and [round(e[2], 4), round(e[3], 4)] == [0.1768, 0.1857]
            and r["early_warned"] == ["Detected call of `lr_scheduler.step()` before `optimizer.step()`"]
            and not r["train_state_sd"] and r["doc"],
            f"first LR {e[0]}, peak at step {e[1]}, final loss {e[2]:.4f} vs {e[3]:.4f}; warning "
            f"{r['early_warned']}; TrainState.state_dict exists: {r['train_state_sd']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
