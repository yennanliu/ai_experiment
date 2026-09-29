"""Exercise 4 -- routed to DDP.no_sync, all-reduces drop by N-1 per step; the stock stand-in reports 1 while DDP runs all N.

    Introduce a real `DistributedDataParallel` wrapper and route the `no_sync_context` to its method. Confirm sync_calls drops by N-1 per effective batch.

Reading of the exercise: a real `torch.nn.parallel.DistributedDataParallel`
wraps the lesson's `make_model(64, 128, 16)` in a one-process gloo group
(world size 1, file-store rendezvous in a temp dir, CPU, no network). The
lesson's `no_sync_context` is replaced for the run by
`lambda model: model.no_sync()`, which is the routing the exercise asks for.
The lesson's own `train_one_optimizer_step` then runs 3 effective steps at
N = 2, 4 and 8 micro-batches of 4. `sync_calls` is the lesson's
`sync_counter`. It is checked against the collectives DDP actually launched,
counted by a pass-through comm hook (the model fits one bucket, so one hook
call is one all-reduce). World size 1 makes each all-reduce an identity, so
the count is real but the network cost is not measured.

**ANSWER: sync_calls drops by N-1 per effective batch, and so do the real
all-reduces.** With no_sync on every non-final micro-batch, both counters
read 1 per effective step (3 over 3 steps) at every N. With
`no_sync_until_last=False` both read N per step: 6, 12 and 24. The drop is
3, 9 and 21 = 3 * (N - 1). The gradients the optimizer steps on are
bit-identical either way.

**FINDING: the lesson's stock `no_sync_context` reports a saving it does not
make.** Its `_NoSyncCtx` never touches the model. Run it around the same DDP
model with `no_sync_until_last=True` and sync_calls reads 3, 3, 3, while DDP
launches 6, 12, 24 all-reduces: all N, every step. `sync_counter` counts
the code path, not collectives, so on a real cluster the stand-in would
report the saving and pay the full N-fold network cost.

Expected output: two PASS checks.
"""

from __future__ import annotations

import pathlib
import tempfile

from harness import parity, practice

try:
    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.nn.parallel import DistributedDataParallel
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "46-gradient-accumulation"
NS, STEPS = (2, 4, 8), 3


def run(ref, n, until_last, routed):
    """(sync_calls, all-reduces launched, final grads) for STEPS effective steps."""
    torch.manual_seed(0)
    model = DistributedDataParallel(ref.make_model(64, 128, 16))
    launched = [0]

    def count(state, bucket):
        launched[0] += 1
        future = torch.futures.Future()
        future.set_result(bucket.buffer())
        return future

    model.register_comm_hook(None, count)
    opt, gen, syncs = torch.optim.SGD(model.parameters(), lr=0.05), torch.Generator(), [0]
    gen.manual_seed(0)
    saved = ref.no_sync_context
    if routed:
        ref.no_sync_context = lambda m: m.no_sync()
    try:
        for _ in range(STEPS):
            micro = [ref.synthetic_batch(4, 64, 16, gen) for _ in range(n)]
            ref.train_one_optimizer_step(
                model, opt, micro, nn.CrossEntropyLoss(), no_sync_until_last=until_last, sync_counter=syncs
            )
    finally:
        ref.no_sync_context = saved
    return syncs[0], launched[0], [p.grad.clone() for p in model.parameters()]


def solve():
    if dist.is_initialized():
        raise practice.Skip("a process group is already initialised in this interpreter")
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory() as tmp:
        store = pathlib.Path(tmp) / "store"
        dist.init_process_group("gloo", init_method=store.as_uri(), rank=0, world_size=1)
        try:
            rows = {n: {mode: run(ref, n, *mode) for mode in ((True, True), (False, True), (True, False))} for n in NS}
        finally:
            dist.destroy_process_group()
    return {
        n: {
            "routed": r[(True, True)][:2], "all_sync": r[(False, True)][:2], "stand_in": r[(True, False)][:2],
            "same_grads": all(torch.equal(a, b) for a, b in zip(r[(True, True)][2], r[(False, True)][2])),
        }
        for n, r in rows.items()
    }


def verify(result):
    table = {n: (r["routed"], r["all_sync"], r["stand_in"]) for n, r in result.items()}
    return [
        practice.Check(
            "ANSWER: routed to DDP.no_sync, sync_calls and real all-reduces both drop by N-1 per step",
            all(r["routed"] == (STEPS, STEPS) and r["all_sync"] == (n * STEPS, n * STEPS) and r["same_grads"]
                for n, r in result.items()),
            "N: ((sync_calls, all-reduces) routed, without no_sync, stand-in) " + str(table),
        ),
        practice.Check(
            "FINDING: the stock stand-in reports 1 sync per step while DDP all-reduces all N",
            all(r["stand_in"] == (STEPS, n * STEPS) for n, r in result.items()),
            "stand-in (sync_calls, all-reduces) " + str({n: r["stand_in"] for n, r in result.items()}),
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
