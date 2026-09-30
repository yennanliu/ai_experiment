"""Exercise 4 — barrier-only ranks drift 0.23 apart while rank 0's loss stays within 3.6% of true DDP.

    Replace gloo with `torch.distributed.barrier()`-only synchronisation to feel the difference between allreduce-based and barrier-based sync.

Reading of the exercise: gloo cannot actually be removed -- on CPU it is
the backend `barrier()` itself runs on -- so "replace" is read as replacing
the gradient all-reduce with a barrier. The lesson's own run (4 gloo ranks,
`MiniMLP`, seed 7, batch 8, lr 0.05, 20 steps, its `make_dataset` slicing)
is trained three ways: as shipped (`sync_grads`), with `sync_grads` swapped
for `dist.barrier()` (the constructor still broadcasts), and as shipped but
with each rank seeding `SEED + rank` before building the model, the case the
doc says breaks equivalence. Every rank records its full parameter vector
after each step; "spread" is the largest element-wise difference from rank 0.

**ANSWER: a barrier syncs time, not values.** With the all-reduce, the four
ranks are bit-identical after every step. With a barrier they start
identical (the broadcast) and then drift: the largest per-element gap is
0.0538 after step 1, 0.1714 after step 5 and 0.2307 after step 20. The
barrier is about 20x cheaper per step (about 0.12 ms against 2.2-3.6 ms for
the 6 all-reduces), because it moves no gradient. It still runs on gloo.

**FINDING: the loss curve will not warn you.** Rank 0's barrier-only loss
never differs from the true DDP loss (`reference_single_process`) by more
than 3.53% over the 20 steps. The all-reduce run matches it exactly (0.0).
Each rank is now a separate model trained on a quarter of the data.

**FINDING: the doc's seed warning does not apply to this code.** The doc
says a rank-specific seed for parameter init stops sync making the replicas
identical, and that "the test for parameter equivalence fails on step 1".
Seeded with `SEED + rank`, the shipped wrapper's constructor broadcast
overwrites the difference: spread 0.0 at every step, and loss gap 0.0.

Structure: `launch()` starts this file once per rank as a subprocess with a
60 s timeout and kills stragglers; `train()` runs one mode and returns the
per-step parameters, losses and sync times.
"""

from __future__ import annotations

import datetime
import json
import os
import socket
import statistics
import subprocess
import sys
import time

from harness import parity, practice

try:
    import torch
    import torch.distributed as dist
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "77-data-parallel-ddp"
WS, STEPS, BATCH, LR = 4, 20, 8, 0.05
MODES = ("allreduce", "barrier", "rank_seed")


def launch(ws, *args, timeout=60):
    """Run this file as `ws` gloo ranks in subprocesses; return each rank's JSON, kill stragglers."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        argv = [sys.executable, __file__, "--rank", str(ws), str(s.getsockname()[1]), *map(str, args)]
    pipe = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True}
    procs = [subprocess.Popen(argv, env=os.environ | {"RANK": str(r)}, **pipe) for r in range(ws)]
    try:
        outs = [p.communicate(timeout=timeout) for p in procs]
    finally:
        for p in procs:
            p.kill()
    if any(p.returncode for p in procs):
        raise RuntimeError(f"a rank failed: {[err[-300:] for _, err in outs]}")
    return [json.loads(out.splitlines()[-1]) for out, _ in outs]


def train(ref, rank, mode):
    torch.manual_seed(ref.SEED + (rank if mode == "rank_seed" else 0))
    model = ref.MiniMLP()
    ddp = ref.DistributedDataParallel(model, WS)
    sync = dist.barrier if mode == "barrier" else ddp.sync_grads
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    x_all, y_all = ref.make_dataset(ref.SEED + 1000, n_total=WS * BATCH * STEPS)
    trace, losses, sync_s = [], [], []
    for step in range(STEPS):
        at = step * WS * BATCH + rank * BATCH
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x_all[at:at + BATCH]), y_all[at:at + BATCH])
        loss.backward()
        begin = time.perf_counter()
        sync()
        sync_s.append(time.perf_counter() - begin)
        opt.step()
        losses.append(loss.item())
        trace.append(torch.cat([p.detach().flatten() for p in model.parameters()]).tolist())
    return {"trace": trace, "losses": losses, "sync_ms": round(statistics.median(sync_s[2:]) * 1e3, 3)}


def worker(rank, ws, port):
    torch.set_num_threads(1)
    ref = parity.load_reference(PHASE, LESSON, "main")
    os.environ["GLOO_SOCKET_IFNAME"] = ref._loopback_iface()
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=ws,
                            timeout=datetime.timedelta(seconds=30))
    out = {mode: train(ref, rank, mode) for mode in MODES} | {"backend": dist.get_backend()}
    dist.destroy_process_group()
    return out


def spread(ranks, mode, step):
    return max(max(abs(a - b) for a, b in zip(ranks[0][mode]["trace"][step], r[mode]["trace"][step])) for r in ranks)


def solve():
    ranks, ref = launch(WS), parity.load_reference(PHASE, LESSON, "main")
    ref_losses, _ = ref.reference_single_process(world_size=WS, steps=STEPS, batch=BATCH, lr=LR)
    return {
        "spread": {m: [round(spread(ranks, m, s), 4) for s in (0, 4, STEPS - 1)] for m in MODES},
        "loss_gap": {m: round(max(abs(a - b) / b for a, b in zip(ranks[0][m]["losses"], ref_losses)), 4) for m in MODES},
        "sync_ms": {m: ranks[0][m]["sync_ms"] for m in MODES}, "backend": ranks[0]["backend"],
        "doc": parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    s, gap, ms = result["spread"], result["loss_gap"], result["sync_ms"]
    return [
        practice.Check(
            "ANSWER: a barrier keeps the ranks in step, not in agreement: they drift 0.054 after one step",
            s["allreduce"] == [0.0, 0.0, 0.0] and s["barrier"] == [0.0538, 0.1714, 0.2307]
            and result["backend"] == "gloo" and 2 * ms["barrier"] < ms["allreduce"],
            f"max element-wise spread from rank 0 after steps 1/5/20 {s}; median sync ms {ms}; "
            f"barrier runs on backend {result['backend']}",
        ),
        practice.Check(
            "FINDING: the barrier run's loss curve stays within 3.6% of true DDP, so the loss will not warn you",
            gap["allreduce"] == 0.0 and gap["barrier"] == 0.0353,
            f"largest relative gap of rank 0's loss to reference_single_process over 20 steps {gap}",
        ),
        practice.Check(
            "FINDING: a per-rank seed does not break equivalence, whatever the doc says; the broadcast repairs it",
            s["rank_seed"] == [0.0, 0.0, 0.0] and gap["rank_seed"] == 0.0
            and "the test for parameter equivalence fails on step 1" in result["doc"],
            f"seed SEED + rank before MiniMLP: spread {s['rank_seed']}, loss gap {gap['rank_seed']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    if sys.argv[1:2] == ["--rank"]:
        raise SystemExit(print(json.dumps(worker(*map(int, [os.environ["RANK"], *sys.argv[2:]])))))
    raise SystemExit(practice.selfcheck(globals()))
