"""Exercise 5 — sync is 93% of the step at batch 1, 16 and 256, and compute catches up only past 4,096 rows.

    Measure the gradient-sync overhead as a fraction of step time for batch sizes 1, 16, 256 and explain the scaling.

Reading of the exercise: the model and wrapper are the lesson's own
(`MiniMLP`, `DistributedDataParallel.sync_grads`) on its 4 gloo ranks, one
intra-op thread per rank. A step is zero_grad + forward + backward (compute),
`sync_grads` (sync) and `optimizer.step`, each timed after a barrier; the
fraction is median sync over median step. Batch sizes 1, 16 and 256 as
asked, plus 4,096 and 32,768 to find where compute catches up. All five
sizes are interleaved over 30 rounds, so machine load hits them alike.

**ANSWER: about 93% at all three sizes, and it does not scale.** Median
per-step time is about 0.12-0.15 ms of compute, 2.2-2.4 ms of sync and 0.03 ms
of optimizer step, so sync / step is 0.93-0.94 at batch 1, 16 and 256. Sync
cost depends on the gradient, not on the batch. `MiniMLP` has 6 tensors and
6,928 bytes of gradient at every batch size, and the lesson makes one
all-reduce per tensor. So sync is 6 x gloo's call latency. Compute does not
grow either: 256 rows cost the same ~0.14 ms as 1 row, because at this size
the step is framework overhead, not arithmetic.

**FINDING: compute catches up only at thousands of rows.** At 4,096 rows per
rank sync is still 0.74-0.76 of the step. At 32,768 rows compute (about 9 ms)
finally dominates and sync falls to 0.21. The doc's goal of making "the
gradient sync nearly free relative to compute" is far off on its own model.
The same 6,928 bytes in one flat all-reduce take about 0.4 ms instead of 2.3:
the call count, not the data, is what costs.

Structure: `launch()` starts this file once per rank as a subprocess with a
90 s timeout and kills stragglers; `worker()` times the phases of a step.
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
WS, ROUNDS, BATCHES = 4, 30, (1, 16, 256, 4096, 32768)


def launch(ws, *args, timeout=90):
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


def timed(fn):
    dist.barrier()
    begin = time.perf_counter()
    fn()
    return time.perf_counter() - begin


def one_step(model, ddp, opt, x, y):
    def compute():
        opt.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(model(x), y).backward()

    def flat_sync():  # control: the same bytes in one all-reduce
        dist.all_reduce(torch.cat([p.grad.flatten() for p in model.parameters()]))

    return timed(compute), timed(ddp.sync_grads), timed(opt.step), timed(flat_sync)


def worker(rank, ws, port):
    torch.set_num_threads(1)
    ref = parity.load_reference(PHASE, LESSON, "main")
    os.environ["GLOO_SOCKET_IFNAME"] = ref._loopback_iface()
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=ws,
                            timeout=datetime.timedelta(seconds=30))
    torch.manual_seed(ref.SEED)
    model = ref.MiniMLP()
    ddp, opt = ref.DistributedDataParallel(model, ws), torch.optim.SGD(model.parameters(), lr=1e-4)
    data = {b: ref.make_dataset(ref.SEED + rank, b) for b in BATCHES}
    times = {b: [] for b in BATCHES}
    for _ in range(ROUNDS):
        for b in BATCHES:
            times[b].append(one_step(model, ddp, opt, *data[b]))
    dist.destroy_process_group()
    out = {}
    for b, rows in times.items():
        medians = (round(statistics.median(col[3:]) * 1e3, 3) for col in zip(*rows))
        out[b] = dict(zip(("compute", "sync", "step", "flat"), medians))
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    run, params = {int(b): v for b, v in launch(WS)[0].items()}, list(ref.MiniMLP().parameters())
    return {"run": run, "frac": {b: round(r["sync"] / (r["compute"] + r["sync"] + r["step"]), 3) for b, r in run.items()},
            "asked": {k: [run[b][k] for b in (1, 16, 256)] for k in ("sync", "compute")},
            "flat": [v["flat"] for v in run.values()], "flat_2x": all(v["flat"] * 2 < v["sync"] for v in run.values()),
            "grad_bytes": sum(p.numel() * p.element_size() for p in params), "tensors": len(params),
            "doc": parity.doc_text(PHASE, LESSON)}


def verify(result):
    f, syncs, compute = result["frac"], result["asked"]["sync"], result["asked"]["compute"]
    asked = [f[b] for b in (1, 16, 256)]
    return [
        practice.Check(
            "ANSWER: sync is over 60% of the step at batch 1, 16 and 256 alike -- the fraction does not scale",
            min(asked) >= 0.6 and max(asked) - min(asked) < 0.15,
            f"sync / step {f}; median ms per phase {result['run']}",
        ),
        practice.Check(
            "FINDING: the sync is a fixed 6-call latency: 6,928 bytes whatever the batch, 256x rows ~ free compute",
            (result["grad_bytes"], result["tensors"]) == (6928, 6) and max(syncs) < 2 * min(syncs)
            and compute[2] < 2.5 * compute[0] and result["flat_2x"],
            f"sync ms at 1/16/256 {syncs}; compute ms {compute}; one flat all-reduce of the same bytes {result['flat']} ms",
        ),
        practice.Check(
            "FINDING: the doc's 'nearly free relative to compute' needs > 4,096 rows per rank here",
            f[4096] > 0.5 > f[32768] and "nearly free relative to compute" in result["doc"],
            f"sync / step at 4,096 rows {f[4096]}, at 32,768 rows {f[32768]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    if sys.argv[1:2] == ["--rank"]:
        raise SystemExit(print(json.dumps(worker(*map(int, [os.environ["RANK"], *sys.argv[2:]])))))
    raise SystemExit(practice.selfcheck(globals()))
