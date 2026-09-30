"""Exercise 5 -- a 2x2 pipeline x DDP grid matches the full batch, and send-before-recv deadlocks gloo at 16 floats.

    Combine pipeline with DDP (each pipeline rank is replicated across a data-parallel group) and reason through the 2D schedule.

Reading of the exercise: 4 gloo ranks on localhost form a 2 x 2 grid. Rank
r holds pipeline stage r % 2 (the lesson's two `StageMLP` stages, seeded as
its ranks seed them) in data-parallel replica r // 2. Pipeline traffic is
send/recv inside each replica ({0,1} and {2,3}); gradient averaging is an
all-reduce inside each stage's DP group ({0,2} and {1,3}). Each replica
pipelines its own M = 4 microbatches of 8 rows with loss / M, then the DP
group sums and halves. "Reason through the 2D schedule" is answered from the
lesson's `gpipe_schedule`: the cycle at which each stage issues its last
backward, i.e. how long its all-reduce can hide behind other stages' work.

**ANSWER: after one step the averaged gradients equal a single process's
full-batch gradient on all 64 rows (relative error under 1e-6), the two
replicas of each stage hold bit-identical gradients, and before the
all-reduce they differ.** Each rank makes 2M = 8 point-to-point calls and 1
all-reduce per tensor. In the 2D schedule the two pipelines never talk to
each other; they meet only at the DP all-reduce, one per stage. Stage s
finishes its backward s cycles before the step ends (N=4, M=8: slack 0, 1,
2, 3), so stage 3 can hide its all-reduce behind the drain and stage 0,
which finishes last, cannot hide any of it.

**FINDING: the doc's deadlock warning is real on gloo, even at 16 floats.**
With both ranks calling `send` before `recv`, the pair makes no progress
within 12 s and is killed; the same pair ordered as the lesson orders it
(rank 0 sends first, rank 1 receives first) swaps its 16 floats. Gloo's `send` does not
return until the peer receives, so the ordering is not optional.

Structure: `launch()` starts this file once per rank as a subprocess and
kills stragglers after a timeout; `worker()` is one grid rank, `pair()`
the deadlock probe. Expected output: two PASS checks.
"""

from __future__ import annotations

import datetime
import json
import os
import socket
import subprocess
import sys

from harness import parity, practice

try:
    import torch
    import torch.distributed as dist
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "79-pipeline-parallel"
BATCH, M, DP = 8, 4, 2


def launch(ws, mode, timeout):
    """Run this file as `ws` gloo ranks; return each rank's JSON, or None if they had to be killed."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        argv = [sys.executable, __file__, "--rank", mode, str(ws), str(s.getsockname()[1])]
    pipe = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True}
    procs = [subprocess.Popen(argv, env=os.environ | {"RANK": str(r)}, **pipe) for r in range(ws)]
    try:
        outs = [p.communicate(timeout=timeout) for p in procs]
    except subprocess.TimeoutExpired:
        return None
    finally:
        for p in procs:
            p.kill()
    if any(map(subprocess.Popen.poll, procs)):
        raise RuntimeError(f"a rank failed: {[err[-300:] for _, err in outs]}")
    return [json.loads(out.splitlines()[-1]) for out, _ in outs]


def init(rank, ws, port):
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=ws,
                            timeout=datetime.timedelta(seconds=30))


def make_stage(ref, stage):
    torch.manual_seed(ref.SEED + stage)
    return ref.StageMLP(*[(16, 32, 16), (16, 32, 4)][stage])


def data(ref):
    g = torch.Generator().manual_seed(ref.SEED + 99)
    return [torch.randn(BATCH, 16, generator=g) for _ in range(DP * M)]


def worker(rank, ws, port):
    ref = parity.load_reference(PHASE, LESSON, "main")
    init(rank, ws, port)
    groups = [dist.new_group([s, s + 2]) for s in range(2)]
    stage, replica, p2p = rank % 2, rank // 2, 0
    model = make_stage(ref, stage)
    for x in data(ref)[replica * M:(replica + 1) * M]:
        if stage == 0:
            act, grad = model(x), torch.zeros(BATCH, 16)
            dist.send(act.detach(), dst=rank + 1)
            dist.recv(grad, src=rank + 1)
            act.backward(grad)
        else:
            act = torch.zeros(BATCH, 16)
            dist.recv(act, src=rank - 1)
            pred = model(act.requires_grad_(True))
            (torch.nn.functional.mse_loss(pred, torch.zeros_like(pred)) / M).backward()
            dist.send(act.grad, dst=rank - 1)
        p2p += 2
    local = [p.grad.clone() for p in model.parameters()]
    for p in model.parameters():
        dist.all_reduce(p.grad, group=groups[stage])
        p.grad /= DP
    dist.destroy_process_group()
    return {"local": [g.tolist() for g in local], "avg": [p.grad.tolist() for p in model.parameters()], "p2p": p2p}


def pair(rank, ws, port, both_send):
    """Two ranks swap 16 floats; with both_send every rank sends before it receives."""
    init(rank, ws, port)
    buf = torch.zeros(16)
    ops = [lambda: dist.send(torch.ones(16), dst=1 - rank), lambda: dist.recv(buf, src=1 - rank)]
    for op in ops if both_send or rank == 0 else ops[::-1]:
        op()
    return {"got": buf.sum().item()}


def full_batch(ref):
    stages = [make_stage(ref, s) for s in range(2)]
    pred = stages[1](stages[0](torch.cat(data(ref))))
    torch.nn.functional.mse_loss(pred, torch.zeros_like(pred)).backward()
    return [[p.grad for p in s.parameters()] for s in stages]


def rel(a, b):
    return max(((torch.tensor(x) - y).abs().max() / y.abs().max()).item() for x, y in zip(a, b))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ranks, full = launch(4, "grid", 60), full_batch(ref)
    sched = ref.gpipe_schedule(4, 8)  # slack: cycles from each stage's last backward to the step's end
    end = max(c for c, *_ in sched)
    return {
        "err": max(rel(r["avg"], full[i % 2]) for i, r in enumerate(ranks)),
        "replicas_equal": all(ranks[s]["avg"] == ranks[s + 2]["avg"] for s in range(2)),
        "locals_differ": all(ranks[s]["local"] != ranks[s + 2]["local"] for s in range(2)),
        "p2p": [r["p2p"] for r in ranks],
        "slack": [end - max(c for c, s, _, ph in sched if (s, ph) == (st, "B")) for st in range(4)],
        "ordered": launch(2, "ordered", 60), "deadlock": launch(2, "both_send", 12) is None,
    }


def verify(r):
    return [
        practice.Check(
            "ANSWER: the 2x2 grid's averaged gradients equal the full batch; only stage 0's all-reduce is exposed",
            r["err"] < 1e-6 and r["replicas_equal"] and r["locals_differ"] and r["p2p"] == [8] * 4
            and r["slack"] == [0, 1, 2, 3],
            f"relative error vs full batch {r['err']:.2g}; replicas bit-identical {r['replicas_equal']}, "
            f"differ before all-reduce {r['locals_differ']}; p2p calls per rank {r['p2p']}; "
            f"GPipe N=4 M=8 cycles between each stage's last backward and step end {r['slack']}",
        ),
        practice.Check(
            "FINDING: the doc's deadlock warning is real on gloo, even at 16 floats",
            r["deadlock"] and r["ordered"] == [{"got": 16.0}] * 2,
            f"ordered send/recv pair: {r['ordered']}; both ranks send-then-recv killed after 12 s "
            f"without finishing: {r['deadlock']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    if sys.argv[1:2] == ["--rank"]:
        rank, args = int(os.environ["RANK"]), [int(a) for a in sys.argv[3:]]
        out = worker(rank, *args) if sys.argv[2] == "grid" else pair(rank, *args, sys.argv[2] == "both_send")
        raise SystemExit(print(json.dumps(out)))
    raise SystemExit(practice.selfcheck(globals()))
