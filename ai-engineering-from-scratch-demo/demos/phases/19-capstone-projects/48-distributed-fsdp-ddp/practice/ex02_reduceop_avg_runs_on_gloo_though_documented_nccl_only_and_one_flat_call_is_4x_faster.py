"""Exercise 2 — ReduceOp.AVG works on gloo and is bit-identical to the manual average, although torch documents it as NCCL-only.

    Replace the manual averaging with `dist.all_reduce(op=dist.ReduceOp.AVG)` and time the difference.

Reading of the exercise: "the manual averaging" is the lesson's
`all_reduce_grads_` (SUM, then `div_(world_size)`). The replacement keeps its
signature and return value but issues one AVG all-reduce per parameter; it
is patched into the loaded module, so the lesson's own
`manual_all_reduce_matches_single_process` runs through it. Both are timed
on the lesson's model gradients (596 floats in 4 tensors): 7 interleaved
blocks of 100 calls each, fastest block, as seen by rank 0. The measured numbers land in
the check details; only differences of 2x or more are asserted.

**ANSWER: nothing changes but one line, and the time difference is noise.**
AVG gives bit-identical gradients to SUM-then-divide at world sizes 2, 3 and
4, and the lesson's gradient check gives the same max diff through either.
The two timings land within about 30% of each other, in either direction
from run to run (world size 4: about 3 ms per full-gradient sync for both).
The division alone costs about 0.005 ms, under 1% of the sync, so no change
to it could save more than that.

**FINDING: torch's own docs say this should not work.** The installed torch's
`ReduceOp` docstring says "AVG is only available with the NCCL backend" and
that AVG "divides values by the world size before summing". On this gloo
build AVG runs, and at world size 3 its result equals sum-then-divide bit for
bit, while divide-then-sum differs by up to 1.2e-7. What
it actually computes is the lesson's manual order. (Source: the
`torch.distributed.ReduceOp.__doc__` of the installed torch 2.14.0, read at
run time.)

**FINDING: the cost is the number of collectives, not the averaging.** One
all-reduce of all 596 gradients flattened into a single buffer was 3.4x to
5.4x faster than the lesson's four per-tensor calls, at every world size. That
is the bucketing the doc's "Use It" section credits to production DDP.

Structure: `launch()` starts this file once per rank as a subprocess with a
60 s timeout and kills stragglers; `worker()` runs the comparisons and the
timing inside one gloo group. Expected output: three PASS checks.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time

from harness import parity, practice

try:
    import torch
    import torch.distributed as dist
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "48-distributed-fsdp-ddp"


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


def avg_grads_(module, world_size):
    """The lesson's all_reduce_grads_ with SUM + div_(world_size) replaced by ReduceOp.AVG."""
    for p in module.parameters():
        dist.all_reduce(p.grad.data, op=dist.ReduceOp.AVG)
    return sum(float(p.grad.data.pow(2).sum().item()) for p in module.parameters()) ** 0.5


def seeded(ref, rank):
    torch.manual_seed(0)
    model = ref.make_model(32, 16, 4)
    torch.manual_seed(100 + rank)
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    return model


def timed(fn, reps=100):
    dist.barrier()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - start) / reps * 1e3


def timings(ref, model, ws):
    grads = [p.grad for p in model.parameters()]
    flat = torch.cat([g.flatten() for g in grads])
    ways = {"sum_div": lambda: ref.all_reduce_grads_(model, ws), "avg": lambda: avg_grads_(model, ws),
            "div_only": lambda: [g.div_(1.0) for g in grads],
            "one_flat_call": lambda: dist.all_reduce(flat, op=dist.ReduceOp.AVG)}
    blocks = [{name: timed(fn) for name, fn in ways.items()} for _ in range(7)]  # interleaved
    return {name: round(min(block[name] for block in blocks), 3) for name in ways}


def worker(rank, ws, port):
    ref = parity.load_reference(PHASE, LESSON, "main")
    ref.init_process_group(rank, ws, "gloo", port)
    manual, avg, pre = seeded(ref, rank), seeded(ref, rank), seeded(ref, rank)
    ref.all_reduce_grads_(manual, ws)
    avg_grads_(avg, ws)
    for p in pre.parameters():  # what the torch docstring says AVG does: divide, then sum
        dist.all_reduce(p.grad.div_(ws), op=dist.ReduceOp.SUM)
    pairs = list(zip(manual.parameters(), avg.parameters(), pre.parameters()))
    out = {"avg_equal": all(torch.equal(a.grad, b.grad) for a, b, _ in pairs),
           "pre_diff": max(float((a.grad - c.grad).abs().max()) for a, _, c in pairs),
           "lesson_check": [ref.manual_all_reduce_matches_single_process(rank, ws, 32, 4, 8)]}
    ref.all_reduce_grads_ = avg_grads_
    out["lesson_check"].append(ref.manual_all_reduce_matches_single_process(rank, ws, 32, 4, 8))
    m = out["ms"] = timings(ref, seeded(ref, rank), ws)
    ref.shutdown_process_group()
    return out | {"same": out["avg_equal"] and out["lesson_check"][0] == out["lesson_check"][1],
                  "avg_ratio": m["avg"] / m["sum_div"], "div_share": m["div_only"] / m["sum_div"],
                  "flat_speedup": round(m["sum_div"] / m["one_flat_call"], 1)}


def solve():
    runs = [launch(ws)[0] for ws in (2, 3, 4)]
    return {key: [r[key] for r in runs] for key in runs[0]} | {"doc": " ".join(dist.ReduceOp.__doc__.split())}


def verify(result):
    r, doc = result, result["doc"]
    return [
        practice.Check(
            "ANSWER: nothing changes but one line, and the time difference is noise",
            r["same"] == [True] * 3 and 1 / 2 < min(r["avg_ratio"]) and max(r["avg_ratio"]) < 2
            and max(r["div_share"]) < 1 / 20,
            f"lesson check (norm, diff) SUM vs AVG at 4 ranks {r['lesson_check'][2]}; "
            f"fastest-block ms per call at 2/3/4 ranks {r['ms']}",
        ),
        practice.Check(
            "FINDING: torch's own docs say AVG is NCCL-only and divides first; on gloo it runs and sums first",
            ("AVG`` is only available with the ``NCCL`` backend" in doc, r["same"][1], r["pre_diff"][1] > 0)
            == (True, True, True) and "divides values by the world size before summing" in doc,
            f"divide-then-sum vs sum-then-divide, max diff at 2/3/4 ranks: {r['pre_diff']}",
        ),
        practice.Check(
            "FINDING: the cost is the number of collectives, not the averaging",
            min(r["flat_speedup"]) > 2,
            f"one flat call is {r['flat_speedup']}x faster than 4 per-tensor calls at 2/3/4 ranks",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    if sys.argv[1:2] == ["--rank"]:
        raise SystemExit(print(json.dumps(worker(*map(int, [os.environ["RANK"], *sys.argv[2:]])))))
    raise SystemExit(practice.selfcheck(globals()))
