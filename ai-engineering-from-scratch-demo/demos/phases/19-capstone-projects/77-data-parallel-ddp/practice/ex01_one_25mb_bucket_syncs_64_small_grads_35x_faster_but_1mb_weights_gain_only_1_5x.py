"""Exercise 1 — one 25 MB bucket syncs 64 small gradients 35x faster, but on 1 MB weights buckets gain only 1.5x.

    Add gradient buckets of configurable size and measure the speedup vs one-allreduce-per-parameter on a deeper model.

Reading of the exercise: the "one-allreduce-per-parameter" baseline is the
lesson's own `DistributedDataParallel.sync_grads`, unchanged. `bucket_sync()`
walks the same gradients in the same order, packs them into flat buffers of
at most `cap` bytes (a tensor bigger than the cap gets a bucket to itself, as
in PyTorch), issues one SUM all-reduce per buffer, divides by the world size
and copies the slices back. The "deeper model" is 32 `Linear(64, 64)` + ReLU
layers (64 parameter tensors, 133,120 floats). The time measured is the
gradient sync alone on 4 gloo ranks, one thread each, median of 35 steps.
Caps: 64 KB, 256 KB, 1 MB and PyTorch's 25 MB default. A second model, 32
`Linear(512, 512)` layers (1 MB per weight), shows where buckets stop paying.

**ANSWER: on the deep narrow model one bucket is about 35x faster.** The
lesson's 64 per-parameter calls take about 25 ms per sync. With a 64 KB cap
there are 11 calls (about 4.5 ms, 5.5x faster), with 256 KB 3 calls (about
1.4 ms, 17x), and with 1 MB or 25 MB a single call (about 0.7 ms, 35x). The
gradients are bit-identical to the per-parameter sync in every case. The
64 calls cost about 0.39 ms each, which is gloo's latency floor on this
machine: the 532 KB of data are not the cost, the calls are. Exact
milliseconds vary with load and go in the check details; the check asserts
the call counts, identical gradients and at least 5x (25 MB) and 2x (64 KB).

**FINDING: buckets only pay when tensors are small.** On 32 `Linear(512, 512)`
layers every weight is 1 MB, bigger than any cap up to 1 MB. Those caps still
make 64 calls, and the flatten and copy make them about 1.1-1.3x slower than
the lesson's sync. The 25 MB cap (2 calls) gains only 1.5-2x there (1.5x on an
idle machine), against 35x on the narrow model.

**FINDING: the lesson does not bucket at all.** The doc says "For the lesson's
tiny model we group everything into one bucket", but `sync_grads` issues one
`all_reduce` per parameter: 6 calls per step on `MiniMLP`. The doc's "~25 MB"
bucket size is right: it is torch 2.14's `_DEFAULT_BUCKET_CAP_MB`.

Structure: `launch()` starts this file once per rank as a subprocess with a
90 s timeout and kills stragglers; `time_model()` builds each model from one
seed and times every sync variant in turn on the same gradients.
"""

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
CAPS = {"per_param": None, "64KB": 64 << 10, "256KB": 256 << 10, "1MB": 1 << 20, "25MB": 25 << 20}


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


def bucket_sync(params, world_size, cap):
    """One all-reduce per bucket of at most `cap` bytes (a bigger tensor goes alone); returns the call count."""
    buckets, size = [[]], 0
    for g in map(lambda p: p.grad, params):
        if buckets[-1] and size + g.nbytes > cap:
            buckets, size = buckets + [[]], 0
        buckets[-1], size = buckets[-1] + [g], size + g.nbytes
    for bucket in buckets:
        flat = torch.cat([g.flatten() for g in bucket])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        for g, piece in zip(bucket, flat.div_(world_size).split([g.numel() for g in bucket])):
            g.copy_(piece.view_as(g))
    return len(buckets)


def time_model(ref, ws, depth, width):
    torch.manual_seed(0)
    model = torch.nn.Sequential(*sum(([torch.nn.Linear(width, width), torch.nn.ReLU()] for _ in range(depth)), []))
    ddp, params, x = ref.DistributedDataParallel(model, ws), list(model.parameters()), torch.randn(8, width)
    times, calls, grads = {k: [] for k in CAPS}, {}, {}
    for _ in range(40):
        for name, cap in CAPS.items():
            model.zero_grad(set_to_none=True)
            model(x).pow(2).mean().backward()
            dist.barrier()
            begin = time.perf_counter()
            calls[name] = bucket_sync(params, ws, cap) if cap else (ddp.sync_grads(), len(params))[1]
            times[name].append(time.perf_counter() - begin)
            grads[name] = torch.cat([p.grad.flatten() for p in params])
    return {"ms": {k: round(statistics.median(v[5:]) * 1e3, 3) for k, v in times.items()}, "calls": calls,
            "same": all(map(grads["per_param"].equal, grads.values())),
            "shape": (len(params), grads["per_param"].numel())}


def worker(rank, ws, port):
    torch.set_num_threads(1)
    ref = parity.load_reference(PHASE, LESSON, "main")
    os.environ["GLOO_SOCKET_IFNAME"] = ref._loopback_iface()
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=ws,
                            timeout=datetime.timedelta(seconds=30))
    result = {name: time_model(ref, ws, *shape) for name, shape in {"narrow": (32, 64), "wide": (32, 512)}.items()}
    counted, real = [], ref.dist.all_reduce
    ref.dist.all_reduce = lambda t, **kw: (counted.append(t.numel()), real(t, **kw))[1]
    lesson = ref.MiniMLP()
    lesson(torch.randn(8, ref.IN_DIM)).sum().backward()
    ref.DistributedDataParallel(lesson, ws).sync_grads()
    dist.destroy_process_group()
    return result | {"lesson_calls": len(counted)}


def solve():
    run = launch(4)[0]
    n, w = run["narrow"]["ms"], run["wide"]["ms"]
    return {"run": run, "doc": parity.doc_text(PHASE, LESSON),
            "speed": {k: round(n["per_param"] / v, 1) for k, v in n.items()},
            "wide_best": round(w["per_param"] / min(w.values()), 2), "torch_cap_mb": torch.nn.parallel.distributed._DEFAULT_BUCKET_CAP_MB}


def verify(result):
    n, w, speed, run = result["run"]["narrow"], result["run"]["wide"], result["speed"], result["run"]
    return [
        practice.Check("ANSWER: on 64 small tensors one 25 MB bucket syncs >= 5x faster than 64 per-parameter calls",
            (n["calls"], n["same"], n["shape"]) == ({"per_param": 64, "64KB": 11, "256KB": 3, "1MB": 1, "25MB": 1},
                                                   True, [64, 133120]) and speed["25MB"] >= 5 and speed["64KB"] >= 2,
            f"calls {n['calls']}, bit-identical grads {n['same']}; median ms {n['ms']}; speedup {speed}"),
        practice.Check("FINDING: with 1 MB weights caps up to 1 MB still make 64 calls and 25 MB gains 4x less",
            (w["calls"]["1MB"], w["calls"]["25MB"], w["same"]) == (64, 2, True) and speed["25MB"] >= 4 * result["wide_best"],
            f"wide calls {w['calls']}; median ms {w['ms']}; best speedup {result['wide_best']}x"),
        practice.Check("FINDING: the doc says the tiny model goes in one bucket; sync_grads makes 6 calls",
            "we group everything into one bucket" in result["doc"] and run["lesson_calls"] == 6
            and result["torch_cap_mb"] == 25,
            f"lesson sync_grads all_reduce calls on MiniMLP: {run['lesson_calls']}; torch default bucket_cap_mb {result['torch_cap_mb']} (the doc's ~25 MB is right)"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    if sys.argv[1:2] == ["--rank"]:
        raise SystemExit(print(json.dumps(worker(*map(int, [os.environ["RANK"], *sys.argv[2:]])))))
    raise SystemExit(practice.selfcheck(globals()))
