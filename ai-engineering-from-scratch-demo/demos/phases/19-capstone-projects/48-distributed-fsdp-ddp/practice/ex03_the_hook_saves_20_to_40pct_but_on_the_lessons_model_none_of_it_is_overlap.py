"""Exercise 3 — the hook saves 20-40% of the step, but on the lesson's model none of that is overlap with backward.

    Add a post-backward hook to the DDP wrapper so the all-reduce overlaps with the rest of the backward; measure the wallclock improvement.

Reading of the exercise: the wrapper is the lesson's `MinimalDDP`, left
unchanged; `add_overlap_hooks()` registers a
`register_post_accumulate_grad_hook` on each of its parameters that starts
an async SUM all-reduce the moment that gradient is ready, and `finish()`
waits on the handles and divides, where the lesson calls `sync_grads()`.
The measured time is forward + backward + gradient sync on 2 gloo ranks, one
intra-op thread per rank, median of 15 steps after 3 warm-up steps. It is
measured on the lesson's own model (`make_model(32, 16, 4)`, batch 8) and on
a scaled-up `make_model(1024, 1024, 1024)` (2.1M parameters, batch 64). Two
controls split the saving: the step with no sync at all, and the same async
calls all issued after backward (concurrency without overlap).

**ANSWER: the hook cuts the step by about 20-40% at both sizes, with
bit-identical gradients.** Over eight runs the lesson's model went from
0.8-1.9 ms to 0.6-1.2 ms per step, and 2.1M parameters from 11-24 ms to
8-16 ms. Exact medians vary with machine load and go in the check details.
The asserted parts are the identical gradients and a saving of at least 10%
at 2.1M parameters.

**FINDING: on the lesson's model the saving is not overlap.** The whole
forward and backward takes 0.1-0.3 ms, less than the time saved, so there is
not enough backward to hide anything behind. Issuing the same four async
calls after backward finishes saves the same amount. The gain comes from
keeping four collectives in flight at once instead of waiting on each one.
Even at 2.1M parameters the hook beat that control by only 0.1-3 ms across
runs, so most of the saving is concurrency there too.

**FINDING: the lesson's docstring says the wrapper already does this.** The
module docstring of `main.py` promises a wrapper that "averages gradients in
a post-backward hook", but `MinimalDDP` registers no hook: the trainer has
to call `sync_grads()` after `backward()`, which is exactly what this
exercise asks to replace. (The class docstring does say production DDP uses
a hook.)

Structure: `launch()` starts this file once per rank as a subprocess with a
60 s timeout and kills stragglers; `worker()` builds the two wrappers from
one seeded model and alternates them step by step. Expected output: three
PASS checks.
"""

from __future__ import annotations

import copy
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

PHASE, LESSON = "19-capstone-projects", "48-distributed-fsdp-ddp"
SIZES = {"lesson": (32, 16, 4, 8), "scaled": (1024, 1024, 1024, 64)}


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


def start(p):
    return p, dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)


def add_overlap_hooks(ddp):
    """Start each gradient's all-reduce as soon as autograd has accumulated it."""
    pending = []
    for p in ddp.module.parameters():
        p.register_post_accumulate_grad_hook(lambda p: pending.append(start(p)))
    return pending


def finish(pending, world_size):
    for p, work in pending:
        work.wait()
        p.grad.div_(world_size)
    pending.clear()


def step(model, x, done):
    model.zero_grad(set_to_none=True)
    dist.barrier()
    begin = time.perf_counter()
    model(x).pow(2).mean().backward()
    done()
    return time.perf_counter() - begin


def measure(ref, ws, rank, in_dim, hidden, out_dim, batch):
    torch.manual_seed(0)
    base = ref.make_model(in_dim, hidden, out_dim)
    plain, hooked = ref.MinimalDDP(base, ws), ref.MinimalDDP(copy.deepcopy(base), ws)
    pending = add_overlap_hooks(hooked)
    torch.manual_seed(rank)
    x = torch.randn(batch, in_dim)
    ways = {"no_sync": (plain, lambda: None),  # control: compute only
            "async_after": (plain, lambda: finish([start(p) for p in plain.parameters()], ws)),  # no overlap
            "sync_grads": (plain, plain.sync_grads), "hook": (hooked, lambda: finish(pending, ws))}
    times = {name: [] for name in ways}
    for _ in range(18):
        for name, (model, done) in ways.items():
            times[name].append(step(model, x, done))
    same = all(torch.equal(a.grad, b.grad) for a, b in zip(plain.parameters(), hooked.parameters()))
    ms = {k: round(statistics.median(v[3:]) * 1e3, 3) for k, v in times.items()}
    return {"same": same, "ms": ms}


def worker(rank, ws, port):
    torch.set_num_threads(1)
    ref = parity.load_reference(PHASE, LESSON, "main")
    ref.init_process_group(rank, ws, "gloo", port)
    result = {name: measure(ref, ws, rank, *size) for name, size in SIZES.items()}
    ref.shutdown_process_group()
    return result


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    names = {n for f in vars(ref.MinimalDDP).values() if callable(f) for n in f.__code__.co_names}
    return {"run": launch(2)[0], "module_doc": ref.__doc__, "hooks": sorted(n for n in names if "hook" in n)}


def verify(result):
    ms = {k: r["ms"] for k, r in result["run"].items()}
    gain = {k: round(1 - m["hook"] / m["sync_grads"], 3) for k, m in ms.items()}
    saved = round(ms["lesson"]["sync_grads"] - ms["lesson"]["hook"], 3)
    return [
        practice.Check(
            "ANSWER: the hook cuts the step with bit-identical gradients",
            all(r["same"] for r in result["run"].values()) and gain["scaled"] >= 0.10,
            f"bit-identical grads {[r['same'] for r in result['run'].values()]}; wallclock saved {gain}",
        ),
        practice.Check(
            "FINDING: on the lesson's model the saving is not overlap",
            saved > ms["lesson"]["no_sync"],
            f"lesson model saves {saved} ms, more than its whole forward+backward; median ms {ms}",
        ),
        practice.Check(
            "FINDING: the lesson's docstring says the wrapper already averages in a post-backward hook",
            "averages gradients in a post-backward hook" in result["module_doc"] and result["hooks"] == [],
            f"hook-related names MinimalDDP's methods use: {result['hooks']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    if sys.argv[1:2] == ["--rank"]:
        raise SystemExit(print(json.dumps(worker(*map(int, [os.environ["RANK"], *sys.argv[2:]])))))
    raise SystemExit(practice.selfcheck(globals()))
