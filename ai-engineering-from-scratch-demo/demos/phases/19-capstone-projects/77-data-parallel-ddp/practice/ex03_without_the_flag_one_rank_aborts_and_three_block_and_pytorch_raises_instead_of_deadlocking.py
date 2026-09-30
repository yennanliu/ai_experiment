"""Exercise 3 — without the flag one rank aborts and three block, and PyTorch's DDP raises instead of deadlocking.

    Add a `find_unused_parameters` mode where the forward sometimes skips one of the MLP layers; without the flag the run should deadlock.

Reading of the exercise: the forward runs the lesson's `MiniMLP` layers by
hand and skips the middle `Linear(32, 32)` + ReLU whenever
(step + rank) % 4 == 0, so on each of 6 steps exactly one of the 4 gloo ranks
leaves 2 of the 6 parameters without a gradient. `find_unused=True` gives
every such parameter a zero gradient before the lesson's own `sync_grads`,
so every rank issues the same 6 all-reduces in the same order; unused on one
rank then means "contributes 0 to the mean". It is checked against one
process that walks the same 4 microbatches with the same skips (loss / 4)
and against PyTorch's `DistributedDataParallel(find_unused_parameters=True)`.
"Without the flag" is run twice: the lesson's wrapper as shipped, and
PyTorch's DDP with the flag off. Each rank gets a 5 s process-group timeout,
and any rank still alive after 15 s is recorded as hung and killed.

**ANSWER: zero-filling the unused gradients before the lesson's
`sync_grads` works.** Every rank runs all 6 steps, and the parameters match
the single-process baseline to 3.0e-8. PyTorch's `find_unused_parameters=True`
matches it to 1.5e-8.

**FINDING: without the flag the lesson's wrapper does not deadlock cleanly.**
Its `sync_grads` skips a parameter whose `grad` is None. So on step 0, rank
0 issues 4 all-reduces while the other ranks issue 6, and the third call
pairs a 128-float tensor with a 1,024-float one. Rank 0 dies with SIGABRT
(gloo `EnforceNotMet`, exit -6). Ranks 1-3 complete 0 steps: they either
raise at the 5 s timeout set here or hang past it and are killed at 15 s.
With torch's default process-group timeout of 1,800 s, that is a 30-minute
stall.

**FINDING: PyTorch's DDP without the flag fails fast rather than
deadlocking.** The doc says "the allreduce deadlocks". Rank 0, the rank that
skipped, finishes step 0 and raises "Expected to have finished reduction in
the prior iteration" at step 1. The other ranks block in their bucket
all-reduce until the timeout.

Structure: `launch()` starts this file once per rank and returns each rank's
exit code ("hung" if it had to be killed) and JSON; `worker()` trains one
mode and reports its parameters or the exception it hit.
"""

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

PHASE, LESSON, WS, STEPS, BATCH, LR = "19-capstone-projects", "77-data-parallel-ddp", 4, 6, 8, 0.05
ERRORS = (("Timed out waiting 5000ms", "5s timeout"), ("Expected to have finished reduction", "unfinished reduction"))


def launch(ws, mode, timeout=15):
    """Run this file as `ws` gloo ranks; return [(exit code or "hung", JSON or None)], killing stragglers."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        argv = [sys.executable, __file__, "--rank", str(ws), str(s.getsockname()[1]), mode]
    pipe = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True}
    procs = [subprocess.Popen(argv, env=os.environ | {"RANK": str(r)}, **pipe) for r in range(ws)]
    out = []
    for p in procs:
        try:
            text, _ = p.communicate(timeout=timeout)
            out.append((p.returncode, json.loads(text.splitlines()[-1]) if text.strip() else None))
        except subprocess.TimeoutExpired:
            p.kill()
            p.communicate()
            out.append(("hung", None))
    return out


def forward(model, x, skip):
    return model.net[4](model.net[:2](x) if skip else model.net[:4](x))  # [:2] = first Linear + ReLU


def zero_unused(module):
    for p in (p for p in module.parameters() if p.grad is None):
        p.grad = torch.zeros_like(p)


def loss_at(data, step, rank, run):
    at = (step * WS + rank) * BATCH
    return torch.nn.functional.mse_loss(run(data[0][at:at + BATCH], (step + rank) % 4 == 0), data[1][at:at + BATCH])


def worker(rank, ws, port, mode):
    ref = parity.load_reference(PHASE, LESSON, "main")
    os.environ["GLOO_SOCKET_IFNAME"] = ref._loopback_iface()
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=ws,
                            timeout=datetime.timedelta(seconds=5))
    torch.manual_seed(ref.SEED)
    model = ref.MiniMLP()
    ddp, run = ref.DistributedDataParallel(model, ws), lambda x, skip: forward(model, x, skip)
    sync = {"flag": lambda: (zero_unused(model), ddp.sync_grads()), "lesson": ddp.sync_grads}.get(mode, lambda: None)
    if mode.startswith("torch"):  # torch's DDP syncs in backward itself
        model.forward = run
        run = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=mode == "torch_flag")
    opt, done = torch.optim.SGD(model.parameters(), lr=LR), 0
    data = ref.make_dataset(ref.SEED + 1000, WS * BATCH * STEPS)
    try:
        for done in range(STEPS):
            opt.zero_grad(set_to_none=True)
            loss_at(data, done, rank, run).backward()
            sync()
            opt.step()
    except RuntimeError as exc:
        return {"steps": done, "error": next((tag for key, tag in ERRORS if key in str(exc)), str(exc)[:80])}
    return {"steps": STEPS, "params": torch.cat([p.detach().flatten() for p in model.parameters()]).tolist()}


def baseline(ref):
    torch.manual_seed(ref.SEED)
    model = ref.MiniMLP()
    opt, data = torch.optim.SGD(model.parameters(), lr=LR), ref.make_dataset(ref.SEED + 1000, WS * BATCH * STEPS)
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        for rank in range(WS):
            (loss_at(data, step, rank, lambda x, s: forward(model, x, s)) / WS).backward()
        opt.step()
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runs, base = {mode: launch(WS, mode) for mode in ("flag", "torch_flag", "lesson", "torch")}, baseline(ref)
    gap = {m: max(torch.tensor(o["params"]).sub(base).abs().max().item() for _, o in runs[m]) for m in ("flag", "torch_flag")}
    return {"gap": gap, "outcomes": {m: [(c, o and o["steps"], o and o.get("error")) for c, o in r] for m, r in runs.items()},
            "default_timeout_s": dist.constants.default_pg_timeout.total_seconds(), "doc": parity.doc_text(PHASE, LESSON)}


def verify(result):
    o, g, stuck = result["outcomes"], result["gap"], {(0, 0, "5s timeout"), ("hung", None, None)}
    return [
        practice.Check("ANSWER: zero-filling unused grads before sync_grads matches single-process and torch's flag",
            g["flag"] < 1e-6 and g["torch_flag"] < 1e-6 and o["flag"] == o["torch_flag"] == [(0, STEPS, None)] * WS,
            f"max |param - baseline| after {STEPS} steps {g}; every rank ran {STEPS} steps in both"),
        practice.Check("FINDING: without the flag the lesson's wrapper kills one rank and strands the other 3",
            o["lesson"][0] == (-6, None, None) and set(o["lesson"][1:]) <= stuck and result["default_timeout_s"] == 1800,
            f"(exit code, steps done, error) per rank {o['lesson']}; rank 0 SIGABRTs on a gloo size mismatch, the rest "
            f"time out at 5 s or hang past it (killed at 15 s); torch's default timeout is {result['default_timeout_s']} s"),
        practice.Check("FINDING: PyTorch's DDP does not deadlock either: the skipping rank raises a named error",
            o["torch"][0] == (0, 1, "unfinished reduction") and set(o["torch"][1:]) <= stuck and "allreduce deadlocks" in result["doc"],
            f"torch DDP without the flag, per rank {o['torch']}; the doc says the allreduce deadlocks"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    if sys.argv[1:2] == ["--rank"]:
        raise SystemExit(print(json.dumps(worker(int(os.environ["RANK"]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]))))
    raise SystemExit(practice.selfcheck(globals()))
