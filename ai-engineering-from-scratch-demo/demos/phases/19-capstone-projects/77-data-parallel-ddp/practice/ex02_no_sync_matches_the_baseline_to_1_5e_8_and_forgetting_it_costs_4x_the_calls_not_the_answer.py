"""Exercise 2 — no_sync matches the baseline to 1.5e-8, and forgetting it costs 4x the calls, not a wrong answer.

    Implement `no_sync()` as a context manager and verify gradient accumulation matches a single-process baseline over K microbatches.

Reading of the exercise: `no_sync(ddp)` wraps the lesson's own
`DistributedDataParallel`: inside the block its `sync_grads` is a no-op, on
exit the lesson's method is back. Each of 4 gloo ranks runs K = 4
microbatches of 8 rows per step, backward on loss / K, the first K - 1
inside `no_sync`, then one `sync_grads` and an SGD step (lr 0.05, 10 steps,
the lesson's `MiniMLP`, seed and `make_dataset`). The baseline is one
process that walks all 4 x 4 microbatches of a step with loss / 16, the
lesson's `reference_single_process` generalised to K. Two more runs: the
same loop with `sync_grads` after every microbatch (no_sync forgotten), and
PyTorch's own `DistributedDataParallel.no_sync()` on the same data.

**ANSWER: it matches.** After 10 steps the largest parameter difference to
the single-process baseline is 1.5e-8, while the parameters themselves
moved by 0.084. All 4 ranks hold bit-identical parameters. PyTorch's own
`no_sync()` lands on the same 1.5e-8.

**FINDING: forgetting `no_sync` wastes calls, not correctness.** The doc says
"Forget the manager and you allreduce K times for nothing". The call count
is right: 240 all-reduces instead of 60 per run. But the model is the same
one to 1.5e-8, because averaging an already averaged gradient changes
nothing. The result is not bit-identical to the no_sync run, only equal to
float rounding.

**FINDING: the lesson's promised "byte-equal parameter equivalence" does not
hold.** The doc's phrase is "byte-equal parameter equivalence after each step". In the lesson's own
setting (K = 1) the DDP parameters differ from the single-process ones by
1.5e-8 after 10 steps. The lesson's test only compares losses to 4 decimal
places and parameter norms to 5, so it cannot see this.

Structure: `launch()` starts this file once per rank as a subprocess with a
60 s timeout and kills stragglers; `train()` is the accumulation loop for
one mode (the baseline is its "single" mode in one process) and `worker()`
counts the lesson's `dist.all_reduce` calls around it.
"""

import contextlib
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

PHASE, LESSON, WS, K, STEPS, BATCH, LR = "19-capstone-projects", "77-data-parallel-ddp", 4, 4, 10, 8, 0.05


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


@contextlib.contextmanager
def no_sync(ddp):
    """Pause the lesson wrapper's gradient all-reduce for the body of the block."""
    ddp.sync_grads = lambda: None
    try:
        yield ddp
    finally:
        del ddp.sync_grads  # the instance attribute goes, the class method is back


def train(ref, mode, k, first, count):
    """Per step, backward on microbatches first..first+count-1 (of WS * k) with loss / count; (start, end)."""
    torch.manual_seed(ref.SEED)
    model = forward = ref.MiniMLP()
    pause, sync = contextlib.nullcontext, lambda: None  # "single": one process, nothing to sync
    if mode == "torch":
        forward = torch.nn.parallel.DistributedDataParallel(model)
        pause = forward.no_sync
    elif mode != "single":
        ddp = ref.DistributedDataParallel(model, WS)
        pause = pause if mode == "forgot" else lambda: no_sync(ddp)
        sync = lambda: ddp.sync_grads()  # noqa: E731 - looked up per call, so no_sync can pause it
    flat = lambda: torch.cat([p.detach().flatten() for p in model.parameters()])  # noqa: E731
    start, opt = flat(), torch.optim.SGD(model.parameters(), lr=LR)
    x_all, y_all = ref.make_dataset(ref.SEED + 1000, WS * k * BATCH * STEPS)
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        for i in range(count):
            at = (step * WS * k + first + i) * BATCH
            with pause() if i < count - 1 else contextlib.nullcontext():
                (torch.nn.functional.mse_loss(forward(x_all[at:at + BATCH]), y_all[at:at + BATCH]) / count).backward()
                sync()
        opt.step()
    return start, flat()


def worker(rank, ws, port):
    torch.set_num_threads(1)
    ref = parity.load_reference(PHASE, LESSON, "main")
    os.environ["GLOO_SOCKET_IFNAME"] = ref._loopback_iface()
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=ws,
                            timeout=datetime.timedelta(seconds=30))
    calls, real, out = [], ref.dist.all_reduce, {}
    ref.dist.all_reduce = lambda t, **kw: (calls.append(1), real(t, **kw))[1]
    for mode, k in (("no_sync", K), ("forgot", K), ("torch", K), ("k1", 1)):
        before = len(calls)
        out[mode] = {"params": train(ref, mode, k, rank * k, k)[1].tolist(), "calls": len(calls) - before}
    dist.destroy_process_group()
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ranks, (start, k4), (_, k1) = launch(WS), train(ref, "single", K, 0, WS * K), train(ref, "single", 1, 0, WS)
    r0 = ranks[0]
    gap = {m: max(torch.tensor(r[m]["params"]).sub(k1 if m == "k1" else k4).abs().max().item() for r in ranks) for m in r0}
    return {"gap": gap, "doc": parity.doc_text(PHASE, LESSON), "moved": float((k4 - start).abs().max()),
            "rank_differs": any(r[m]["params"] != r0[m]["params"] for r in ranks for m in r0),
            "calls": {m: v["calls"] for m, v in r0.items()},
            "same": r0["forgot"]["params"] == r0["no_sync"]["params"]}


def verify(result):
    g, c = result["gap"], result["calls"]
    return [
        practice.Check("ANSWER: no_sync accumulation over K = 4 matches the single-process baseline to < 1e-6",
                       g["no_sync"] < 1e-6 and not result["rank_differs"] and result["moved"] > 0.01,
                       f"max |param - baseline| {g}; ranks identical; params moved {result['moved']:.3f}"),
        practice.Check("FINDING: forgetting no_sync gives the same model with 4x the all-reduce calls",
                       c == {"no_sync": 60, "forgot": 240, "torch": 0, "k1": 60} and g["forgot"] < 1e-6 and not result["same"],
                       f"lesson all_reduce calls {c}; forgot == no_sync bit for bit: {result['same']}"),
        practice.Check("FINDING: the doc promises byte-equal parameters; the DDP path is off by float rounding",
                       "byte-equal parameter equivalence" in result["doc"] and 0 < g["k1"] < 1e-6 and 0 < g["no_sync"],
                       f"max gap in the lesson's K = 1 setup {g['k1']:.2g}, K = 4 {g['no_sync']:.2g}, torch {g['torch']:.2g}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    if sys.argv[1:2] == ["--rank"]:
        raise SystemExit(print(json.dumps(worker(*map(int, [os.environ["RANK"], *sys.argv[2:]])))))
    raise SystemExit(practice.selfcheck(globals()))
