"""Exercise 3 — on gloo ZeRO-1's comm takes 1.9-3.8x DDP's, so "memory wins, throughput holds" does not hold.

    Measure the per-step wall-clock time of vanilla DDP versus ZeRO-1 and decompose into forward, backward, comm.

Reading of the exercise: both run on 4 gloo ranks on localhost, same seed,
same data, 40 steps of Adam (lr 0.05), at two sizes: the lesson's
`MiniMLP` (hidden 32, 1,732 parameters) and the same MLP at hidden 1,024
(1.07M parameters). ZeRO-1 is the lesson's `ZeroOptimizer`, with its
`reduce_scatter` and `all_gather` wrapped to time them. Vanilla DDP is
one `all_reduce` of the flattened gradient, then `torch.optim.Adam` on the
full state. Each step is split into forward, backward, comm, and
"optimizer" (the step minus comm). Numbers are rank 0's median over 5
interleaved blocks, ms per step. Only ratios with a wide margin are
asserted; the exact times vary from run to run.

**ANSWER: comm is most of the step, and ZeRO-1's comm is the larger.**
One run, ms per step (times move with machine load; the ratios hold):

| hidden | run | forward | backward | comm | optimizer |
|---|---|---:|---:|---:|---:|
| 32 | DDP | 0.02 | 0.05 | 0.40 | 0.13 |
| 32 | ZeRO-1 | 0.02 | 0.05 | 1.51 | 0.08 |
| 1,024 | DDP | 1.29 | 0.94 | 4.78 | 5.00 |
| 1,024 | ZeRO-1 | 1.24 | 1.47 | 9.41 | 4.12 |

On the lesson's model, forward plus backward is under 0.1 ms and comm is
over three quarters of a ZeRO-1 step. Both runs reach the same parameters
(max difference 4.6e-7 at hidden 32).

**FINDING: the doc's "Memory wins, throughput holds" is false on gloo.**
Over 8 runs, ZeRO-1's reduce_scatter plus all_gather took 2.5x to 3.8x
DDP's single all_reduce on the small model and 1.9x to 2.7x at 1.07M
parameters; the whole ZeRO-1 step was 1.3x to 2.8x slower. The check
asserts only wide margins: comm ratio above 1.3x at both sizes and a
slower whole step on the lesson's model. The doc's claim
that the wire traffic is the same assumes a ring algorithm. On the gloo
build installed here, `reduce_scatter` alone is slower than a full
`all_reduce` (exercise 5 measures this). Sharding Adam does not win the
time back: ZeRO-1 steps a quarter of the parameters, but flattening,
padding and gathering leave its "optimizer" time only 10-40% below DDP's.

Structure: `launch()` starts this file once per rank as a subprocess with a
90 s timeout and kills stragglers; `run()` times one 40-step run.
Expected output: two PASS checks.
"""

import json
import os
import socket
import subprocess
import sys
import statistics
import time
import types

from harness import parity, practice

try:
    import torch
    import torch.distributed as dist
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "78-zero-parameter-sharding"
STEPS, BLOCKS = 40, 5


def launch(ws, *args, timeout=90):
    """Run this file as `ws` gloo ranks in subprocesses; return each rank's JSON, kill stragglers."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        argv = [sys.executable, __file__, "--rank", str(ws), str(s.getsockname()[1]), *map(str, args)]
    procs = [subprocess.Popen(argv, env=os.environ | {"RANK": str(r)}, stdout=-1, stderr=-1, text=True) for r in range(ws)]
    try:
        outs = [p.communicate(timeout=timeout) for p in procs]
    finally:
        [p.kill() for p in procs]
    if any(p.returncode for p in procs):
        raise RuntimeError(f"a rank failed: {[err[-300:] for _, err in outs]}")
    return [json.loads(out.splitlines()[-1]) for out, _ in outs]


def ddp_step(ref, model, opt, ws, clock):
    """Vanilla DDP: one all_reduce of the flat gradient, then full-state Adam on every rank."""
    flat = ref.gather_flat_grads(model)
    clock("comm", lambda: dist.all_reduce(flat))
    for p, g in zip(model.parameters(), flat.div_(ws).split([p.numel() for p in model.parameters()])):
        p.grad.copy_(g.view_as(p))
    opt.step()


def run(ref, rank, ws, kind, hid):
    """`STEPS` training steps; returns mean ms per step in each phase and the final parameters."""
    torch.manual_seed(ref.SEED)
    model = ref.MiniMLP(hid_dim=hid)
    times = {"forward": [], "backward": [], "comm": [], "optimizer": []}

    def clock(name, fn):
        start = time.perf_counter()
        out = fn()
        times[name].append(time.perf_counter() - start)
        return out

    timed = {n: lambda *a, n=n, **k: clock("comm", lambda: getattr(dist, n)(*a, **k)) for n in ("reduce_scatter", "all_gather")}
    ref.dist = types.SimpleNamespace(**vars(dist) | timed)
    zero1 = kind == "zero1"
    opt = ref.ZeroOptimizer(model, ws, rank, lr=0.05) if zero1 else torch.optim.Adam(model.parameters(), lr=0.05)
    x_all, y_all = ref.make_dataset(ref.SEED + 1000, n_total=ws * ref.BATCH * STEPS)
    for step in range(STEPS):
        lo = step * ws * ref.BATCH + rank * ref.BATCH
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(clock("forward", lambda: model(x_all[lo:lo + ref.BATCH])), y_all[lo:lo + ref.BATCH])
        clock("backward", loss.backward)
        comm_before = sum(times["comm"])
        clock("optimizer", opt.step if zero1 else lambda: ddp_step(ref, model, opt, ws, clock))
        times["optimizer"][-1] -= sum(times["comm"]) - comm_before  # the step minus its comm
    return {k: sum(v) / STEPS * 1e3 for k, v in times.items()}, ref.gather_flat_params(model).tolist()


def worker(rank, ws, port, hid):
    ref = parity.load_reference(PHASE, LESSON, "main")
    os.environ["GLOO_SOCKET_IFNAME"] = ref._loopback_iface()
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=int(rank), world_size=int(ws))
    blocks = {"ddp": [], "zero1": []}
    for _ in range(BLOCKS):  # interleaved, so drift hits both alike
        for kind in blocks:
            blocks[kind].append(run(ref, int(rank), int(ws), kind, int(hid)))
    dist.destroy_process_group()
    ms = {k: {ph: round(statistics.median(b[0][ph] for b in v), 4) for ph in v[0][0]} for k, v in blocks.items()}
    gap = max(abs(a - b) for a, b in zip(blocks["ddp"][0][1], blocks["zero1"][0][1]))
    return {"ms": ms, "gap": gap}


def solve():
    return {str(hid): launch(4, hid)[0] for hid in (32, 1024)}


def answer(result):
    small, z = result["32"]["ms"], result["32"]["ms"]["zero1"]
    ok = z["comm"] > sum(z.values()) / 2 and all(v["forward"] + v["backward"] < v["comm"] for v in small.values())
    return practice.Check("ANSWER: comm is most of the step, and ZeRO-1's comm is the larger",
        ok and result["32"]["gap"] < 1e-5,
        f"ms per step, hidden 32: {small}; hidden 1024: {result['1024']['ms']}; "
        f"max |DDP - ZeRO-1| param {result['32']['gap']:.2g}")


def finding(result):
    ratio = {h: round(result[h]["ms"]["zero1"]["comm"] / result[h]["ms"]["ddp"]["comm"], 2) for h in result}
    step = {h: {k: round(sum(v.values()), 3) for k, v in result[h]["ms"].items()} for h in result}
    doc = " ".join(parity.doc_text(PHASE, LESSON).split())
    return practice.Check("FINDING: the doc's 'Memory wins, throughput holds' is false on gloo",
        "Memory wins, throughput holds." in doc and min(ratio.values()) > 1.3 and step["32"]["zero1"] > step["32"]["ddp"],
        f"ZeRO-1 comm / DDP comm {ratio}; whole step ms {step}")


def verify(result):
    return [answer(result), finding(result)]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    if sys.argv[1:2] == ["--rank"]:
        raise SystemExit(print(json.dumps(worker(os.environ["RANK"], *sys.argv[2:]))))
    raise SystemExit(practice.selfcheck(globals()))
