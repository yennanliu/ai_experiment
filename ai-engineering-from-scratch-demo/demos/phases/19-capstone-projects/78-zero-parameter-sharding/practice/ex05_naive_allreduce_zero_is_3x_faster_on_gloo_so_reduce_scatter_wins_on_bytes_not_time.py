"""Exercise 5 — on gloo the "naive" allreduce is 2-3x faster than reduce_scatter, so the defence rests on bytes, not time.

    Implement a "naive ZeRO" with allreduce instead of reduce_scatter, measure the wire-time difference. Defend the reduce_scatter choice with numbers.

Reading of the exercise: "naive ZeRO" keeps the lesson's `ZeroOptimizer`
and swaps only its gradient collective: `ref.dist.reduce_scatter` is
replaced by an `all_reduce` of the whole padded gradient, after which the
rank copies out its own chunk. "Wire time" is the rank-0 time spent in that
collective per step, median over 7 interleaved repeats of the lesson's
20-step, 4-rank gloo run. The same two collectives are also timed on a
1M-float gradient (256K floats per shard), fastest of 7 blocks of 10
calls. Bytes on the wire cannot be counted from Python, so they are
computed with the pipelined-ring volumes the ZeRO paper uses: Psi elements
each for a reduce-scatter and an all-gather, 2 Psi for an all-reduce.

**ANSWER: naive ZeRO trains the same model and, on this gloo, is faster.**
Its parameters match ZeRO-1 to 2.4e-7, a floating-point summation-order
difference. On the lesson's model its gradient collective takes about 0.4
ms per step against about 1.3 ms for `reduce_scatter`, 2.5x to 3.2x
faster over 8 runs. At 1M floats the two are close (naive 1.02x to 1.22x
faster). Timing does not favour reduce_scatter here.

**FINDING: the defence is bytes, and it holds only where bandwidth is the
limit.** Per rank per step (reduce-scatter or all-reduce, then the
all-gather):

| gradient | ZeRO-1 | naive | naive / ZeRO-1 |
|---|---:|---:|---:|
| lesson model (1,732 floats) | 10,392 B | 15,588 B | 1.5x |
| 1M floats | 6.29 MB | 9.44 MB | 1.5x |

ZeRO-1's 10,392 bytes equal DDP's one all-reduce, as the doc's "Net wire
is identical to DDP" says. The gradient collective alone sends 2x as many
bytes in the naive version, and the naive
version reduces a full-size sum of which the rank keeps a quarter. That is
the paper's case, section 7.1 ("the standard DP incurs 2 Psi data
movement"), and it is what decides a bandwidth-bound NCCL ring. Gloo on
loopback is latency-bound: here the extra bytes cost nothing measurable,
while this build's `reduce_scatter` carries a fixed overhead. (Source:
arXiv:1910.02054, https://arxiv.org/pdf/1910.02054, section 7.1, read
2026-09-29.)

Structure: `launch()` starts this file once per rank as a subprocess with a
90 s timeout and kills stragglers; `naive_scatter()` is the naive
collective; `ring_bytes()` is the paper's volume model.
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
MODES, STEPS, BLOCKS, BIG = ("zero1", "naive"), 20, 7, 1 << 18


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


def naive_scatter(rank, out, chunks, op=dist.ReduceOp.SUM):
    """'naive ZeRO': all_reduce the whole padded gradient, then keep this rank's chunk."""
    dist.all_reduce(full := torch.cat(chunks), op=op)
    out.copy_(full.chunk(len(chunks))[rank])


def train(ref, rank, ws, mode):
    """The lesson's run; returns rank-local ms spent in the gradient collective per step, and the params."""
    torch.manual_seed(ref.SEED)
    opt, spent = ref.ZeroOptimizer((model := ref.MiniMLP()), world_size=ws, rank=rank, lr=0.05), []
    scatter = dist.reduce_scatter if mode == "zero1" else lambda out, chunks, op: naive_scatter(rank, out, chunks, op)
    def timed(*args, **kwargs):
        start = time.perf_counter()
        scatter(*args, **kwargs)
        spent.append(time.perf_counter() - start)
    ref.dist = types.SimpleNamespace(**vars(dist) | {"reduce_scatter": timed})
    data = ref.make_dataset(ref.SEED + 1000, n_total=ws * ref.BATCH * STEPS)
    for x, y in zip(*(t.split(ref.BATCH)[rank::ws] for t in data)):  # the lesson's per-rank, per-step offsets
        opt.zero_grad()
        torch.nn.functional.mse_loss(model(x), y).backward()
        opt.step()
    return sum(spent) / STEPS * 1e3, ref.gather_flat_params(model).tolist()


def big_collectives(rank, ws):
    """The same two choices on a 1M-float gradient (256K floats per shard): best ms per call."""
    chunks, out = list(torch.randn(BIG * ws, generator=torch.Generator().manual_seed(rank)).chunk(ws)), torch.zeros(BIG)
    ways, best = {"zero1": lambda: dist.reduce_scatter(out, chunks), "naive": lambda: naive_scatter(rank, out, chunks)}, {}
    for name, fn in list(ways.items()) * BLOCKS:  # interleaved
        dist.barrier()
        start = time.perf_counter()
        [fn() for _ in range(10)]
        best[name] = min(best.get(name, 1e9), (time.perf_counter() - start) / 10 * 1e3)
    return best


def worker(rank, ws, port):
    ref, rank, ws = parity.load_reference(PHASE, LESSON, "main"), int(rank), int(ws)
    os.environ["GLOO_SOCKET_IFNAME"] = ref._loopback_iface()
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=ws)
    runs = [[train(ref, rank, ws, m) for m in MODES] for _ in range(BLOCKS)]  # interleaved against drift
    big = big_collectives(rank, ws)
    dist.destroy_process_group()
    return {"ms": {m: statistics.median(b[i][0] for b in runs) for i, m in enumerate(MODES)},
            "flat": {m: runs[0][i][1] for i, m in enumerate(MODES)}, "big": big}


def ring_bytes(numel, ws):
    """Bytes each rank sends per step in a ring (ZeRO paper 7.1: Psi per reduce-scatter or all-gather)."""
    s = 4 * numel * (ws - 1) // ws  # fp32 bytes of one Psi-sized pass: rs + ag, all_reduce + ag, all_reduce
    return {"zero1": s + s, "naive": 2 * s + s, "ddp": 2 * s}


def solve():
    r0 = launch(4)[0]
    return {
        "gap": max(abs(a - b) for a, b in zip(r0["flat"]["zero1"], r0["flat"]["naive"])),
        "ms": {m: round(v, 3) for m, v in r0["ms"].items()}, "big": {m: round(v, 3) for m, v in r0["big"].items()},
        "doc_claim": "Net wire is identical to DDP" in parity.doc_text(PHASE, LESSON),
        "wire": ring_bytes(len(r0["flat"]["zero1"]), 4), "wire_big": ring_bytes(BIG * 4, 4),  # P = 1,732 floats
    }


def verify(r):
    speed, big = round(r["ms"]["zero1"] / r["ms"]["naive"], 2), round(r["big"]["naive"] / r["big"]["zero1"], 2)
    return [
        practice.Check("ANSWER: naive ZeRO trains the same model and, on this gloo, is faster",
            r["gap"] < 1e-5 and speed > 1.5 and big < 1.25,
            f"max |naive - ZeRO-1| param {r['gap']:.2g}; gradient collective ms per step {r['ms']} "
            f"(naive {speed}x faster); 1M floats ms per call {r['big']} (naive / ZeRO-1 {big})"),
        practice.Check("FINDING: the defence is bytes, and it holds only where bandwidth is the limit",
            r["wire"] == {"zero1": 10392, "naive": 15588, "ddp": 10392}
            and r["wire_big"] == {"zero1": 6291456, "naive": 9437184, "ddp": 6291456} and r["doc_claim"],
            f"ring bytes sent per rank per step, lesson model {r['wire']}, 1M floats {r['wire_big']}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    if sys.argv[1:2] == ["--rank"]:
        raise SystemExit(print(json.dumps(worker(os.environ["RANK"], *sys.argv[2:]))))
    raise SystemExit(practice.selfcheck(globals()))
