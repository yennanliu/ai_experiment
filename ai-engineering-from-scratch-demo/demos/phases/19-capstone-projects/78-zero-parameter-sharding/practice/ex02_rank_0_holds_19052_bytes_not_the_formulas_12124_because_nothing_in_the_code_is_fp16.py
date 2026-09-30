"""Exercise 2 — rank 0 really holds 19,052 bytes, not the 12,124 the lesson's formula prints, because nothing in the code is fp16.

    Add a memory profiler that prints actual fp32 byte usage on rank 0 versus the formula prediction.

Reading of the exercise: "the formula prediction" is the lesson's own
`memory_table(P, 4)` for its `MiniMLP` (P = 1,732) at world size 4.
"Actual" is measured on rank 0 of a real 4-rank gloo run after one
`ZeroOptimizer.step()`: the bytes of the distinct storages behind the
parameters, the gradients and the optimiser's three shards, and, with a
`TorchDispatchMode` counter, the bytes of every new tensor the step itself
allocates. Vanilla DDP's actual is measured the same way on
`torch.optim.Adam` after one step.

**ANSWER: rank 0 holds 19,052 bytes; the formula predicts 12,124.**

| term | formula | actual |
|---|---:|---:|
| params | 2P = 3,464 | 4P = 6,928 |
| grads | 2P = 3,464 | 4P = 6,928 |
| master + m + v shards | 12P/4 = 5,196 | 5,196 |

The optimiser shards match exactly. The 6,928-byte gap is the formula's
fp16 params and grads: every tensor in the run is float32. With 4 bytes
for those two terms the formula gives 19,052, exact to the byte.

**FINDING: the real saving is 31.2%, not the 56.2% the demo prints.**
Vanilla DDP with `torch.optim.Adam` measures 27,712 bytes, which is also the
formula's vanilla row. In fp32 Adam needs no separate master copy, and the
ZeRO row still pays for one. After the step, rank 0's `master_shard` is
bit-identical to its slice of the parameters it copies back into: a
1,732-byte duplicate. The table's heading says "per-rank optimiser memory",
but its rows include params and grads.

**FINDING: `step()` allocates 34,640 bytes of temporaries (20P) to save
optimiser state.** That is 6.7x the 5,196-byte shard and more than vanilla
DDP's whole per-rank state. Flattening and padding the gradient costs 4P
twice, gathering and concatenating the parameters costs 4P twice more, and
the shard-sized received gradient, `m_hat`, `v_hat` and the square root make
the last 4P.
At N = 3 the padding shows up in the shards too: every rank holds 6,936
bytes against 12P/3 = 6,928.

Structure: `launch()` starts this file once per rank as a subprocess with a
90 s timeout and kills stragglers; `worker()` is the profiler.
Expected output: three PASS checks.
"""

import json
import os
import re
import socket
import subprocess
import sys

from harness import parity, practice

try:
    import torch
    import torch.distributed as dist
    from torch.utils._pytree import tree_leaves
    from torch.utils._python_dispatch import TorchDispatchMode
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "78-zero-parameter-sharding"


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


class AllocCounter(TorchDispatchMode):
    bytes = 0  # of every new tensor storage an aten op creates while active

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        seen = {t.untyped_storage().data_ptr() for t in tree_leaves((args, kwargs)) if isinstance(t, torch.Tensor)}
        out = func(*args, **(kwargs or {}))
        new = [t for t in tree_leaves(out) if isinstance(t, torch.Tensor) and t.untyped_storage().data_ptr() not in seen]
        self.bytes += sum(t.untyped_storage().nbytes() for t in new)
        return out


def nbytes(*tensors):
    return sum({t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors}.values())


def one_step(ref, opt_for, n):
    torch.manual_seed(ref.SEED)
    model = ref.MiniMLP()
    opt = opt_for(model)
    x, y = ref.make_dataset(ref.SEED + 1000, n_total=n)
    torch.nn.functional.mse_loss(model(x[: ref.BATCH]), y[: ref.BATCH]).backward()
    return model, opt


def worker(rank, ws, port):
    """The profiler, on rank `rank` of a real gloo group after one ZeroOptimizer.step()."""
    ref, rank, ws = parity.load_reference(PHASE, LESSON, "main"), int(rank), int(ws)
    os.environ["GLOO_SOCKET_IFNAME"] = ref._loopback_iface()
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=ws)
    model, opt = one_step(ref, lambda m: ref.ZeroOptimizer(m, world_size=ws, rank=rank, lr=0.05), ws * ref.BATCH)
    with AllocCounter() as counter:
        opt.step()
    params, shards = list(model.parameters()), (opt.master_shard, opt.m_shard, opt.v_shard)
    own = ref.gather_flat_params(model)[opt.shard_start : opt.shard_end]
    dist.destroy_process_group()
    return {
        "dtypes": sorted({str(t.dtype) for t in params + list(shards)}), "params": nbytes(*params),
        "grads": nbytes(*[p.grad for p in params]), "shards": [nbytes(t) for t in shards],
        "shard_bytes": opt.shard_bytes(), "step_alloc": counter.bytes, "master_is_param_slice": torch.equal(shards[0], own),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    model, adam = one_step(ref, lambda m: torch.optim.Adam(m.parameters(), lr=0.05), ref.BATCH)
    adam.step()  # vanilla DDP's per-rank state, measured: params, grads, Adam's m and v
    state = [t for s in adam.state.values() for t in s.values() if t.dim() > 0]
    table = ref.memory_table(ref.flat_param_numel(model), 4)
    return {
        "p": ref.flat_param_numel(model), "rank0": launch(4)[0], "title": table.splitlines()[0],
        "ddp_actual": nbytes(*model.parameters()) + nbytes(*[p.grad for p in model.parameters()]) + nbytes(*state),
        "formula": [int(b) for b in re.findall(r"(\d+) bytes", table)],
        "formula_drop": float(re.search(r"drop: ([\d.]+)%", table).group(1)),
        "shard_bytes_n3": [ref.ZeroOptimizer(ref.MiniMLP(), 3, r).shard_bytes() for r in range(3)],
    }


def verify(result):
    r, z, p = result, result["rank0"], result["p"]
    actual = z["params"] + z["grads"] + z["shard_bytes"]
    drop = round(100 * (r["ddp_actual"] - actual) / r["ddp_actual"], 1)
    return [
        practice.Check("ANSWER: rank 0 holds 19,052 bytes; the formula predicts 12,124",
            (actual, r["formula"][1], z["shards"], z["dtypes"]) == (19052, 12124, [1732] * 3, ["torch.float32"])
            and actual - r["formula"][1] == 4 * p and actual == (4 + 4) * p + 12 * p // 4,
            f"params {z['params']} + grads {z['grads']} + master/m/v {z['shards']} = {actual} vs {r['formula'][1]}"),
        practice.Check("FINDING: the real saving is 31.2%, not the 56.2% the demo prints",
            (r["ddp_actual"], r["formula"][0], drop, r["formula_drop"], r["title"])
            == (27712, 27712, 31.2, 56.2, "per-rank optimiser memory:") and z["master_is_param_slice"],
            f"torch Adam DDP {r['ddp_actual']} bytes; drop {drop}% vs printed {r['formula_drop']}%; "
            f"master shard == param slice: {z['master_is_param_slice']}"),
        practice.Check("FINDING: step() allocates 34,640 bytes of temporaries (20P) to save optimiser state",
            z["step_alloc"] == 20 * p == 34640 and r["shard_bytes_n3"] == [6936] * 3,
            f"one step allocates {z['step_alloc']} bytes; shard bytes at N=3 {r['shard_bytes_n3']} vs {12 * p // 3}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    if sys.argv[1:2] == ["--rank"]:
        raise SystemExit(print(json.dumps(worker(os.environ["RANK"], *sys.argv[2:]))))
    raise SystemExit(practice.selfcheck(globals()))
