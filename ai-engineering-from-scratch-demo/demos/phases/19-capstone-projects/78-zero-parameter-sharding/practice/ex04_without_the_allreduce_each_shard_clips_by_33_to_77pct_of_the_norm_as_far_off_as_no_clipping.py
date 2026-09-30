"""Exercise 4 — the allreduce is the whole fix: without it each shard clips by 33-77% of the norm and lands as far off as no clipping.

    Implement gradient clipping under ZeRO-1: the L2 norm must be computed across all shards via allreduce of the local norm squared.

Reading of the exercise: clipping goes between the lesson's
`reduce_scatter` and its Adam update, where each rank holds only its
summed shard. `ref.dist.reduce_scatter` is wrapped so that, once the shard
arrives, the rank squares and sums its averaged gradient, all-reduces that
one number, and scales the shard by min(1, max_norm / (norm + 1e-6)),
torch's own formula. The run is the lesson's (seed 13, 4 gloo ranks, batch
8 per rank, 20 Adam steps, lr 0.05) with max_norm 1.0. The target is a
single process on the same global batch of 32 using
`torch.nn.utils.clip_grad_norm_` and `torch.optim.Adam`. For contrast, the
same run is repeated clipping each shard by its own norm (no allreduce),
and with no clipping.

**ANSWER: clipping under ZeRO-1 matches torch's single-process clipping.**
After 20 steps the parameters are within 3.0e-7 of the target, and the
global norm each rank computes is within 4.8e-6 of what `clip_grad_norm_`
returns. The norm runs from 1.54 to 13.98, so all 20 steps clip. The
padding elements are zero and do not change the norm.

**FINDING: without the allreduce each rank clips by its own shard's norm,
which is 33-77% of the true norm.** On step 1 the four shards carry 0.331,
0.389, 0.378 and 0.772 of the global norm; their squares sum to 1. Each
shard is scaled by a different, too-large factor. The parameters end
0.413 from the target, about as far as not clipping at all (0.503).

**FINDING: the broken run is invisible to a rank-agreement check.** All
4 ranks end bit-identical in all three runs (spread 0.0), because the
all_gather hands every rank the same shards whatever each did to its own.
Only comparing against a single-process run shows the difference.

Structure: `launch()` starts this file once per rank as a subprocess with a
90 s timeout and kills stragglers; `clipping_scatter()` is the clipping.
Expected output: three PASS checks.
"""

import functools
import json
import os
import socket
import subprocess
import sys
import types

from harness import parity, practice

try:
    import torch
    import torch.distributed as dist
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "78-zero-parameter-sharding"
MODES, MAX_NORM, STEPS = ("clip", "shard_norm", "none"), 1.0, 20


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


def clipping_scatter(ws, mode, norms, out, chunks, op):
    """reduce_scatter, then clip the rank's summed shard by the global norm of the averaged gradient."""
    dist.reduce_scatter(out, chunks, op=op)
    sq = (out / ws).pow(2).sum()  # the lesson divides by world_size after this call
    if mode == "clip":
        dist.all_reduce(sq)  # the exercise's allreduce of the local norm squared
    norms.append((norm := sq.sqrt()).item())
    if mode != "none":
        out.mul_(torch.clamp(MAX_NORM / (norm + 1e-6), max=1.0))


def train(ref, rank, ws, mode):
    torch.manual_seed(ref.SEED)
    opt, norms = ref.ZeroOptimizer((model := ref.MiniMLP()), world_size=ws, rank=rank, lr=0.05), []
    ref.dist = types.SimpleNamespace(**vars(dist) | {"reduce_scatter": functools.partial(clipping_scatter, ws, mode, norms)})
    x_all, y_all = ref.make_dataset(ref.SEED + 1000, n_total=ws * ref.BATCH * STEPS)
    for x, y in zip(x_all.split(ref.BATCH)[rank::ws], y_all.split(ref.BATCH)[rank::ws]):  # the lesson's offsets
        opt.zero_grad()
        torch.nn.functional.mse_loss(model(x), y).backward()
        opt.step()
    return {"norms": norms, "flat": ref.gather_flat_params(model).tolist()}


def single_process(ref, ws, clip):
    """The target: one process, the global batch, torch's clip_grad_norm_, torch.optim.Adam."""
    torch.manual_seed(ref.SEED)
    opt, norms = torch.optim.Adam((model := ref.MiniMLP()).parameters(), lr=0.05), []
    x_all, y_all = ref.make_dataset(ref.SEED + 1000, n_total=ws * ref.BATCH * STEPS)
    for x, y in zip(x_all.split(ws * ref.BATCH), y_all.split(ws * ref.BATCH)):
        opt.zero_grad()
        torch.nn.functional.mse_loss(model(x), y).backward()
        norms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM if clip else float("inf")).item())
        opt.step()
    return {"norms": norms, "flat": ref.gather_flat_params(model).tolist()}


def worker(rank, ws, port):
    ref = parity.load_reference(PHASE, LESSON, "main")
    os.environ["GLOO_SOCKET_IFNAME"] = ref._loopback_iface()
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=int(rank), world_size=int(ws))
    out = {mode: train(ref, int(rank), int(ws), mode) for mode in MODES}
    dist.destroy_process_group()
    return out


def gap(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    r0, target, unclipped = (ranks := launch(4))[0], single_process(ref, 4, True), single_process(ref, 4, False)
    ratios = [round(r["shard_norm"]["norms"][0] / target["norms"][0], 3) for r in ranks]
    return {
        "vs_torch": {m: gap(r0[m]["flat"], target["flat"]) for m in MODES},
        "none_vs_unclipped": gap(r0["none"]["flat"], unclipped["flat"]), "norm_err": gap(r0["clip"]["norms"], target["norms"]),
        "clipped_steps": sum(n > MAX_NORM for n in target["norms"]), "norm_range": [round(f(target["norms"]), 2) for f in (min, max)],
        "shard_norm_ratio": ratios, "ratio_sq_sum": sum(x * x for x in ratios),
        "rank_spread": {m: max(gap(r[m]["flat"], r0[m]["flat"]) for r in ranks) for m in MODES},
    }


def verify(r):
    return [
        practice.Check("ANSWER: clipping under ZeRO-1 matches torch's single-process clipping",
            r["vs_torch"]["clip"] < 1e-5 and r["norm_err"] < 1e-4 and r["none_vs_unclipped"] < 1e-5
            and r["clipped_steps"] == STEPS and r["norm_range"] == [1.54, 13.98],
            f"max |ZeRO clip - torch clip| param {r['vs_torch']['clip']:.2g}, norm {r['norm_err']:.2g}; "
            f"{r['clipped_steps']}/{STEPS} steps clipped, norm range {r['norm_range']}"),
        practice.Check("FINDING: without the allreduce each rank clips by its own shard's norm (33-77% of it)",
            r["shard_norm_ratio"] == [0.331, 0.389, 0.378, 0.772] and abs(r["ratio_sq_sum"] - 1) < 1e-2 
            and (round(r["vs_torch"]["shard_norm"], 3), round(r["vs_torch"]["none"], 3)) == (0.413, 0.503),
            f"step-1 shard norm / global norm by rank {r['shard_norm_ratio']}; max |param - target| {r['vs_torch']}"),
        practice.Check("FINDING: the broken run is invisible to a rank-agreement check",
            set(r["rank_spread"].values()) == {0.0}, f"max parameter difference from rank 0 {r['rank_spread']}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    if sys.argv[1:2] == ["--rank"]:
        raise SystemExit(print(json.dumps(worker(os.environ["RANK"], *sys.argv[2:]))))
    raise SystemExit(practice.selfcheck(globals()))
