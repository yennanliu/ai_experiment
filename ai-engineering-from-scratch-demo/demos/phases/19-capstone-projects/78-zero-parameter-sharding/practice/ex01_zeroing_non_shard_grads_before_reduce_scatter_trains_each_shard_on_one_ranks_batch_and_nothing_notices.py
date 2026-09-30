"""Exercise 1 — zeroing the non-shard gradient before the reduce_scatter trains each shard on one rank's batch, and nothing notices.

    Extend to ZeRO-2 by sharding gradients: each rank only stores the gradient for its shard, achieved by zeroing out the non-shard portion after backward.

Reading of the exercise: the lesson's own run (seed 13, `MiniMLP` with
1,732 parameters, 4 gloo ranks, batch 8 per rank, 20 Adam steps, lr 0.05)
is driven three ways through the lesson's unmodified `ZeroOptimizer`:
ZeRO-1 as shipped; the recipe read literally, zeroing every gradient
element outside this rank's shard right after `backward()` and then
calling `step()`; and ZeRO-2 done after the reduce_scatter, where
`ref.dist.reduce_scatter` is wrapped to set every `p.grad` to None as soon
as the rank's summed shard has arrived. "Stores" is the gradient bytes a
rank holds right after the reduce_scatter.

**ANSWER: sharding the gradient after the reduce_scatter is ZeRO-2.** It
ends bit-identical to ZeRO-1 (max parameter difference 0.0 on every rank)
and a rank holds 1,732 bytes of gradient, its summed shard, instead of
8,660 (the full 6,928-byte gradient plus that shard). Backward still
builds the full gradient first; production ZeRO-2 frees it bucket by
bucket during backward, which this per-step wrap cannot show.

**FINDING: the literal recipe is wrong if the zeroing runs before the
reduce_scatter.** The reduce_scatter sums shard r over all ranks, and the
other ranks have just zeroed shard r. On all 20 steps, on every rank, what
comes back is exactly the rank's own gradient: each shard learns from one
rank's 8 samples, not the global 32. The parameters end 0.907 away from
ZeRO-1. It also stores nothing less: zeroing in place keeps the full
6,928-byte buffer.

**FINDING: the lesson's own checks cannot see the bug.** The broken run
still has all 4 ranks bit-identical (spread 0.0), because all_gather
rebuilds the same model everywhere, and its loss still falls, 10.119 to
2.786 (ZeRO-1: 2.48). Those are the two things `tests/test_zero.py` asserts.

Structure: `launch()` starts this file once per rank as a subprocess with a
90 s timeout and kills stragglers; `train()` is the lesson's worker loop
with the variant switched in. Expected output: three PASS checks.
"""

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
MODES = ("zero1", "zero_then_scatter", "scatter_then_free")


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


def zero_non_shard_(model, opt):
    """The exercise's recipe, literally: keep only this rank's slice of every gradient, zero the rest."""
    keep = torch.zeros(opt.total).index_fill_(0, torch.arange(opt.shard_start, opt.shard_end), 1)
    for p, mask in zip(model.parameters(), keep.split([p.numel() for p in model.parameters()])):
        p.grad.mul_(mask.view_as(p))


def train(ref, rank, ws, mode, steps):
    """The lesson's _zero_worker loop, with the ZeRO-2 variant under test switched in."""
    torch.manual_seed(ref.SEED)
    model = ref.MiniMLP()
    opt, seen = ref.ZeroOptimizer(model, world_size=ws, rank=rank, lr=0.05), {"own_only": 0, "held": 0}

    def scatter(out, chunks, op):
        dist.reduce_scatter(out, chunks, op=op)
        seen["own_only"] += torch.equal(out, chunks[rank])  # nothing but this rank's own gradient came back
        if mode == "scatter_then_free":  # ZeRO-2 proper: once the shard is summed, drop the full gradient
            model.zero_grad(set_to_none=True)
        grads = [p.grad.untyped_storage().nbytes() for p in model.parameters() if p.grad is not None]
        seen["held"] = sum(grads) + out.numel() * out.element_size()

    (x_all, y_all), losses = ref.make_dataset(ref.SEED + 1000, n_total=ws * ref.BATCH * steps), []
    ref.dist = types.SimpleNamespace(**vars(dist) | {"reduce_scatter": scatter})
    for step in range(steps):
        lo = step * ws * ref.BATCH + rank * ref.BATCH
        opt.zero_grad()
        (loss := torch.nn.functional.mse_loss(model(x_all[lo : lo + ref.BATCH]), y_all[lo : lo + ref.BATCH])).backward()
        if mode == "zero_then_scatter":
            zero_non_shard_(model, opt)
        opt.step()
        losses.append(loss.item())
    return {"losses": losses, "flat": ref.gather_flat_params(model).tolist(), **seen}


def worker(rank, ws, port, steps):
    ref = parity.load_reference(PHASE, LESSON, "main")
    os.environ["GLOO_SOCKET_IFNAME"] = ref._loopback_iface()
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=int(rank), world_size=int(ws))
    out = {mode: train(ref, int(rank), int(ws), mode, int(steps)) for mode in MODES}
    dist.destroy_process_group()
    return out


def summarize(ranks, mode):
    r0, flat = ranks[0][mode], ranks[0][mode]["flat"]
    spread = max(max(abs(x - y) for x, y in zip(r[mode]["flat"], flat)) for r in ranks)
    vs_zero1 = max(abs(x - y) for x, y in zip(flat, ranks[0]["zero1"]["flat"]))
    loss = [round(r0["losses"][0], 3), round(r0["losses"][-1], 3)]
    return {"spread": spread, "vs_zero1": vs_zero1, "held": r0["held"], "loss": loss,
            "own_only": min(r[mode]["own_only"] for r in ranks)}


def solve():
    ranks = launch(4, 20)
    return {"numel": len(ranks[0]["zero1"]["flat"])} | {m: summarize(ranks, m) for m in MODES}


def verify(result):
    z1, lit, z2 = (result[m] for m in MODES)
    return [
        practice.Check("ANSWER: sharding the gradient after the reduce_scatter is ZeRO-2",
            (z2["vs_zero1"], z2["spread"], z1["held"], z2["held"], result["numel"]) == (0.0, 0.0, 8660, 1732, 1732),
            f"max |ZeRO-2 - ZeRO-1| {z2['vs_zero1']}; gradient bytes held: ZeRO-1 {z1['held']}, ZeRO-2 {z2['held']}"),
        practice.Check("FINDING: the literal recipe is wrong if the zeroing runs before the reduce_scatter",
            (z1["own_only"], lit["own_only"], round(lit["vs_zero1"], 3), lit["held"]) == (0, 20, 0.907, 8660),
            f"steps whose summed shard is the rank's own gradient {lit['own_only']}/20 (ZeRO-1 {z1['own_only']}); "
            f"max |literal - ZeRO-1| {lit['vs_zero1']:.4f}; bytes held {lit['held']}"),
        practice.Check("FINDING: the lesson's own checks cannot see the bug",
            (lit["spread"], lit["loss"], z1["loss"]) == (0.0, [10.119, 2.786], [10.119, 2.48]),
            f"literal: rank spread {lit['spread']}, first/last loss {lit['loss']}; ZeRO-1 loss {z1['loss']}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    if sys.argv[1:2] == ["--rank"]:
        raise SystemExit(print(json.dumps(worker(os.environ["RANK"], *sys.argv[2:]))))
    raise SystemExit(practice.selfcheck(globals()))
