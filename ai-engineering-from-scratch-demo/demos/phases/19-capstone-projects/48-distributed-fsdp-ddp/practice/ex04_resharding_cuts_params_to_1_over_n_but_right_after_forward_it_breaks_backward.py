"""Exercise 4 — re-sharding cuts parameter memory to 1/N, but done right after forward it breaks backward.

    Implement the FSDP re-shard step: after the forward pass, replace the full tensor with the local shard again. Confirm per-rank memory drops.

Reading of the exercise: the model is the lesson's `make_model(32, 16, 4)`
(596 parameters, 2,384 bytes), seed 0, at world sizes 2, 3 and 4 over gloo.
`reshard()` pads each flattened parameter to a multiple of the world size,
keeps this rank's slice as `p.data` and drops the rest; `unshard()` is the
lesson's `all_gather` step, run for real before the forward. "Per-rank
memory" is the bytes of the storages the parameters hold, measured on every
rank before, during and after the forward. The forward output is compared
with an unsharded copy of the same model.

**ANSWER: per-rank parameter memory drops from 2,384 bytes to 1,192, 804
and 596 at world sizes 2, 3 and 4**, on every rank, and the output computed
from the gathered weights is bit-identical to the unsharded model's. At 3
ranks it is not exactly 1/N: each tensor is padded to a multiple of 3, so a
rank holds 201 floats against 198.7 (1.2% over), and the gathered tensor is
a view into a 603-float padded buffer (2,412 bytes). The doc says "the
memory win is exact".

**FINDING: re-sharding right after forward, as the exercise words it, makes
backward fail.** Autograd saved the parameter tensors for backward, and
swapping their `.data` for the shard changes what it finds there: backward
raises a shape-mismatch `RuntimeError` at all three world sizes. Gathering
again before backward (what production FSDP does in its pre-backward hook)
gives gradients bit-identical to the unsharded model. Those gradients are
full size on every rank, though, so at 4 ranks parameters plus gradients
drop from 4,768 to 2,980 bytes (37.5% less), not to a quarter.

**FINDING: the lesson's sketch never shards anything.**
`fsdp_round_trip_sketch` says it "keeps the per-rank memory at 1/world_size
of the model", but it leaves every parameter full size (2,384 bytes after the
call, on every rank) and only checks a gathered copy. It checks with
`torch.allclose` (relative tolerance 1e-5), not the equality the doc calls
"bit-equal".

Structure: `launch()` starts this file once per rank as a subprocess with a
60 s timeout and kills stragglers. Expected output: three PASS checks.
"""


from __future__ import annotations

import copy
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


def reshard(model, rank, ws):
    """Keep only this rank's padded slice of every parameter; return the full shapes."""
    shapes = [p.shape for p in model.parameters()]
    for p in model.parameters():
        per = -(-p.numel() // ws)
        padded = torch.cat([p.data.flatten(), p.data.new_zeros(per * ws - p.numel())])
        p.data = padded[rank * per : (rank + 1) * per].clone()
    return shapes


def unshard(model, shapes, ws):
    """The lesson's all_gather step: rebuild each full parameter from the N shards."""
    for p, shape in zip(model.parameters(), shapes):
        parts = [torch.empty_like(p.data) for _ in range(ws)]
        dist.all_gather(parts, p.data)
        p.data = torch.cat(parts)[: shape.numel()].view(shape)


def worker(rank, ws, port):
    ref = parity.load_reference(PHASE, LESSON, "main")
    ref.init_process_group(rank, ws, "gloo", port)
    torch.manual_seed(0)
    full, x = ref.make_model(32, 16, 4), torch.randn(8, 32)
    model, want = copy.deepcopy(full), full(x)
    want.pow(2).mean().backward()
    nbytes = lambda ts=None: sum(t.untyped_storage().nbytes() for t in ts or model.parameters())  # noqa: E731
    try:
        mem = {"sketch_ok": ref.fsdp_round_trip_sketch(model, ws, rank), "after_sketch": nbytes()}
        shapes = reshard(model, rank, ws)
        mem["sharded"] = nbytes()
        unshard(model, shapes, ws)
        mem["gathered"], out = nbytes(), model(x)
        reshard(model, rank, ws)  # the exercise's re-shard, right after forward
        mem["after_forward"], error = nbytes(), None
        try:
            out.pow(2).mean().backward()
        except RuntimeError as exc:
            error = str(exc).split("\n")[0]
        unshard(model, shapes, ws)  # the fix: gather again before backward
        model.zero_grad(set_to_none=True)
        model(x).pow(2).mean().backward()
        grads = [p.grad for p in model.parameters()]
        reshard(model, rank, ws)
        mem["params_plus_grads"] = nbytes() + nbytes(grads)
        same_grad = all(torch.equal(g, p.grad) for g, p in zip(grads, full.parameters()))
        return {"mem": mem, "same_out": torch.equal(out, want), "error": error, "same_grad": same_grad,
                "shape_error": bool(error) and ("size" in error or "shape" in error)}
    finally:
        ref.shutdown_process_group()


def collect():
    """Every rank at world sizes 2/3/4; a stage lists one value per world size if all ranks agree."""
    every = [r | {"ws": ws} for ws in (2, 3, 4) for r in launch(ws)]
    mem = {k: sorted({(r["ws"], r["mem"][k]) for r in every}) for k in every[0]["mem"]}
    return {k: [v for _, v in pairs] for k, pairs in mem.items()}, every


def solve():
    sketch = parity.load_reference(PHASE, LESSON, "main").fsdp_round_trip_sketch
    col, every = collect()
    flags = {k: all(r[k] for r in every) for k in ("same_out", "same_grad", "shape_error")}
    doc, names = parity.doc_text(PHASE, LESSON), sketch.__code__.co_names
    return {"col": col, "flags": flags, "errors": sorted({r["error"] for r in every}),
            "doc": ["memory win is exact" in doc, "bit-equal" in doc],
            "sketch": ["1/world_size" in sketch.__doc__, "allclose" in names, "equal" in names]}


def verify(result):
    r, col, ok = result, result["col"], result["flags"]
    return [
        practice.Check(
            "ANSWER: per-rank parameter memory drops from 2,384 bytes to 1,192 / 804 / 596",
            (col["sharded"], col["after_forward"], col["gathered"], ok["same_out"], r["doc"][0])
            == ([1192, 804, 596], [1192, 804, 596], [2384, 2412, 2384], True, True),
            f"bytes per rank at world 2/3/4 (one value per world size = identical on every rank): {col}",
        ),
        practice.Check(
            "FINDING: re-sharding right after forward makes backward fail",
            (ok["shape_error"], ok["same_grad"], col["params_plus_grads"][2]) == (True, True, 2980),
            f"backward errors {r['errors']}",
        ),
        practice.Check(
            "FINDING: the lesson's sketch never shards anything",
            (col["after_sketch"], col["sketch_ok"], r["doc"][1], r["sketch"])
            == ([2384] * 3, [True] * 3, True, [True, True, False]),
            f"the sketch names 1/world_size, allclose, torch.equal: {r['sketch']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    if sys.argv[1:2] == ["--rank"]:
        raise SystemExit(print(json.dumps(worker(*map(int, [os.environ["RANK"], *sys.argv[2:]])))))
    raise SystemExit(practice.selfcheck(globals()))
