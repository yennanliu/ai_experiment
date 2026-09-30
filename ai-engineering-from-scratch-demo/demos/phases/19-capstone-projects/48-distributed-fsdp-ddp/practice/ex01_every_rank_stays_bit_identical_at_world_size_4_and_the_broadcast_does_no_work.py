"""Exercise 1 — at world size 4 the ranks stay bit-identical, but the broadcast is not what keeps them so.

    Run with `--world-size 4` and confirm the param spread stays under 1e-3 across the run.

Reading of the exercise: "the run" is the lesson's own default run
(`rank_main` with 32/16/4 dims, batch 8, 6 SGD steps, seed 0) launched as
4 gloo ranks on localhost. The reference only measures the spread once, at
the end, as max - min of each rank's parameter *sum*. "Across the run" is
read literally: every rank records its full flattened parameter vector
before each of the 6 steps and after the last one, and the spread is the
largest element-wise difference from rank 0 at any of those 7 points. The
sum-based number the lesson reports is kept alongside.

**ANSWER: the spread is exactly 0.0 at all 7 checkpoints, not merely under
1e-3.** The four ranks hold bit-identical parameters from init to the end,
and the lesson's sum spread is 0.0 too. The manual all-reduce matches the
single-process gradient to 2.1e-8 at world size 4.

**FINDING: the broadcast does no work in the shipped run.** Every rank calls
`torch.manual_seed(seed)` right before `make_model`, so the ranks already
start equal. With `broadcast_module` patched to a no-op the parameters are
still bit-identical at all 7 points. The broadcast only matters when the
init differs. Seed each rank differently and the ranks stay identical with
the broadcast, and differ by up to 0.457 per element without it (sum spread 4.23).

**FINDING: the lesson reports each rank's local loss, not the global
one.** The four final losses are 1.3705, 1.5942, 1.3202 and 1.3473, and
`rank_main` makes no collective call on it (no `dist.` call at all). The doc warns: "If you average
gradients but not the loss the dashboard lies". The `losses` list is also in
queue-arrival order, not rank order. The committed `outputs/ddp-demo.json`
lists rank "1" before rank "0".

Structure: `launch()` starts this file once per rank as a subprocess with a
60 s timeout and kills stragglers; `worker()` runs the reference
`rank_main` with `MinimalDDP.sync_grads` wrapped to record parameters.
Expected output: three PASS checks.
"""

from __future__ import annotations

import inspect
import json
import os
import socket
import subprocess
import sys

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "48-distributed-fsdp-ddp"
MODES = ("shipped", "no_broadcast", "rank_seeds", "rank_seeds_no_broadcast")


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


def flat(module):
    return torch.cat([p.detach().flatten() for p in module.parameters()]).tolist()


def worker(rank, ws, port, mode):
    """One rank of the lesson's own run, recording its parameters at every step."""
    ref = parity.load_reference(PHASE, LESSON, "main")
    trace, queue, sync, make, sketch = [], [], ref.MinimalDDP.sync_grads, ref.make_model, ref.fsdp_round_trip_sketch
    ref.MinimalDDP.sync_grads = lambda self: (trace.append(flat(self)), sync(self))[1]
    ref.fsdp_round_trip_sketch = lambda m, w, r: (trace.append(flat(m)), sketch(m, w, r))[1]
    if mode.startswith("rank_seeds"):
        ref.make_model = lambda *a, **k: (torch.manual_seed(1000 + int(rank)), make(*a, **k))[1]
    if mode.endswith("no_broadcast"):
        ref.broadcast_module = lambda module, src=0: None
    sink = type("Queue", (), {"put": lambda self, item: queue.append(item)})()
    ref.rank_main(int(rank), int(ws), "gloo", int(port), sink, 32, 16, 4, 8, 6, 0.05, 0)
    _, payload, diff = queue[0]
    return {"payload": payload, "diff": diff, "trace": trace}


def spreads(ranks):
    sums = [r["payload"]["post_param_sum"] for r in ranks]
    points = zip(*(r["trace"] for r in ranks))
    elem = max(max(abs(a - b) for vec in pt for a, b in zip(pt[0], vec)) for pt in points)
    return {"elem": elem, "sum": max(sums) - min(sums), "points": len(ranks[0]["trace"])}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runs = {mode: launch(4, mode) for mode in MODES}
    shipped_json = (parity.lesson_dir(PHASE, LESSON) / "outputs" / "ddp-demo.json").read_text()
    return {
        "spread": {mode: spreads(ranks) for mode, ranks in runs.items()},
        "diff": runs["shipped"][0]["diff"],
        "losses": [round(r["payload"]["final_loss"], 4) for r in runs["shipped"]],
        "loss_reduced": "dist." in inspect.getsource(ref.rank_main),
        "json_order": list(json.loads(shipped_json)["param_sum_per_rank"]),
        "doc_warns": "average gradients but not the loss the dashboard lies" in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    s = result["spread"]
    zero = {"elem": 0.0, "sum": 0.0, "points": 7}
    return [
        practice.Check(
            "ANSWER: the spread is exactly 0.0 at all 7 checkpoints, not merely under 1e-3",
            s["shipped"] == zero and result["diff"] < 1e-7,
            f"shipped world-4 run: {s['shipped']}; manual all-reduce vs single process {result['diff']:.2g}",
        ),
        practice.Check(
            "FINDING: the broadcast does no work in the shipped run",
            s["no_broadcast"] == zero and s["rank_seeds"] == zero
            and round(s["rank_seeds_no_broadcast"]["elem"], 3) == 0.457
            and round(s["rank_seeds_no_broadcast"]["sum"], 2) == 4.23,
            f"no broadcast {s['no_broadcast']}; per-rank seeds with broadcast {s['rank_seeds']}, "
            f"without {s['rank_seeds_no_broadcast']}",
        ),
        practice.Check(
            "FINDING: the lesson reports each rank's local loss, not the global one",
            result["losses"] == [1.3705, 1.5942, 1.3202, 1.3473] and not result["loss_reduced"]
            and result["doc_warns"] and result["json_order"] == ["1", "0"],
            f"final losses by rank {result['losses']}; rank_main calls dist.*: "
            f"{result['loss_reduced']}; committed JSON rank order {result['json_order']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    if sys.argv[1:2] == ["--rank"]:
        raise SystemExit(print(json.dumps(worker(os.environ["RANK"], *sys.argv[2:]))))
    raise SystemExit(practice.selfcheck(globals()))
