"""Exercise 4 — the JSONL export shows rank 0's grad norm is 2x the run's and the loss curve is not monotone.

    Add a metrics export (loss, grad norm, step time) to JSONL so the run can be visualised after the fact.

Reading of the exercise: "the run" is the lesson's own `_train_worker` as 4
gloo ranks for 20 steps. The export is added from outside, without editing
the loop: the lesson's `cross_entropy` call and `ZeroOptimizer.zero_grad`
and `step` are wrapped. At each step every rank averages its loss and its
full gradient over the ranks, and rank 0 appends one JSON line: the global
loss and grad norm, rank 0's own values next to them, and the step time
from `zero_grad` to the end of the optimizer step. Each line is written
when its step ends, so a crashed run keeps its rows. "Visualised after the fact" is
tested by reading the file back and checking what the curve shows.

**ANSWER: rank 0 writes 20 well-formed rows, one per step, with finite
values and step times of about 5-15 ms.** The export does not change
training: rank 0's losses and the final parameter norm (54.513171) are
bit-identical to a run without it.

**FINDING: the grad norm rank 0 sees is 1.8-2.1x the run's.** Averaging 4
independent noisy gradients divides the norm by about sqrt(4) = 2, and the
ratio sits near that at every step. A dashboard fed by one rank would be
off by 2x.

**FINDING: the curve does not decrease monotonically, as the doc's
invariant (a) says it will.** The global loss rises at 10 of 19 steps and
rank 0's at 8. The lesson's corpus is uniform random tokens, so no model
can beat ln 64 = 4.1589 in expectation. The global loss falls from 4.3373
to 4.2111 and never goes below it. Rank 0's own loss dips below it at steps
12 and 14, which is batch noise, not learning.

Structure: `launch()` runs the lesson's worker as 4 gloo subprocess ranks
with a 150 s timeout, once plain and once exporting, and kills stragglers;
the JSONL and checkpoints go to a temporary directory. Expected output:
three PASS checks.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

from harness import parity, practice

try:
    import torch.distributed as dist
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "81-end-to-end-distributed-train"
WS = 4


def launch(ws, *args, timeout=150):
    """Run this file as `ws` gloo ranks in subprocesses; return each rank's JSON, kill stragglers."""
    argv = [sys.executable, __file__, "--rank", str(ws), *map(str, args)]
    pipe = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True}
    procs = [subprocess.Popen(argv, env=os.environ | {"RANK": str(r)}, **pipe) for r in range(ws)]
    try:
        outs = [p.communicate(timeout=timeout) for p in procs]
    finally:
        for p in procs:
            p.kill()
    if any(p.returncode for p in procs):
        raise RuntimeError(f"a rank failed: {[err[-400:] for _, err in outs]}")
    return [json.loads(out.splitlines()[-1]) for out, _ in outs]


PRINT_QUEUE = type("PrintQueue", (), {  # the lesson's mp.Queue; `_train_worker` ends in os._exit, so print first
    "put": lambda self, item: print(json.dumps(dict(zip(("rank", "losses", "norm"), item))), flush=True),
    "close": lambda self: None, "join_thread": lambda self: None,
})()


def mean_over_ranks(t):
    t = t.detach().clone()
    dist.all_reduce(t)
    return t / dist.get_world_size()


def install_exporter(ref, rank, path):
    """Wrap the lesson's loss call and ZeroOptimizer so rank 0 appends one JSON line per step."""
    seen, zero_grad, step = {}, ref.ZeroOptimizer.zero_grad, ref.ZeroOptimizer.step

    def loss_fn(*args, **kwargs):
        seen["loss"] = F.cross_entropy(*args, **kwargs)
        return seen["loss"]

    def timed_zero_grad(self):
        seen["t0"] = time.perf_counter()
        zero_grad(self)

    def exporting_step(self):
        local = ref.gather_flat_grads(self.module)
        grad, loss = mean_over_ranks(local), mean_over_ranks(seen["loss"])
        step(self)
        row = {"step": self.step_count - 1, "loss": loss.item(), "loss_rank0": seen["loss"].item(),
               "grad_norm": grad.norm().item(), "grad_norm_rank0": local.norm().item(),
               "step_time_s": time.perf_counter() - seen["t0"]}
        if rank == 0:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")

    ref.F = types.SimpleNamespace(cross_entropy=loss_fn)
    ref.ZeroOptimizer.zero_grad, ref.ZeroOptimizer.step = timed_zero_grad, exporting_step


def worker(rank, ws, mode, init_file, ckpt, path):
    ref = parity.load_reference(PHASE, LESSON, "main")
    if mode == "export":
        install_exporter(ref, int(rank), path)
    ref._train_worker(int(rank), int(ws), init_file, ref._loopback_iface(), ckpt, ref.STEPS, PRINT_QUEUE)


def solve():
    with tempfile.TemporaryDirectory(prefix="ex04_metrics_") as tmp:
        t = Path(tmp)
        runs = {m: launch(WS, m, t / f"rdv_{m}", t / f"ckpt_{m}", t / "metrics.jsonl") for m in ("plain", "export")}
        rows = [json.loads(line) for line in (t / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    rank0 = {m: next(r for r in rs if r["rank"] == 0) for m, rs in runs.items()}
    return {"rows": rows, "plain": rank0["plain"], "norms": {m: sorted({r["norm"] for r in rs}) for m, rs in runs.items()}}


KEYS = ["step", "loss", "loss_rank0", "grad_norm", "grad_norm_rank0", "step_time_s"]


def well_formed(rows):
    shape = [list(r) for r in rows] == [KEYS] * 20 and [r["step"] for r in rows] == list(range(20))
    return shape and all(math.isfinite(v) and v >= 0 for r in rows for v in r.values())


def rises(rows, key):
    return sum(b[key] > a[key] for a, b in zip(rows, rows[1:]))


def summary(result):
    rows = result["rows"]
    ratio = sorted(r["grad_norm_rank0"] / r["grad_norm"] for r in rows)
    same = [r["loss_rank0"] for r in rows] == result["plain"]["losses"]
    below = [r["step"] for r in rows if r["loss_rank0"] < math.log(64)]
    times = sorted(r["step_time_s"] for r in rows)
    return rows, ratio, same and len({n for ns in result["norms"].values() for n in ns}) == 1, below, times


def verify(result):
    (rows, ratio, same, below, times), floor = summary(result), math.log(64)
    up = {k: rises(rows, k) for k in ("loss", "loss_rank0")}
    return [
        practice.Check(
            "ANSWER: rank 0 writes one JSON line per step with loss, grad norm and step time, and training is unchanged",
            well_formed(rows) and same,
            f"20 rows with keys {KEYS}; step time {times[0] * 1e3:.1f}-{times[-1] * 1e3:.1f} ms; rank-0 losses and "
            f"final norm {result['norms']['export']} identical to the run without the export",
        ),
        practice.Check(
            "FINDING: rank 0's grad norm is about twice the run's, the sqrt(4) that averaging 4 noisy gradients gives",
            1.5 < ratio[0] and ratio[-1] < 2.5,
            f"rank-0 / global grad norm {ratio[0]:.2f}-{ratio[-1]:.2f}; global grad norm "
            f"{rows[0]['grad_norm']:.3f} at step 0, {rows[-1]['grad_norm']:.3f} at step 19",
        ),
        practice.Check(
            "FINDING: the exported curve is not the monotone decrease the doc promises; the corpus is uniform noise",
            min(up.values()) >= 5 and min(r["loss"] for r in rows) > floor and bool(below),
            f"global loss rises at {up['loss']} of 19 steps, rank 0's at {up['loss_rank0']}; "
            f"global loss {rows[0]['loss']:.4f} -> {rows[-1]['loss']:.4f}, never below ln 64 = {floor:.4f}; "
            f"rank 0's dips below it at steps {below}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    if sys.argv[1:2] == ["--rank"]:
        raise SystemExit(worker(os.environ["RANK"], *sys.argv[2:]))
    raise SystemExit(practice.selfcheck(globals()))
