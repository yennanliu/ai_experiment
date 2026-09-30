"""Exercise 4 — the JSONL log shows broadcast's per-rank bytes are 2T/T/0/0, not the table's T, and padding breaks 2T(N-1)/N.

    Add a bandwidth instrumentation hook so the per-rank byte counter logs to JSONL.

Reading of the exercise: `jsonl_hook(mesh, path, op)` wraps the lesson
`Mesh.send`. The lesson's shared `byte_counter` still counts, and each send
also appends one line: op, rank, dst, seq, nbytes and a monotonic
timestamp. All four ranks share one file, and each line is a single short
append. Each primitive runs on the lesson's queue mesh at 4 ranks, at
`main.py`'s sizes (64 floats; 256 for reduce_scatter), plus a 66-float
allreduce that does not divide by 4. Per-rank bytes are summed from the
log and compared with the lesson counter and with the doc's table.

**ANSWER: the log is complete and agrees with the lesson's counter.** 75
lines, one per send (24 + 24 + 3 + 12 + 12), every one parseable, with six
keys. For every case, the per-rank sums from the log add up to the lesson
counter's total. The log's advantage is that it keeps the per-rank split
that the shared counter throws away.

**FINDING: the doc's per-rank byte table holds for allreduce and
reduce_scatter only.** At T = 256 bytes, allreduce sends 384 from every rank
(2T(N-1)/N), and reduce_scatter of a 1024-byte input sends 768 (T(N-1)/N).
The broadcast row says T per rank, but the log shows 512/256/0/0: the root
sends 2T = T log2(N), two leaves send nothing, and the mean is 3T/4. The
allgather row, T(N-1)/N, gives 192 for a 256-byte input, but each rank sends
768 = T(N-1). The row only fits if T means the 1024-byte gathered output.
`main.py` prints total / N as "per-rank bytes", which cannot show the
broadcast's skew.

**FINDING: the 2T(N-1)/N "proof" holds only when N divides the length.**
For 66 floats (T = 264 bytes), the ring pads to 68, and the log shows 408
bytes per rank. The doc's formula gives 396, and `main.py`'s expected value
`2*(N-1)*(n//N)*4` gives 384. Neither matches.

Structure: `jsonl_hook` is the deliverable; `run_logged` forks the 4 ranks
with the lesson counter; `per_rank` sums the log. Expected output: three
PASS checks.
"""

import collections
import json
import multiprocessing as mp
import pathlib
import tempfile
import time

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "76-collective-ops-from-scratch"
W = 4


def jsonl_hook(mesh, path, op):
    """Wrap a lesson Mesh's send so every send appends one JSON line; the lesson's counter still counts."""
    send, seq = mesh.send, [0]

    def logged(dst, tensor):
        send(dst, tensor)
        seq[0] += 1
        row = {"op": op, "rank": mesh.rank, "dst": dst, "seq": seq[0],
               "nbytes": tensor.numel() * tensor.element_size(), "t_ns": time.monotonic_ns()}
        with open(path, "a", encoding="utf-8") as f:  # one short O_APPEND write per line
            f.write(json.dumps(row) + "\n")

    mesh.send = logged
    return mesh


def _worker(ref, op, label, rank, grid, counter, x, path, out):
    mesh = jsonl_hook(ref.mesh_from_grid(rank, W, grid, counter), path, label)
    if op == "broadcast":
        ref.broadcast(mesh, x, src=0)
    else:
        {"allreduce": ref.ring_allreduce, "allgather": ref.allgather, "reduce_scatter": ref.reduce_scatter}[op](
            mesh, x)
    out.put(rank)


def run_logged(ref, label, op, xs, path):
    """Run one primitive on the lesson's queue mesh with the hook; return the lesson counter's total."""
    ctx = mp.get_context("fork")
    grid, counter, out = ref.build_queue_grid(ctx, W), ctx.Value("q", 0), ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(ref, op, label, r, grid, counter, xs[r], path, out)) for r in range(W)]
    for p in procs:
        p.start()
    try:
        for _ in procs:
            out.get(timeout=30)
    finally:
        for p in procs:
            p.join(timeout=5)
            p.kill()
    return counter.value


def per_rank(rows, op):
    sums = collections.Counter({r: 0 for r in range(W)})
    for row in rows:
        if row["op"] == op:
            sums[row["rank"]] += row["nbytes"]
    return [sums[r] for r in range(W)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    g = torch.Generator().manual_seed(7)
    cases = {"allreduce": 64, "allreduce_66": 66, "broadcast": 64, "allgather": 64, "reduce_scatter": 256}
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "bytes.jsonl"
        counters = {name: run_logged(ref, name, name.split("_6")[0], [torch.randn(n, generator=g) for _ in range(W)],
                                     path) for name, n in cases.items()}
        lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    return {"lines": len(lines), "keys": sorted(rows[0]), "counters": counters,
            "per_rank": {name: per_rank(rows, name) for name in cases}, "t_bytes": {k: v * 4 for k, v in cases.items()}}


def table_check(pr, t):
    return practice.Check(
        "FINDING: the table's per-rank bytes hold for allreduce and reduce_scatter, not broadcast or allgather",
        pr["allreduce"] == [2 * t["allreduce"] * 3 // 4] * 4 and pr["reduce_scatter"] == [t["reduce_scatter"] * 3 // 4] * 4
        and pr["broadcast"] == [2 * t["broadcast"], t["broadcast"], 0, 0]
        and pr["allgather"] == [t["allgather"] * 3] * 4,
        f"broadcast {pr['broadcast']} vs table T = {t['broadcast']}; allgather {pr['allgather']} vs "
        f"T(N-1)/N = {t['allgather'] * 3 // 4}",
    )


def verify(result):
    r = result
    pr, t = r["per_rank"], r["t_bytes"]
    return [
        practice.Check(
            "ANSWER: one JSONL line per send, and the per-rank sums add up to the lesson's counter",
            r["lines"] == 75 and r["keys"] == ["dst", "nbytes", "op", "rank", "seq", "t_ns"]
            and all(sum(pr[name]) == r["counters"][name] for name in pr),
            f"{r['lines']} lines, keys {r['keys']}; per-rank bytes {pr}; lesson counter totals {r['counters']}",
        ),
        table_check(pr, t),
        practice.Check(
            "FINDING: with padding the ring sends more than 2T(N-1)/N and more than main.py expects",
            pr["allreduce_66"] == [408] * 4 and 2 * t["allreduce_66"] * 3 / 4 == 396 and 2 * 3 * (66 // 4) * 4 == 384,
            f"66 floats: {pr['allreduce_66']} bytes per rank vs formula 396 and main.py's 384",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
