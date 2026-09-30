"""Exercise 1 — tree allreduce wins 2x below 16 KB on the queue mesh, and above 4 MB neither topology wins.

    Add a tree allreduce variant and switch between ring and tree by message size. Measure the crossover.

Reading of the exercise: the tree is a binomial reduce onto rank 0 followed
by the lesson's own `broadcast`, written against the lesson's `Mesh`
send/recv so it runs on the same queue mesh as `ring_allreduce`. The switch
is `allreduce(mesh, tensor, threshold_bytes)`. The crossover is measured on 4
fork-context ranks over five float32 sizes, 1 KB to 16 MB: 7 barrier-aligned
calls per point, slowest rank per call, median call. Correctness is checked
against a float64 sum at world sizes 2, 3 and 4, for 64 and 66 elements.

**ANSWER: the tree agrees with the ring to float32 rounding, and it wins
where latency dominates.** At every world size and length, both land within
6e-7 of the float64 sum. The switch dispatches correctly: its byte pattern
per rank matches the tree below the threshold and the ring at or above it.
At 1 KB and 16 KB the tree is 1.7x to 3.4x faster, because it takes 4
dependent rounds against the ring's 6. The advantage fades between 256 KB
and 4 MB. At 4 MB and 16 MB, which one wins depends on the run: ring/tree
ratios of 0.78 to 1.4 were seen over repeated runs. So the crossover is a
band, not a point, and a 1 MB threshold sits inside it.

**FINDING: the tree sends exactly as many bytes as the ring; it just puts
them on fewer ranks.** At T = 4096 bytes on 4 ranks, the ring sends 6144
bytes from every rank, 6 messages each. The tree sends 8192/8192/4096/4096
bytes in 2/2/1/1 messages. Both total 24,576 bytes, so the mean per rank is
2T(N-1)/N in both cases. The doc's "T log2(N)" for the tree is its busiest
rank, not every rank.

**FINDING: the queue mesh has no bandwidth term, which is why the ring never
pulls ahead.** Pickling for `multiprocessing.Queue` sends a torch tensor as a
shared-memory handle: about 360 bytes for both 1 KB and 16 MB. What grows
with size is the local clone and add, and both topologies pay for those.
The doc's "ring above ~1 MB" rule assumes bytes that cross a link.

Structure: `tree_allreduce` and `allreduce` are the deliverable; `run()` wires
the lesson's `build_queue_grid`/`mesh_from_grid` with a per-rank lesson byte
counter and a send counter. Expected output: three PASS checks.
"""

import functools
import multiprocessing as mp
import statistics
import time
from multiprocessing.reduction import ForkingPickler

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "76-collective-ops-from-scratch"
SIZES = [1 << 8, 1 << 12, 1 << 16, 1 << 20, 1 << 22]  # float32 elements: 1 KB .. 16 MB


ref = functools.cache(lambda: parity.load_reference(PHASE, LESSON, "main"))


def tree_allreduce(mesh, tensor):
    """Binomial-tree reduce onto rank 0, then the lesson's own tree broadcast back out."""
    w, r, acc, k = mesh.world_size, mesh.rank, tensor.clone(), 1
    while k < w:
        if r % (2 * k) == k:
            mesh.send(r - k, acc)
        elif r % (2 * k) == 0 and r + k < w:
            acc = acc + mesh.recv(r + k)
        k *= 2
    return ref().broadcast(mesh, acc, src=0)


def allreduce(mesh, tensor, threshold_bytes):
    """The switch: tree below the threshold, the lesson's ring at or above it."""
    small = tensor.numel() * tensor.element_size() < threshold_bytes
    return (tree_allreduce if small else ref().ring_allreduce)(mesh, tensor)


def _worker(rank, grid, counters, fn, xs, want, reps, bar, out):
    mesh = ref().mesh_from_grid(rank, len(xs), grid, counters[rank])
    sends, send, times = [0], mesh.send, []
    mesh.send = lambda dst, t: (sends.__setitem__(0, sends[0] + 1), send(dst, t))[1]
    for _ in range(reps):
        bar.wait()
        start = time.perf_counter()
        y = fn(mesh, xs[rank])
        times.append(time.perf_counter() - start)
    out.put((rank, float((y.double() - want).abs().max()), sends[0] // reps, times))


def run(fn, xs, reps=1):
    """Run `fn(mesh, tensor)` on a fork-context queue mesh; per-rank bytes and sends, max error, median time."""
    ctx, w = mp.get_context("fork"), len(xs)
    grid, bar, out = ref().build_queue_grid(ctx, w), ctx.Barrier(w), ctx.Queue()
    counters, want = [ctx.Value("q", 0) for _ in range(w)], sum(x.double() for x in xs)
    procs = [ctx.Process(target=_worker, args=(r, grid, counters, fn, xs, want, reps, bar, out)) for r in range(w)]
    for p in procs:
        p.start()
    try:
        return summary(sorted(out.get(timeout=60) for _ in procs), counters, reps)
    finally:
        for p in procs:
            p.join(timeout=10)
            p.kill()


def summary(rows, counters, reps):
    per_rep_s = statistics.median(max(row[3][i] for row in rows) for i in range(reps))
    return {"err": max(row[1] for row in rows), "sends": [row[2] for row in rows],
            "bytes": [c.value // reps for c in counters], "ms": round(per_rep_s * 1e3, 3)}


def inputs(w, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, generator=g) for _ in range(w)]


def solve():
    ring = ref().ring_allreduce
    errors = {(w, n): max(run(tree_allreduce, inputs(w, n))["err"], run(ring, inputs(w, n))["err"])
              for w in (2, 3, 4) for n in (64, 66)}
    shape = {name: run(fn, inputs(4, 1024)) for name, fn in [("ring", ring), ("tree", tree_allreduce)]}
    switch = {t: run(lambda m, x, t=t: allreduce(m, x, t), inputs(4, 1024))["bytes"] for t in (1, 1 << 30)}
    sweep = [(n * 4, run(ring, xs, reps=7)["ms"], run(tree_allreduce, xs, reps=7)["ms"])
             for n in SIZES for xs in [inputs(4, n)]]
    return {"errors": errors, "shape": shape, "switch": switch, "sweep": sweep,
            "handle_bytes": [len(ForkingPickler.dumps(torch.zeros(n))) for n in (1 << 8, 1 << 22)]}


def verify(r):
    ring, tree = r["shape"]["ring"], r["shape"]["tree"]
    ratio = {size: round(rt / tt, 2) for size, rt, tt in r["sweep"]}
    crossover = next((size for size, rt, tt in r["sweep"] if rt < 1.2 * tt), None)
    return [
        practice.Check(
            "ANSWER: the tree is correct, the switch dispatches by size, and the tree wins below 16 KB",
            max(r["errors"].values()) < 1e-6 and r["switch"] == {1: ring["bytes"], 1 << 30: tree["bytes"]}
            and min(ratio[1024], ratio[16384]) > 1.2 and 0.5 < ratio[16777216] < 2,
            f"max error {max(r['errors'].values()):.1e}; ring/tree time ratio by bytes {ratio}; "
            f"tree advantage below 1.2x from {crossover} bytes; (bytes, ring ms, tree ms) {r['sweep']}",
        ),
        practice.Check(
            "FINDING: the tree sends as many bytes as the ring, concentrated on fewer ranks",
            (ring["bytes"], ring["sends"], tree["bytes"], tree["sends"])
            == ([6144] * 4, [6] * 4, [8192, 8192, 4096, 4096], [2, 2, 1, 1])
            and sum(ring["bytes"]) == sum(tree["bytes"]) == 2 * 4096 * 3,
            f"ring bytes {ring['bytes']} in {ring['sends']} sends; tree {tree['bytes']} in {tree['sends']}",
        ),
        practice.Check("FINDING: a queued tensor is a ~360-byte shared-memory handle at any size",
                       max(r["handle_bytes"]) < 512, f"pickled size of a 1 KB and a 16 MB tensor: {r['handle_bytes']} bytes"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
