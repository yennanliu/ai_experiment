"""Exercise 5 — over TCP the ring overtakes the tree at about 1 MB, but on the lesson's queue mesh the tree wins at every size.

    Compare wall-clock time of ring versus tree on 4 ranks for tensors of size 1KB, 1MB, 16MB. Defend the crossover empirically.

Reading of the exercise: both allreduces run on 4 fork-context ranks at the
three sizes (256, 262,144 and 4,194,304 float32s). The ring is the lesson's
`ring_allreduce`. The tree is a binomial reduce onto rank 0 plus the
lesson's `broadcast`. Each runs on two transports: the lesson's own queue
mesh, and a loopback TCP mesh with a feeder thread per peer. On TCP, gloo's
`all_reduce` is timed in the same ranks as a production baseline. Each point
is the median over 7 barrier-aligned, interleaved calls of the slowest
rank's wall-clock time.

**ANSWER: on a real wire the crossover sits at about 1 MB, as the
bandwidth-latency argument predicts.** Over TCP, the tree is 1.4x to 2.1x
faster at 1 KB. At 1 MB the two are about even (ring/tree 0.8 to 1.3). At
16 MB the ring is 1.2x to 1.45x faster (roughly 23-39 ms against 33-51 ms). The
defence is the critical path. The ring takes 6 dependent messages of T/4,
so 1.5T crosses in sequence. The tree takes 4 dependent messages of the
whole T, so 4T. At 1 KB the two extra hops cost more than the bytes, and at
16 MB the 2.5T of extra bytes cost more than the hops.

**FINDING: on the lesson's queue mesh there is no crossover.** The tree
wins at all three sizes: 1.7x to 2.9x at 1 KB, 1.3x to 2.1x at 1 MB and
1.05x to 1.6x at 16 MB. A queued tensor is a shared-memory handle, so the
extra 2.5T never crosses anything, and only the hop count is left.

**FINDING: gloo sits on the other side of both.** At 16 MB gloo's C++
allreduce beats the Python ring by 1.5x to 2x. At 1 KB it is 2x to 4x
slower than the Python tree (0.6-1.5 ms), because a fixed per-call cost
dominates small messages. The doc's "ring above ~1 MB, tree below" is
NCCL's rule for real links. It shows up here only once bytes actually
move.

Only relations with at least a 1.1x margin, in the direction seen on every
run, are asserted; the milliseconds are in the check details. Expected
output: three PASS checks.
"""

import multiprocessing as mp
import os
import queue
import socket
import statistics
import struct
import threading
import time
import types

from harness import parity, practice

try:
    import torch
    import torch.distributed as dist
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "76-collective-ops-from-scratch"
W, REPS = 4, 7
SIZES = {"1KB": 1 << 8, "1MB": 1 << 18, "16MB": 1 << 22}  # float32 elements
LEN = struct.Struct("!Q")  # frame = 8-byte payload length + raw float32 bytes


def tree_allreduce(ref, mesh, tensor):
    """Binomial-tree reduce onto rank 0, then the lesson's broadcast back out."""
    r, acc, k = mesh.rank, tensor.clone(), 1
    while k < mesh.world_size:
        if r % (2 * k) == k:
            mesh.send(r - k, acc)
        elif r % (2 * k) == 0 and r + k < mesh.world_size:
            acc = acc + mesh.recv(r + k)
        k *= 2
    return ref.broadcast(mesh, acc, src=0)


def tcp_mesh(rank, listeners):
    """Mesh send/recv over loopback TCP, float32 1-D frames, one feeder thread per peer (never blocks send)."""
    conns = {p: socket.create_connection(listeners[p].getsockname(), timeout=30) for p in range(rank)}
    for conn in conns.values():
        conn.sendall(bytes([rank]))
    while len(conns) < W - 1:
        conn = listeners[rank].accept()[0]
        conn.settimeout(30)
        conns[conn.recv(1)[0]] = conn
    files, feeds = {p: c.makefile("rb") for p, c in conns.items()}, {p: queue.Queue() for p in conns}
    for p in conns:
        threading.Thread(target=lambda p=p: [conns[p].sendall(LEN.pack(len(b)) + b)
                                             for b in iter(feeds[p].get, None)], daemon=True).start()

    def recv(src):
        size = LEN.unpack(files[src].read(LEN.size))[0]
        return torch.frombuffer(bytearray(files[src].read(size)), dtype=torch.float32)

    return types.SimpleNamespace(rank=rank, world_size=W, recv=recv,
                                 send=lambda dst, t: feeds[dst].put(t.detach().contiguous().numpy().tobytes()))


def _worker(ref, kind, rank, links, port, n, bar, out):
    mesh = ref.mesh_from_grid(rank, W, links, None) if kind == "queue" else tcp_mesh(rank, links)
    x = torch.randn(n, generator=torch.Generator().manual_seed(rank))
    fns = {"ring": ref.ring_allreduce, "tree": lambda m, t: tree_allreduce(ref, m, t)}
    if port:
        os.environ["GLOO_SOCKET_IFNAME"] = ref._loopback_iface()
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=W)
        fns["gloo"] = lambda m, t: dist.all_reduce(t.clone())
    times = {name: [] for name in fns}
    for _ in range(REPS):
        for name, fn in fns.items():  # interleaved, so drift hits every method alike
            bar.wait(timeout=30)
            start = time.perf_counter()
            fn(mesh, x)
            times[name].append(time.perf_counter() - start)
    if port:
        dist.destroy_process_group()
    out.put(times)


def links(ref, ctx, kind):
    """The queue grid and no gloo port, or one loopback listener per rank and a free port for gloo."""
    if kind == "queue":
        return ref.build_queue_grid(ctx, W), 0
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return [socket.create_server(("127.0.0.1", 0)) for _ in range(W)], s.getsockname()[1]


def timed(ref, kind, n):
    """Median over REPS of the slowest rank's ms, per method, for one transport and size."""
    ctx = mp.get_context("fork")
    (grid, port), bar, out = links(ref, ctx, kind), ctx.Barrier(W), ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(ref, kind, r, grid, port, n, bar, out)) for r in range(W)]
    for p in procs:
        p.start()
    try:
        rows = [out.get(timeout=120) for _ in procs]
    finally:
        for p in procs:
            p.join(timeout=10)
            p.kill()
    return {name: round(statistics.median(max(row[name][i] for row in rows) for i in range(REPS)) * 1e3, 3)
            for name in rows[0]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {kind: {label: timed(ref, kind, n) for label, n in SIZES.items()} for kind in ("queue", "tcp")}


def verify(result):
    q, t = result["queue"], result["tcp"]
    ratio = {kind: {size: round(v["ring"] / v["tree"], 2) for size, v in res.items()} for kind, res in result.items()}
    return [
        practice.Check("ANSWER: over TCP the tree wins at 1 KB and the ring wins at 16 MB",
                       ratio["tcp"]["1KB"] > 1.1 and ratio["tcp"]["16MB"] < 1 / 1.1,
                       f"ring/tree time ratio over TCP {ratio['tcp']}; ms {t}"),
        practice.Check("FINDING: on the lesson's queue mesh the tree is never slower than the ring",
                       ratio["queue"]["1KB"] > 1.1 and min(ratio["queue"].values()) > 0.8,
                       f"ring/tree time ratio on the queue mesh {ratio['queue']}; ms {q}"),
        practice.Check("FINDING: gloo beats the Python ring at 16 MB and loses to the Python tree at 1 KB",
                       t["16MB"]["gloo"] * 1.1 < t["16MB"]["ring"] and t["1KB"]["gloo"] > 1.1 * t["1KB"]["tree"],
                       f"gloo ms {[t[s]['gloo'] for s in SIZES]} at {list(SIZES)}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
