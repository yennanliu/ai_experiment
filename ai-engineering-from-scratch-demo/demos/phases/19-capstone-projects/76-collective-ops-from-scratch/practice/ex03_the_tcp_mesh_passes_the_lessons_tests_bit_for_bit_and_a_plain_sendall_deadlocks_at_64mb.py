"""Exercise 3 — the TCP mesh passes the lesson's tests bit for bit, and a plain sendall deadlocks the ring at 64 MB.

    Replace `multiprocessing.Queue` with TCP sockets for the four primitives. Same tests, real wire.

Reading of the exercise: `tcp_mesh` gives the lesson's `send`/`recv`
interface over one loopback TCP connection per rank pair. Each frame is an
8-byte length and the raw float32 bytes (every tensor the lesson's tests
send is 1-D float32). The
lesson's own four primitives run on it unchanged. "Same tests" means the
inputs of the lesson's `tests/test_collectives.py`, seed for seed. The
reference is gloo, run in the same forked ranks, with the tests' own
tolerances. The queue-mesh output from the lesson's `run_mesh` is compared
as well.

**ANSWER: all four primitives run over real sockets and pass every lesson
test.** On all four 4-rank cases, the output is bit-identical to the queue
mesh. It is within the tests' tolerances of gloo (worst 4.8e-7), and the
2-rank sum of ones and twos is 3.0 everywhere.

**FINDING: a plain blocking `sendall` deadlocks the lesson's ring on large
tensors.** Every ring step is "send to next, then recv from prev". A
`multiprocessing.Queue.put` never blocks, because a feeder thread does the
write. A socket's `sendall` blocks once the kernel buffers fill. At 128 B
the plain socket version passes. At 64 MB (16 MB chunks) all 4 ranks are
stuck in `sendall` at once until the 3 s socket timeout (a rank whose peer
exits first may see a broken pipe instead). Giving
every peer a feeder thread, as the queue has, makes the same 64 MB
allreduce complete. The queue version was only correct because of a thread
it never mentions.

**FINDING: the doc's "more than float32 epsilon, the test fails" would fail
the lesson's own allreduce.** Ring and gloo add in different orders.
Allreduce differs from gloo by 4.8e-7, which is 4 float32 epsilons
(1.19e-7), and reduce_scatter by 2.4e-7 (the check asserts more than one
epsilon, since the exact gap depends on gloo's reduction order). The tests pass because they use
atol 1e-5, about 84 epsilons, and `main.py` uses 1e-5 as well. The
docstring's "byte-for-byte" and the doc's "byte-equal comparison" hold for
broadcast and allgather only.

Structure: `tcp_mesh` is the deliverable; `run_tcp` forks the ranks, which
run the cases on TCP and then on gloo; `lesson_cases` restates the test
inputs. Expected output: three PASS checks.
"""

import multiprocessing as mp
import os
import queue
import socket
import struct
import threading
import types

from harness import parity, practice

try:
    import torch
    import torch.distributed as dist
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "76-collective-ops-from-scratch"
LEN = struct.Struct("!Q")  # frame = 8-byte payload length + raw float32 bytes


def connect(rank, w, listeners, timeout):
    """One TCP connection per rank pair: dial every lower rank, accept every higher one."""
    socks = {p: socket.create_connection(listeners[p].getsockname(), timeout=timeout) for p in range(rank)}
    for sock in socks.values():
        sock.sendall(bytes([rank]))
    while len(socks) < w - 1:
        sock = listeners[rank].accept()[0]
        sock.settimeout(timeout)
        socks[sock.recv(1)[0]] = sock
    return socks


def tcp_mesh(rank, w, listeners, fed=True, timeout=20):
    """The lesson's Mesh send/recv over TCP. With `fed`, send() hands the frame to a per-peer
    feeder thread, as multiprocessing.Queue.put does; without, it blocks in sendall."""
    socks = connect(rank, w, listeners, timeout)
    files, feeds = {p: s.makefile("rb") for p, s in socks.items()}, {p: queue.Queue() for p in socks}
    for p in socks:
        threading.Thread(target=lambda p=p: [socks[p].sendall(b) for b in iter(feeds[p].get, None)], daemon=True).start()

    def send(dst, t):
        data = t.detach().contiguous().numpy().tobytes()
        (feeds[dst].put if fed else socks[dst].sendall)(LEN.pack(len(data)) + data)

    def recv(src):
        size = LEN.unpack(files[src].read(LEN.size))[0]
        return torch.frombuffer(bytearray(files[src].read(size)), dtype=torch.float32)

    return types.SimpleNamespace(rank=rank, world_size=w, send=send, recv=recv)


def gloo_op(op, x, w):
    """The same op through torch.distributed gloo, as the lesson's _gloo_worker issues it."""
    parts, y = [torch.zeros_like(x) for _ in range(w)], torch.zeros(x.numel() // w) if op[0] == "r" else x.clone()
    {"allreduce": lambda: dist.all_reduce(y), "broadcast": lambda: dist.broadcast(y, src=0),
     "allgather": lambda: dist.all_gather(parts, x),
     "reduce_scatter": lambda: dist.reduce_scatter(y, [c.contiguous() for c in x.chunk(w)])}[op]()
    return torch.cat(parts) if op == "allgather" else y


def _worker(ref, rank, w, listeners, port, cases, fed, out):
    call = {"allreduce": ref.ring_allreduce, "allgather": ref.allgather, "reduce_scatter": ref.reduce_scatter,
            "broadcast": lambda m, x: ref.broadcast(m, x, src=0)}
    try:
        mesh = tcp_mesh(rank, w, listeners, fed, timeout=20 if fed else 3)
        rows = [call[op](mesh, xs[rank]) for op, xs in cases]
        if port:
            os.environ["GLOO_SOCKET_IFNAME"] = ref._loopback_iface()
            dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=w)
            rows = [(y, float((y - gloo_op(op, xs[rank], w)).abs().max())) for y, (op, xs) in zip(rows, cases)]
            dist.destroy_process_group()
        out.put((rank, "ok", rows))
    except OSError as exc:
        out.put((rank, type(exc).__name__, []))


def run_tcp(ref, w, cases, fed=True, with_gloo=False):
    """All `cases` [(op, per-rank inputs)] on w forked ranks over TCP (then gloo): [(rank, status, rows)]."""
    ctx = mp.get_context("fork")
    out, listeners = ctx.Queue(), [socket.create_server(("127.0.0.1", 0)) for _ in range(w)]
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1] if with_gloo else 0
    procs = [ctx.Process(target=_worker, args=(ref, r, w, listeners, port, cases, fed, out)) for r in range(w)]
    for p in procs:
        p.start()
    try:
        return sorted(out.get(timeout=90) for _ in procs)
    finally:
        for p in procs:
            p.join(timeout=5)
            p.kill()


def lesson_cases():
    """The inputs of the lesson's tests/test_collectives.py, seed for seed."""
    seeded = [(torch.manual_seed(seed), [torch.randn(n) for _ in range(k)])[1]
              for seed, n, k in [(1, 32, 4), (2, 16, 1), (3, 8, 4), (4, 32, 4)]]
    seeded[1] += [torch.zeros(16)] * 3
    return list(zip(["allreduce", "broadcast", "allgather", "reduce_scatter"], seeded))


def statuses(ref, xs, fed):
    return [status for _, status, _ in run_tcp(ref, 4, [("allreduce", xs)], fed)]


def solve():
    ref, cases = parity.load_reference(PHASE, LESSON, "main"), lesson_cases()
    ranks, big = run_tcp(ref, 4, cases, with_gloo=True), list(torch.randn(4, 1 << 24))
    cols = list(zip(*(rows for _, _, rows in ranks)))  # cols[case][rank] = (output, |tcp - gloo|)
    queue_out = [ref.run_mesh(op, 4, xs)[0] for op, xs in cases]
    return {"status": [s for _, s, _ in ranks], "vs_gloo": [max(row[1] for row in col) for col in cols],
            "same_as_queue": [all(map(torch.equal, next(zip(*col)), q)) for col, q in zip(cols, queue_out)],
            "ws2": run_tcp(ref, 2, [("allreduce", [torch.ones(8), torch.ones(8) * 2])])[0][2][0].tolist(),
            "plain": [statuses(ref, cases[0][1], False), statuses(ref, big, False)], "fed": statuses(ref, big, True)}


def verify(result):
    r, eps, tols = result, torch.finfo(torch.float32).eps, (1e-5, 1e-6, 1e-6, 1e-5)
    return [
        practice.Check(
            "ANSWER: the four primitives over TCP pass the lesson's tests and match the queue mesh bit for bit",
            (r["status"], r["same_as_queue"], r["ws2"]) == (["ok"] * 4, [True] * 4, [3.0] * 8)
            and all(map(float.__le__, r["vs_gloo"], tols)),
            f"max |tcp - gloo| per op {r['vs_gloo']}; bit-equal to queue mesh {r['same_as_queue']}",
        ),
        practice.Check(
            "FINDING: a blocking sendall deadlocks the ring at 64 MB; a per-peer feeder thread fixes it",
            (r["plain"][0], r["fed"], "ok" in r["plain"][1], "TimeoutError" in r["plain"][1])
            == (["ok"] * 4, ["ok"] * 4, False, True),
            f"plain sendall at 128 B / 64 MB {r['plain']}; with feeder threads at 64 MB {r['fed']}",
        ),
        practice.Check(
            "FINDING: allreduce differs from gloo by more than one float32 epsilon",
            min(r["vs_gloo"][0], r["vs_gloo"][3]) > eps and r["vs_gloo"][1:3] == [0.0, 0.0],
            f"(allreduce, broadcast, allgather, reduce_scatter) {r['vs_gloo']}; eps {eps:.3g}; "
            f"allreduce is {r['vs_gloo'][0] / eps:.1f} eps",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
