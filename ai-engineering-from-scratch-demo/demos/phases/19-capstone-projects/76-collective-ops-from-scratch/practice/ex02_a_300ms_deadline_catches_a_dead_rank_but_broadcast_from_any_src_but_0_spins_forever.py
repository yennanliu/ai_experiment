"""Exercise 2 — a 300 ms deadline turns a dead rank into three errors in 0.4 s, but broadcast from any src but 0 spins forever.

    Add a `recv_timeout_ms` so a stalled rank surfaces a deadline error instead of hanging forever.

Reading of the exercise: `with_deadline(mesh, recv_timeout_ms)` swaps the
lesson `Mesh`'s `recv` for one that raises `DeadlineError`, and the message
names the waiting rank, the peer it waited on, and which recv it was. "A
stalled rank" is rank 3 of 4 never starting, while ranks 0-2 run each of the
lesson's four primitives on the lesson's queue mesh with a 300 ms deadline.
The lesson's own timeout path is measured too, with its 30 s constant
shortened to 0.2 s so the run stays short.

**ANSWER: every ring primitive now fails fast and says where.** In
allreduce, allgather and reduce_scatter, all three live ranks raise
`DeadlineError` within 0.3-0.4 s. Rank 0 reports "no message from rank 3
(recv #1)". Broadcast from rank 0 completes on all three live ranks, because
rank 3 is a leaf that only receives, and a dead leaf is invisible to the
senders.

**FINDING: the deadline error blames the neighbour, not the stalled rank.**
Only rank 0 names rank 3. Rank 1 names rank 0 at recv #2, and rank 2 names
rank 1 at recv #3: the stall propagates one hop per ring step. So 2 of the 3
errors point at a healthy rank. Finding the culprit takes the lowest recv
number, not whichever error arrives first.

**FINDING: the lesson already has a timeout, but it is anonymous and
slow.** `Mesh.recv` uses `RECV_TIMEOUT_S = 30.0` and raises a bare
`queue.Empty` whose message is the empty string. `run_mesh` then waits
`get(timeout=60)` for results and `join(timeout=30)` per process. A dead rank
therefore surfaces in the parent after about a minute, with no rank in the
message.

**FINDING: the lesson's broadcast never terminates for `src` other than 0,
and no recv deadline can catch it.** The tree only reaches ranks `h + 2^k`
above the holders, so when `src` > 0 the holder set stops growing. The
`while` loop then spins without ever calling `recv`. Run with a no-op mesh,
`src=0` returns, and `src` = 1, 2 and 3 are all still running after 3 s. The
gloo check and every test in the lesson use `src=0`.

Structure: `with_deadline` is the deliverable; `stalled_run` forks the three
live ranks; `broadcast_terminates` runs one broadcast in a child that is
killed after 3 s. Expected output: four PASS checks.
"""

import inspect
import multiprocessing as mp
import queue
import re
import time
import types

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "76-collective-ops-from-scratch"
DEADLINE_MS, WORLD, STALLED = 300, 4, 3


class DeadlineError(TimeoutError):
    """A recv that missed its deadline; says which rank waited on which peer, and for how long."""


def with_deadline(mesh, recv_timeout_ms):
    """Replace the lesson Mesh's recv with one that raises DeadlineError after `recv_timeout_ms`."""
    count = [0]

    def recv(src):
        count[0] += 1
        try:
            return mesh.in_queues[src].get(timeout=recv_timeout_ms / 1000)
        except queue.Empty:
            raise DeadlineError(f"rank {mesh.rank}: no message from rank {src} within {recv_timeout_ms} ms "
                                f"(recv #{count[0]})") from None

    mesh.recv = recv
    return mesh


def _worker(ref, op, rank, grid, out):
    mesh = with_deadline(ref.mesh_from_grid(rank, WORLD, grid, None), DEADLINE_MS)
    fn = {"allreduce": ref.ring_allreduce, "allgather": ref.allgather, "reduce_scatter": ref.reduce_scatter,
          "broadcast": lambda m, x: ref.broadcast(m, x, src=0)}[op]
    start, status = time.perf_counter(), "ok"
    try:
        fn(mesh, torch.ones(WORLD * 8))
    except DeadlineError as exc:
        status = str(exc)
    out.put((rank, status, round(time.perf_counter() - start, 2)))


def stalled_run(ref, op):
    """Every rank but STALLED runs `op`; STALLED never starts. Returns one row per live rank."""
    ctx = mp.get_context("fork")
    grid, out = ref.build_queue_grid(ctx, WORLD), ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(ref, op, r, grid, out)) for r in range(WORLD) if r != STALLED]
    for p in procs:
        p.start()
    try:
        return sorted(out.get(timeout=30) for _ in procs)
    finally:
        for p in procs:
            p.join(timeout=5)
            p.kill()


def broadcast_terminates(ref, src, wait_s=3.0):
    """Run the lesson's broadcast with a no-op mesh in a child; False if it is still running after wait_s."""
    mesh = types.SimpleNamespace(rank=src, world_size=WORLD, send=lambda dst, t: None, recv=lambda s: torch.zeros(2))
    proc = mp.get_context("fork").Process(target=ref.broadcast, args=(mesh, torch.ones(2), src))
    proc.start()
    proc.join(timeout=wait_s)
    alive = proc.is_alive()
    proc.kill()
    proc.join(timeout=5)
    return not alive


def lesson_timeout(ref):
    """What the lesson's own recv raises on a dead peer, with RECV_TIMEOUT_S shortened to 0.2 s."""
    grid, saved = ref.build_queue_grid(mp.get_context("fork"), 2), ref.RECV_TIMEOUT_S
    ref.RECV_TIMEOUT_S = 0.2
    try:
        return ref.mesh_from_grid(0, 2, grid, None).recv(1)
    except Exception as exc:  # noqa: BLE001 - the type is what is measured
        return type(exc).__name__, str(exc)
    finally:
        ref.RECV_TIMEOUT_S = saved


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {"runs": {op: stalled_run(ref, op) for op in ref.PRIMITIVES},
            "lesson_raises": lesson_timeout(ref), "lesson_recv_s": ref.RECV_TIMEOUT_S,
            "parent_waits": [s in inspect.getsource(ref.run_mesh) for s in ("get(timeout=60)", "join(timeout=30)")],
            "broadcast_ends": {s: broadcast_terminates(ref, s) for s in range(WORLD)}}


def answer_check(runs):
    rows = [row for op in ("allreduce", "allgather", "reduce_scatter") for row in runs[op]]
    named = all(row[1].startswith(f"rank {row[0]}: no message") and 0.3 <= row[2] < 5 for row in rows)
    return practice.Check(
        "ANSWER: each ring primitive raises DeadlineError on all 3 live ranks; broadcast from 0 completes",
        len(rows) == 9 and named and list(zip(*runs["broadcast"]))[1] == ("ok",) * 3,
        f"seconds to the error per rank {list(zip(*rows))[2]}; broadcast {runs['broadcast']}")


def blame_check(runs):
    blamed = {op: [re.findall(r"from rank (\d+) .*#(\d+)", row[1]) for row in rows]
              for op, rows in runs.items() if op != "broadcast"}
    return practice.Check(
        "FINDING: only rank 0's error names the stalled rank; the others blame a healthy neighbour",
        all(b == [[("3", "1")], [("0", "2")], [("1", "3")]] for b in blamed.values()),
        f"(peer named, recv #) for ranks 0-2: {blamed['allreduce']}; allreduce: {runs['allreduce'][2][1]}")


def verify(r):
    return [
        answer_check(r["runs"]),
        blame_check(r["runs"]),
        practice.Check(
            "FINDING: the lesson's own timeout is a bare queue.Empty after 30 s, and run_mesh waits 60 s for results",
            (r["lesson_raises"], r["lesson_recv_s"], r["parent_waits"]) == (("Empty", ""), 30.0, [True, True]),
            f"raises {r['lesson_raises']}; RECV_TIMEOUT_S = {r['lesson_recv_s']}; "
            f"run_mesh has get(timeout=60), join(timeout=30): {r['parent_waits']}"),
        practice.Check("FINDING: the lesson's broadcast spins forever for src != 0, with no recv to time out",
                       r["broadcast_ends"] == {0: True, 1: False, 2: False, 3: False},
                       f"returned within 3 s, by src: {r['broadcast_ends']}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
