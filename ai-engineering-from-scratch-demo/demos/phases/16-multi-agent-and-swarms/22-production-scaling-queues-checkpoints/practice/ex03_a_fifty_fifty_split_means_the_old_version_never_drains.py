"""Exercise 3 — a fifty-fifty split means the old version never drains.

    Simulate a **rainbow deploy**: two concurrent runtime versions; route half
    of new thread_ids to each; confirm that in-flight threads on the old version
    are not interrupted.

Reading of the exercise: a "version" is the reference agent run with a
different goal (v1 stops at 5, v2 at 7), in-flight threads are stopped
mid-run with the reference's own `crash_at`, and "not interrupted" is taken
to mean they finish on the version they started on -- which the reference
checkpoint table has to be able to say.

**ANSWER: pinned threads finish on v1, 10 of 10, and nothing is interrupted
-- but only because the pin lives outside the checkpoint table.** Ten v1
threads paused at super-step 2 all finish at counter 5 when a v1 worker
resumes them. The `checkpoints` table has 3 columns -- thread_id, super_step,
state_json -- and no version, so after a kill-and-replace deploy a v2 worker
resumes all 10 without complaint and ends them at 7: 10 runs whose steps 0-2
ran on v1 and 3-6 on v2, recorded exactly like clean runs.

**FINDING: "route half to each" keeps the old version alive forever.** With
2 new threads per tick living 5 ticks each, a 50/50 split holds v1 between
4 and 6 in-flight threads at every tick after the fifth -- 4 at tick 50 --
while sending all new threads to v2 drains v1 in 3 ticks. A rainbow retires a version only
if it stops receiving new work -- the canary slice goes to the *new* version.

**FINDING: Python's `hash()` cannot be the router.** Split by
`hash(thread_id) % 2`, 1000 thread ids land on a different version in about
half the cases when the process restarts with another hash seed, since
`str` hashes are salted per process. A restarted router would send resumed
threads to the wrong pool; `zlib.crc32` gives the same split in every
process.

Structure: `advance()` runs a thread to a super-step through the reference
`run_agent_with_checkpoint`; `in_flight()` is the tick model of the router.
"""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import zlib

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "22-production-scaling-queues-checkpoints"
GOALS, THREADS = {"v1": 5, "v2": 7}, [f"t-{i}" for i in range(10)]
HASHES = "import sys; print(''.join(str(hash(f't-{i}') % 2) for i in range(1000)))"


def advance(ref, store, thread_id, version, stop_at=None):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.suppress(RuntimeError):
        return ref.run_agent_with_checkpoint(store, thread_id, crash_at=stop_at,
                                             goal=GOALS[version])
    return None


def deploy(ref, pinned):
    """Ten v1 threads paused at step 2, then v2 ships. (finals, mixed-version runs)."""
    store, pins = ref.CheckpointStore(":memory:"), {}
    for tid in THREADS:
        advance(ref, store, tid, "v1", stop_at=3)
        pins[tid] = "v1"
    finals = [advance(ref, store, tid, pins[tid] if pinned else "v2") for tid in THREADS]
    columns = [row[1] for row in store.conn.execute("PRAGMA table_info(checkpoints)")]
    return finals, sum(f != GOALS["v1"] for f in finals), columns


def in_flight(split_to_v1, ticks=50, arrivals=2, life=5, paused=10, remaining=3):
    """v1's in-flight count per tick; paused v1 threads need `remaining` more ticks."""
    ends, counts = [remaining] * paused, []
    for tick in range(1, ticks + 1):
        for n in range(arrivals):
            if split_to_v1 and zlib.crc32(f"t-{tick}-{n}".encode()) % 2 == 0:
                ends.append(tick + life)
        ends = [end for end in ends if end > tick]
        counts.append(len(ends))
    return counts


def salted(seed):
    env = {**os.environ, "PYTHONHASHSEED": str(seed)}
    return subprocess.run([sys.executable, "-c", HASHES], env=env, capture_output=True,
                          text=True, check=True).stdout.strip()


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    half, canary = in_flight(True), in_flight(False)
    a, b = salted(1), salted(2)
    return {
        "pinned": deploy(ref, pinned=True), "replaced": deploy(ref, pinned=False),
        "half_at_50": half[-1], "half_min_after_5": min(half[5:]),
        "canary_drained": canary.index(0) + 1,
        "moved": sum(x != y for x, y in zip(a, b)), "ids": len(a),
        "crc_share": sum(zlib.crc32(f"t-{i}".encode()) % 2 for i in range(1000)),
    }


def verify(result):
    finals, _, columns = result["pinned"]
    replaced, mixed, _ = result["replaced"]
    return [
        practice.Check(
            "ANSWER: pinned threads finish on v1 and none is interrupted",
            all([finals == [5] * 10, replaced == [7] * 10, mixed == 10,
                 columns == ["thread_id", "super_step", "state_json"]]),
            f"pinned: 10 threads finish at {finals[0]}; replaced: all end at {replaced[0]}, "
            f"{mixed} mixed-version runs, because the checkpoint columns {columns} carry "
            "no version",
        ),
        practice.Check(
            "FINDING: 'route half to each' keeps the old version alive forever",
            result["half_min_after_5"] > 0 and result["canary_drained"] == 3,
            f"a 50/50 split holds v1 at {result['half_at_50']} in-flight at tick 50 and "
            f"never below {result['half_min_after_5']} after tick 5; routing all new "
            f"threads to v2 drains v1 at tick {result['canary_drained']}",
        ),
        practice.Check(
            "FINDING: Python's hash() cannot be the router",
            350 < result["moved"] < 650 and 400 < result["crc_share"] < 600,
            f"{result['moved']} of {result['ids']} thread ids change version between hash "
            f"seeds 1 and 2; crc32 puts {result['crc_share']} of 1000 on one side in every "
            "process",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
