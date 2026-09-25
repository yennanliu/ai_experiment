"""Exercise 1 — resume works because the crash is placed where nothing can go wrong.

    Run `code/main.py`. Confirm checkpoint resume works; measure async vs
    thread concurrency difference.

Reading of the exercise: "confirm" is taken as comparing the resumed run's
checkpoint table against an uncrashed run's, and then asking which crashes
the demo can express at all; "measure" is taken literally for both of the
quantities the lesson promises, wall clock and memory.

**ANSWER: resume works, and threads are a small factor slower, not "several
seconds".**
Worker 1 crashes at super-step 3 with checkpoint (2, {counter: 3}); worker 2
resumes and ends at 5, and the table it leaves is row-for-row the table of a
run that never crashed. On the lesson's 500 calls of 50ms, asyncio takes about
0.05s and threads 0.07s here (0.16s on a CI runner, 2.9x) -- the lesson's
"thread version takes several seconds" is off by one to two orders of
magnitude, because a thread costs tens of microseconds to start against a
50ms sleep. The ratio is host-dependent; the check asserts only that the
threaded run stays under a second.

**FINDING: the crash can only happen between steps.** `crash_at` is tested
*before* the increment, so every simulated crash falls right after a committed
checkpoint: the resumed run redoes 0 steps and executes exactly the 2 that
were missing. And the agent has no side effect, so "no duplicate side
effects" is true of this demo vacuously.

**FINDING: there is no lease, and `INSERT OR REPLACE` erases the evidence.**
Two workers that both resume the crashed thread both run the remaining steps
-- 4 step executions for 2 steps -- and the table still holds 5 rows equal to
a clean run's. The double execution leaves no trace to detect it by.

**FINDING: memory is never measured.** The lesson says the demo reports "peak
memory (approximated)"; the module imports nothing that can read memory and
the "~1MB per thread stack" is a string literal. Measured as resident memory in a
fresh interpreter, a sleeping thread costs tens of KB and a pending coroutine
about 1.5KB on macOS -- more than a coroutine, and nowhere near 1MB per
thread, because a thread's stack is reserved virtually and only its touched
pages become resident.

Structure: `crash_then_resume()` drives the reference `run_agent_with_checkpoint`
against its `CheckpointStore`; `per_unit_rss()` runs a subprocess so the reading
starts from a clean process. It reads *current* RSS from /proc on Linux: there
`ru_maxrss` is a high-water mark that interpreter start-up already exceeds, so
the growth from 500 threads reads as 0. macOS has no /proc, and its
`ru_maxrss` (in bytes) climbs with the threads.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import io
import subprocess
import sys

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "22-production-scaling-queues-checkpoints"
N = 500
PROBE = """
import asyncio, os, resource, sys, threading, time
def rss():
    if os.path.exists('/proc/self/statm'):
        return int(open('/proc/self/statm').read().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
base = rss()
if sys.argv[1] == 'thread':
    ts = [threading.Thread(target=time.sleep, args=(0.3,)) for _ in range({n})]
    [t.start() for t in ts]; time.sleep(0.1); peak = rss(); [t.join() for t in ts]
else:
    async def main():
        tasks = [asyncio.ensure_future(asyncio.sleep(0.3)) for _ in range({n})]
        await asyncio.sleep(0.1); p = rss(); await asyncio.gather(*tasks); return p
    peak = asyncio.run(main())
print((peak - base) / {n})
"""


def counted(store):
    """Wrap store.write so each executed step is counted."""
    calls = []
    original = store.write
    store.write = lambda *args: (calls.append(args[1]), original(*args))[1]
    return calls


def crash_then_resume(ref, resumers=1):
    store = ref.CheckpointStore(":memory:")
    calls = counted(store)
    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.suppress(RuntimeError):
            ref.run_agent_with_checkpoint(store, "t-1", crash_at=3)
        crashed_at = store.latest("t-1")
        finals = [ref.run_agent_with_checkpoint(store, "t-1") for _ in range(resumers)]
    rows = store.conn.execute("SELECT * FROM checkpoints ORDER BY super_step").fetchall()
    return {"crashed_at": crashed_at, "finals": finals, "rows": rows, "calls": calls}


def race_resume(ref):
    """Two workers both read the crash checkpoint before either writes."""
    store = ref.CheckpointStore(":memory:")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.suppress(RuntimeError):
        ref.run_agent_with_checkpoint(store, "t-1", crash_at=3)
    snapshot = store.latest("t-1")
    calls = counted(store)
    for _ in range(2):
        store.latest = lambda _tid, snap=snapshot: (snap[0], dict(snap[1]))
        ref.run_agent_with_checkpoint(store, "t-1")
    return calls, store.conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]


def per_unit_rss(kind):
    out = subprocess.run([sys.executable, "-c", PROBE.format(n=N), kind],
                         capture_output=True, text=True, check=True)
    return float(out.stdout)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    clean = ref.CheckpointStore(":memory:")
    ref.run_agent_with_checkpoint(clean, "t-1")
    src = inspect.getsource(ref)
    return {
        "resume": crash_then_resume(ref), "race": race_resume(ref),
        "clean_rows": clean.conn.execute("SELECT * FROM checkpoints ORDER BY super_step").fetchall(),
        "async": asyncio.run(ref.bench_async(N)), "threads": ref.bench_threads(N),
        "thread_rss": per_unit_rss("thread"), "coro_rss": per_unit_rss("coro"),
        "reads_memory": any(m in src for m in ("tracemalloc", "resource", "psutil")),
        "literal": "~1MB per thread stack" in src,
    }


def verify(result):
    resume, (race_calls, race_rows) = result["resume"], result["race"]
    ratio = result["threads"] / result["async"]
    return [
        practice.Check(
            "ANSWER: resume works, and threads are a small factor slower, not several seconds",
            all([resume["crashed_at"] == (2, {"counter": 3}), resume["finals"] == [5],
                 resume["rows"] == result["clean_rows"], result["threads"] < 1.0]),
            f"crash leaves {resume['crashed_at']}, resume ends at {resume['finals'][0]} with "
            f"a table identical to a clean run's; {N} calls take {result['async']:.3f}s async "
            f"and {result['threads']:.3f}s threaded, {ratio:.2f}x",
        ),
        practice.Check(
            "FINDING: the crash can only happen between steps",
            resume["calls"] == [0, 1, 2, 3, 4],
            f"steps executed across both workers: {resume['calls']} -- crash_at is tested "
            "before the increment, so the resumed run redoes nothing, and there is no "
            "side effect to duplicate",
        ),
        practice.Check(
            "FINDING: there is no lease, and INSERT OR REPLACE erases the evidence",
            race_calls == [3, 4, 3, 4] and race_rows == 5,
            f"two resumers execute steps {race_calls} -- 4 executions for 2 steps -- and "
            f"the table still has {race_rows} rows, like a clean run",
        ),
        practice.Check(
            "FINDING: memory is never measured",
            all([not result["reads_memory"], result["literal"],
                 0 < result["thread_rss"] < 256 * 1024,
                 result["thread_rss"] > result["coro_rss"]]),
            f"no memory API is imported and '~1MB per thread stack' is a literal; measured "
            f"resident memory is {result['thread_rss'] / 1024:.1f}KB per thread against "
            f"{result['coro_rss'] / 1024:.1f}KB per coroutine",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
