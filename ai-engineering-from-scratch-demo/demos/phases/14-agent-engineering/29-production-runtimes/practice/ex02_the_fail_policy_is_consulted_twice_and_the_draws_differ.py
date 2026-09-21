"""Exercise 2 — the fail policy is consulted twice, and the draws differ.

    Add a DLQ to the queue-based demo. Simulate 10% job failure; surface DLQ
    size.

Reading of the exercise: `QueueRuntime` already has `dlq`, already appends to
it, and `main` already prints `len(rt.dlq)` -- so the DLQ ships and the work
is the simulation. Running a 10% *random* failure through the shipped worker
is where it gets interesting, because `worker` calls `fail_policy(job)` twice
per iteration and a random policy answers the two calls independently.

**ANSWER: 1000 jobs at 10% failure give 108 DLQ entries where 1 was
expected.** With the failure redrawn per call, **218** attempts fail, **108**
are retried and **108** reach the dead-letter queue -- a **10.8%** DLQ rate
against the **0.1%** that three independent 10% draws predict. The queue
drains to **0**, so the retry budget is doing nothing at all.

**FINDING: the double call lets a job reach the DLQ on its first attempt.**
`worker` evaluates `fail_policy(job)` once for the retry branch and again for
the DLQ branch, so a random policy answering False then True sends a
first-attempt job straight to the dead-letter queue with **2** of its **3**
attempts unused. **100** of the **108** DLQ entries are exactly this.
Caching one draw per attempt -- the semantics the code reads as having --
takes the DLQ to **1**, which is the number the retry policy was designed
to produce.

**FINDING: the retry path re-enqueues at the tail, so latency is unbounded
by design.** A failed job is appended behind every job already waiting, so
its wait grows with queue depth rather than with its own attempt count. On a
1000-job queue the retried jobs finish at positions **1001** to **1108**,
after **100%** of the original work -- which is fine at depth 1000 and is a
multi-hour tail at depth 100 000.

**FINDING: `fail_rate` is declared and never read.** `QueueRuntime` has **4**
fields and `worker` references **2** of them, so the field the exercise's
"10%" would naturally live in is a constructor argument with no behaviour --
the same shape as Lesson 27's unused `sensitive_tools`. The rate has to be
smuggled in through the `fail_policy` closure instead.

Structure: `run()` drives the shipped worker with a seeded policy; `drain()`
replays it with a single draw per attempt for comparison.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "29-production-runtimes"
JOBS, RATE, SEED = 1000, 0.10, 29


def seeded_policy(rng, rate=RATE):
    """The 10% failure the exercise asks for, redrawn on every call."""
    calls = []

    def policy(job):
        outcome = rng.random() < rate
        calls.append((job.jid, job.attempt, outcome))
        return outcome
    return policy, calls


def run(ref, jobs=JOBS, seed=SEED):
    runtime = ref.QueueRuntime()
    for index in range(jobs):
        runtime.enqueue(f"job {index}")
    policy, calls = seeded_policy(random.Random(seed))
    results = runtime.worker(fail_policy=policy)
    return {"runtime": runtime, "results": results, "calls": calls}


def one_draw_policy(rng, rate=RATE):
    """One draw per attempt, cached, so both branches see the same answer."""
    seen = {}

    def policy(job):
        key = (job.jid, job.attempt)
        if key not in seen:
            seen[key] = rng.random() < rate
        return seen[key]
    return policy


def drain(ref, jobs=JOBS, seed=SEED):
    runtime = ref.QueueRuntime()
    for index in range(jobs):
        runtime.enqueue(f"job {index}")
    runtime.worker(fail_policy=one_draw_policy(random.Random(seed)))
    return len(runtime.dlq)


def first_attempt_dlq(out):
    """DLQ entries whose job never got a retry."""
    return sum(job.attempt == 1 for job in out["runtime"].dlq)


def positions(out):
    """Where retried jobs land in the result order."""
    retried = [index for index, (_, status) in enumerate(out["results"], start=1)
               if status == "retry"]
    finished = [index for index, (_, status) in enumerate(out["results"], start=1)
                if status != "retry" and index > JOBS]
    return {"first": min(finished), "last": max(finished), "retries": len(retried)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    out = run(ref)
    runtime = out["runtime"]
    where = positions(out)
    fields = list(ref.QueueRuntime.__dataclass_fields__)
    return {
        "jobs": JOBS, "rate": RATE,
        "dlq": len(runtime.dlq), "queue_left": len(runtime.queue),
        "dlq_rate": round(100 * len(runtime.dlq) / JOBS, 1),
        "failed_calls": sum(outcome for _, _, outcome in out["calls"]),
        "retries": where["retries"],
        "expected": round(100 * RATE ** 3, 1),
        "first_attempt_dlq": first_attempt_dlq(out),
        "single_draw_dlq": drain(ref),
        "first_finish": where["first"], "last_finish": where["last"],
        "fields": fields,
        "read_by_worker": [f for f in fields
                           if f in ref.QueueRuntime.worker.__code__.co_names],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 1000 jobs at 10% failure give 108 DLQ entries, not the 1 expected",
            all([result["jobs"] == 1000, result["dlq"] == 108,
                 result["queue_left"] == 0, result["dlq_rate"] == 10.8,
                 result["failed_calls"] == 218, result["retries"] == 108,
                 result["expected"] == 0.1]),
            f"with the failure redrawn per call, {result['failed_calls']} attempts fail, "
            f"{result['retries']} are retried and {result['dlq']} jobs reach the "
            f"dead-letter queue -- {result['dlq_rate']}% against the "
            f"{result['expected']}% that three independent 10% draws predict. The queue "
            f"drains to {result['queue_left']}",
        ),
        practice.Check(
            "FINDING: the double call lets a job reach the DLQ on its first attempt",
            all([result["first_attempt_dlq"] == 100, result["dlq"] == 108,
                 result["single_draw_dlq"] == 1]),
            f"worker evaluates fail_policy once for the retry branch and again for the "
            f"DLQ branch, so a policy answering False then True sends a first-attempt job "
            f"straight to the DLQ: {result['first_attempt_dlq']} of {result['dlq']} "
            f"entries. Caching one draw per attempt takes the DLQ to "
            f"{result['single_draw_dlq']}",
        ),
        practice.Check(
            "FINDING: the retry path re-enqueues at the tail",
            all([result["first_finish"] == 1001, result["last_finish"] == 1108,
                 result["retries"] == 108]),
            f"a failed job is appended behind everything already waiting, so its wait "
            f"grows with queue depth rather than with its attempt count: the retried jobs "
            f"finish at positions {result['first_finish']} to {result['last_finish']}, "
            f"after all {result['jobs']} of the original work",
        ),
        practice.Check(
            "FINDING: fail_rate is declared and never read",
            all([result["fields"] == ["queue", "dlq", "fail_rate", "counter"],
                 result["read_by_worker"] == ["queue", "dlq"]]),
            f"QueueRuntime carries {result['fields']} and worker references "
            f"{result['read_by_worker']}, so the field the exercise's 10% would naturally "
            "live in is a constructor argument with no behaviour. The rate has to be "
            "smuggled in through the fail_policy closure instead",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
