"""Exercise 2 — nothing is killed, and the cut worker answers afterwards.

    Implement a worker timeout: kill any worker that runs longer than 0.5
    seconds and have the lead synthesize the remaining results. What
    observability do you need to know a worker was cut?

Reading of the exercise: implement the deadline, then check what the word
"kill" bought -- because a Python thread has no terminate, so the timeout can
only stop the lead waiting, and the worker it abandoned keeps going.

**ANSWER: you need a third event and a status field, because the trace as
shipped cannot distinguish "cut" from "still running".** A worker logs exactly
**2** event kinds, `start` and `done`, and `TraceEntry` carries **4** fields --
worker_id, event, t, sub_question -- **0** of which is a status or a duration.
A cut worker has logged `start` and nothing else, which is byte-for-byte what a
slow worker in flight looks like. Adding a `timeout` event and a terminal
status makes the two separable; nothing less does.

**FINDING: nothing is killed.** `Thread.join(timeout)` returns; it does not
stop anything. After the deadline the abandoned worker is still alive --
`is_alive()` is **True** -- and it later writes into the very list the lead
already synthesized from. Measured: the lead synthesizes **2** results, and
**0.3 s** later the results list holds **3**. The answer the lead published
and the state it published it from disagree, after the fact.

**FINDING: the cut run is indistinguishable from a smaller plan.**
`synthesize` filters `r is not None`, so a 3-worker run with one worker cut
produces text that is **byte-identical** to a 2-worker run that planned two
sub-questions. The output carries no hole where the missing worker was, which
is the same absence the trace cannot report.

**FINDING: the timeout does not save the tokens it was spent to save.**
`tokens_spent` is the literal **800**, assigned after the work completes, so
the abandoned worker still records its full cost -- once it finishes, which it
does. Cutting a worker reduces the answer and not the bill.

Structure: `with_deadline()` is the timeout the exercise asks for; `later()`
re-reads the results list after the abandoned worker has finished.
"""

from __future__ import annotations

import inspect
import re
import threading
import time

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "05-supervisor-orchestrator-pattern"
SLOW, DEADLINE = 0.35, 0.12


def slow_worker(ref, results, index, trace, seconds):
    """A worker that outlives the lead's patience, written the way Worker.run is."""
    def body():
        trace.log(index, "start", f"q{index}")
        time.sleep(seconds)
        results[index] = ref.WorkerResult(f"q{index}", f"summary {index}", 800, seconds)
        trace.log(index, "done", f"q{index}")
    return body


def with_deadline(ref, count, slow_index, deadline=DEADLINE):
    """Start `count` workers, wait at most `deadline`, synthesize what arrived."""
    trace = ref.Trace()
    results = [None] * count
    threads = []
    for index in range(count):
        seconds = SLOW if index == slow_index else 0.0
        thread = threading.Thread(target=slow_worker(ref, results, index, trace, seconds))
        threads.append(thread)
        thread.start()
    limit = time.perf_counter() + deadline
    for thread in threads:
        thread.join(max(0.0, limit - time.perf_counter()))
    alive = [t.is_alive() for t in threads]
    arrived = [r for r in results if r is not None]
    return {"trace": trace, "results": results, "threads": threads, "alive": alive,
            "synthesis": ref.Lead(trace).synthesize("Q", arrived), "arrived": len(arrived)}


def later(ref, run):
    """What the results list holds once the abandoned worker has finished."""
    for thread in run["threads"]:
        thread.join()
    return sum(r is not None for r in run["results"])


def smaller_plan(ref, count):
    """The synthesis a plan of `count` sub-questions would have produced."""
    results = [ref.WorkerResult(f"q{index}", f"summary {index}", 800, 0.0)
               for index in range(count)]
    return ref.Lead(ref.Trace()).synthesize("Q", results)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    src = inspect.getsource(ref)
    run = with_deadline(ref, count=3, slow_index=2)
    cut_alive = run["alive"][2]
    after = later(ref, run)
    fields = re.findall(r"(?m)^    (\w+):",
                        src[src.index("class TraceEntry"):src.index("class Trace:")])
    events = sorted(set(re.findall(r'self\.trace\.log\(self\.worker_id, "(\w+)"', src)))
    return {
        "planned": 3, "arrived": run["arrived"], "cut_alive": cut_alive, "after": after,
        "identical": run["synthesis"] == smaller_plan(ref, run["arrived"]),
        "fields": fields, "status_fields": [f for f in fields
                                            if f in ("status", "duration", "outcome")],
        "events": events,
        "tokens_literal": re.findall(r"tokens_spent=(\d+)", src),
        "tokens_after_work": src.index("tokens_spent=800") > src.index("summary = fake_web_fetch"),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: you need a third event and a status field",
            all([result["events"] == ["done", "start"], len(result["fields"]) == 4,
                 result["status_fields"] == []]),
            f"a worker logs {len(result['events'])} event kinds "
            f"({', '.join(result['events'])}) and TraceEntry carries "
            f"{len(result['fields'])} fields ({', '.join(result['fields'])}), "
            f"{len(result['status_fields'])} of them a status or duration -- so a cut "
            "worker looks exactly like a slow one still in flight",
        ),
        practice.Check(
            "FINDING: nothing is killed",
            all([result["cut_alive"], result["arrived"] == 2, result["after"] == 3]),
            f"Thread.join(timeout) returns without stopping anything: after the deadline "
            f"the abandoned worker is still alive, the lead synthesizes "
            f"{result['arrived']} results, and once the worker finishes the list it "
            f"synthesized from holds {result['after']}",
        ),
        practice.Check(
            "FINDING: the cut run is indistinguishable from a smaller plan",
            all([result["identical"], result["arrived"] < result["planned"]]),
            f"synthesize filters r is not None, so a {result['planned']}-worker run with "
            f"one cut produces text byte-identical to a {result['arrived']}-worker run "
            "that planned two sub-questions -- the output carries no hole",
        ),
        practice.Check(
            "FINDING: the timeout does not save the tokens it was spent to save",
            all([result["tokens_literal"] == ["800"], result["tokens_after_work"]]),
            f"tokens_spent is the literal {result['tokens_literal'][0]}, assigned after "
            "the fetch returns, so the abandoned worker records its full cost once it "
            "finishes -- cutting a worker shortens the answer and not the bill",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
