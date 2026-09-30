"""Exercise 3 — threaded during-gen stays far inside 50 ms, but an early stop lets the model generate one chunk past the match, so the budget is a cap on chunk time.

    Add an async streaming variant where during-gen runs in a thread; verify the latency impact stays within a 50ms budget.

Reading of the exercise: the variant keeps the lesson's `_during_gen` and
runs it in a worker thread fed by a bounded queue (size 1). The caller's
thread drives the model, checks a stop event before asking for each next
chunk, and stops when the worker signals a match. Everything else is the
lesson's own `SafetyGate.handle`: `_during_gen` and the `stream` it
reads are swapped on one gate and restored after each request. "Latency
impact" is the difference in the trace's own `latency_ms`, threaded minus
synchronous. It is measured over the 60 prompts `main.py` runs, once with
the instant mock stream (best of 5) and once with a model that takes
2 ms per chunk. Timing checks use wide margins, and the thread join has a
5 s timeout, so nothing can hang.

**ANSWER: the thread adds about 0.1 ms per request (max about 0.3 ms)
on the instant stream and at most about 3-4 ms at 2 ms per chunk, well
inside 50 ms.** All 60 requests give the same final action, output and
during-gen verdict in both modes.

**FINDING: on both early stops the threaded model generates 2 chunks where
the synchronous one generates 1.** The caller has already started the next
chunk before the worker has scanned the last one. That chunk never reaches
the user, but its generation time is added to the latency.

**FINDING: so the cost of an early stop is (extra chunks) x (chunk time).**
At 60 ms per chunk both early stops cost about 60-67 ms, which is over the
budget. With a queue of 1, up to 2 extra chunks are possible, so 50 ms
holds for any chunk time under 25 ms.

**FINDING: the regex sweep costs about 1-5 us per chunk, against 2,000 us
to generate one.** The thread has almost nothing to overlap with
generation, so it cannot cut latency. It can only add to it.
"""

from __future__ import annotations

import contextlib
import queue
import statistics
import sys
import threading
import time
import timeit

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "87-end-to-end-safety-gate"
BUDGET_MS, PACE_S, SLOW_S, QSIZE, REPEATS, TIMEOUT_S = 50.0, 0.002, 0.06, 1, 5, 5.0
END = object()


def load():
    names = ("mock_llm_stream", "safety_gate")
    saved = {n: sys.modules.get(n) for n in names}
    try:
        for n in names:
            sys.modules[n] = parity.load_reference(PHASE, LESSON, n)
        return sys.modules["safety_gate"], parity.load_reference(PHASE, LESSON, "main")
    finally:
        for n, mod in saved.items():
            sys.modules.pop(n) if mod is None else sys.modules.__setitem__(n, mod)


def counted(chunks, box, pace):
    """The model: counts every chunk it is asked for, taking `pace` seconds per chunk."""
    for chunk in chunks:
        time.sleep(pace)
        box["pulled"] = box.get("pulled", 0) + 1
        yield chunk


def _put(q, item, done):
    while not done.is_set():
        with contextlib.suppress(queue.Full):
            return q.put(item, timeout=0.01)


def threaded_during(during_gen, chunks):
    """The lesson's own `_during_gen` in a worker thread; the caller's thread drives the model."""
    q, done, box = queue.Queue(maxsize=QSIZE), threading.Event(), {}

    def work():
        box["out"] = during_gen(iter(lambda: q.get(timeout=TIMEOUT_S), END))
        done.set()
    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    source = iter(chunks)
    while not done.is_set() and (chunk := next(source, END)) is not END:
        _put(q, chunk, done)
    _put(q, END, done)
    worker.join(TIMEOUT_S)
    if worker.is_alive():
        raise TimeoutError("during-gen worker did not finish")
    return box["out"]


def run(sg, gate, prompt, threaded, pace):
    """gate.handle with the lesson's `stream` counted, and `_during_gen` optionally threaded."""
    box, original, lesson_during = {}, sg.stream, gate._during_gen
    sg.stream = lambda p: counted(original(p), box, pace)
    if threaded:
        gate._during_gen = lambda chunks: threaded_during(lesson_during, chunks)
    try:
        trace = gate.handle(prompt)
    finally:
        sg.stream, gate._during_gen = original, lesson_during
    return trace.latency_ms, (trace.final_action, trace.final_output, trace.during_gen), box.get("pulled", 0)


def compare(sg, gate, prompts, pace, repeats):
    rows = []
    for p in prompts:
        sync, thr = ([run(sg, gate, p, mode, pace) for _ in range(repeats)] for mode in (False, True))
        rows.append({"overhead": min(t[0] for t in thr) - min(s[0] for s in sync), "same": sync[0][1] == thr[0][1],
                     "early": sync[0][1][2].terminated_early, "pulled": sync[0][2], "extra": thr[0][2] - sync[0][2]})
    return rows


def scan_us(gate, sg):
    chunks = list(sg.stream("Recommend a vegetarian dinner using lentils."))
    return timeit.timeit(lambda: gate._during_gen(iter(chunks)), number=200) / 200 / len(chunks) * 1e6


def solve():
    sg, main = load()
    gate = sg.SafetyGate()
    prompts = [str(f["prompt"]) for f in main.load_fixtures()] + list(main.BENIGN_PROMPTS)
    paced = compare(sg, gate, prompts, PACE_S, 1)
    stops = [p for p, r in zip(prompts, paced) if r["early"]]
    return {"instant": compare(sg, gate, prompts, 0.0, REPEATS), "paced": paced,
            "slow": compare(sg, gate, stops, SLOW_S, 1), "scan_us": scan_us(gate, sg)}


def summary(rows):
    over = [r["overhead"] for r in rows]
    return round(statistics.median(over), 3), round(max(over), 3)


def slow_ok(rows):
    return len(rows) == 2 and all(abs(r["overhead"] - r["extra"] * SLOW_S * 1e3) < 20.0 for r in rows)


def early_ok(early):
    return len(early) == 2 and all(s == 1 and 0 <= x <= QSIZE + 1 for s, x in early)


def verify(result):
    inst, paced, slow, us = result["instant"], result["paced"], result["slow"], result["scan_us"]
    early = [(r["pulled"], r["extra"]) for r in paced if r["early"]]
    same = sum(r["same"] for r in inst + paced)
    over = f"(median, max) ms: instant {summary(inst)}, {PACE_S * 1e3:.0f} ms/chunk {summary(paced)}"
    slow_d = [(round(r["overhead"], 1), r["extra"]) for r in slow]
    C = practice.Check
    return [
        C(f"ANSWER: threaded during-gen adds well under {BUDGET_MS:.0f} ms per request, instant and paced",
          max(summary(inst)[1], summary(paced)[1]) < BUDGET_MS, over),
        C("ANSWER: all 60 requests give the same action, output and during-gen verdict in both modes",
          (same, len(inst)) == (120, 60), f"identical {same}/120"),
        C(f"FINDING: on the 2 early stops the thread lets the model run 0-{QSIZE + 1} chunks past the match",
          early_ok(early), f"(sync chunks, extra threaded chunks): {early}"),
        C(f"FINDING: at {SLOW_S * 1e3:.0f} ms/chunk an early stop costs (extra chunks) x chunk time",
          slow_ok(slow), f"(overhead ms, extra chunks): {slow_d}"),
        C("FINDING: the regex sweep costs microseconds per chunk, so the thread has nothing to overlap",
          us < 1000.0, f"lesson _during_gen: {us:.1f} us per chunk vs {PACE_S * 1e6:.0f} us to generate one"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
