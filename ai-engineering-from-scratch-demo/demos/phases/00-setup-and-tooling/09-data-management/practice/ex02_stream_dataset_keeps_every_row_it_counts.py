"""Exercise 2 — stream_dataset keeps every row it reads, and max_rows=0 still reads one.

    Stream the `c4` dataset and count how many examples you can process in 10 seconds

Reading of the exercise: with no network, `c4` is a stubbed endless stream of
C4-shaped rows (`text, timestamp, url`, a 2,000-character synthetic text each)
handed to the lesson's own `stream_dataset` through a stubbed `load_dataset`.
Wall-clock throughput is the network's, not the code's, so the 10-second count
runs on an injected clock that charges a fixed 4 ms per example; the real
number is `10 / (seconds per example)` on whatever link you have.

**ANSWER: 2,500 examples in 10 seconds at 4 ms each, counted by a loop that
stops on the clock.** The lesson's helper cannot answer the question as asked:
`stream_dataset` takes `max_rows`, not a time budget, so a count has to be
guessed in advance.

**FINDING: the helper holds every row it streams.** The doc says streaming
keeps "memory usage ... constant regardless of dataset size", but
`stream_dataset` returns a list: 2,000 rows retain about 10x the memory of 200
(4.64 MB against 0.46 MB here), while a counting loop over the same stream
retains 0 bytes at either length. Counting C4 for 10 s through the helper means
keeping all of it.

**FINDING: `max_rows=0` returns one row, and so does `-5`.** The break test is
`i >= max_rows - 1` after the append, so the first row is always kept.

**CONTROL: for positive `max_rows` it reads exactly that many.** The stub's
pull counter shows 5 rows read for `max_rows=5`, no extra fetch, and the call
was `load_dataset(path="c4", split="train", streaming=True)`.

Structure: `stream` is the stubbed C4; `count_within` is the timed count.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import sys
import tracemalloc
import types
from unittest import mock

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "09-data-management"
BUDGET, COST = 10.0, 0.004


def stream(pulls):
    """An endless C4-shaped stream; `pulls[0]` counts rows handed out."""
    for i in itertools.count():
        pulls[0] += 1
        yield {"text": f"SYNTHETIC c4 document {i} " + "x" * 2000,
               "timestamp": "2019-04-25T12:57:54Z", "url": f"https://example.invalid/{i}"}


def load_lesson(calls, pulls):
    def load_dataset(**kwargs):
        calls.append(kwargs)
        return stream(pulls)
    hf = types.ModuleType("datasets")
    hf.load_dataset, hf.Dataset = load_dataset, None
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = None
    with mock.patch.dict(sys.modules, {"datasets": hf, "huggingface_hub": hub}):
        return parity.load_reference(PHASE, LESSON, "data_utils")


def count_within(rows, budget, cost):
    """Count rows until an injected clock (cost seconds per row) passes budget."""
    clock = count = 0
    for _ in rows:
        clock += cost
        if clock > budget:
            break
        count += 1
    return count


def retained(fn):
    """Bytes still allocated by fn's return value."""
    tracemalloc.start()
    kept = fn()
    size = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    del kept
    return size


def helper(ref, n, pulls=None):
    if pulls:
        pulls[0] = 0
    with contextlib.redirect_stdout(io.StringIO()):
        return ref.stream_dataset("c4", max_rows=n)


def solve():
    calls, pulls = [], [0]
    ref = load_lesson(calls, pulls)
    five = len(helper(ref, 5, pulls))
    return {
        "count": count_within(stream([0]), BUDGET, COST),
        "five": (five, pulls[0]),
        "call": calls[0],
        "degenerate": {n: len(helper(ref, n)) for n in (0, -5)},
        "helper_mem": {n: retained(lambda: helper(ref, n)) for n in (200, 2000)},
        "loop_mem": {n: retained(lambda: sum(1 for _ in itertools.islice(stream([0]), n)))
                     for n in (200, 2000)},
        "doc_says_constant": "Memory usage stays constant" in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    hm, lm = result["helper_mem"], result["loop_mem"]
    five, pulled = result["five"]
    return [
        practice.Check(
            "ANSWER: 2,500 examples in 10 s at 4 ms each, by a clock-bounded loop",
            result["count"] == round(BUDGET / COST),
            f"count_within(stream, {BUDGET} s, {COST * 1000:.0f} ms/row) = {result['count']}; "
            "stream_dataset takes max_rows, not a time budget",
        ),
        practice.Check(
            "FINDING: the helper keeps every streamed row; the doc says memory is constant",
            result["doc_says_constant"] and 8 < hm[2000] / hm[200] < 12
            and max(lm.values()) < 10_000,
            f"stream_dataset retains {hm[200] / 1e6:.2f} MB at 200 rows and "
            f"{hm[2000] / 1e6:.2f} MB at 2000 ({hm[2000] / hm[200]:.1f}x); a counting loop "
            f"retains {lm[200]} and {lm[2000]} bytes",
        ),
        practice.Check(
            "FINDING: max_rows=0 and max_rows=-5 each return one row",
            result["degenerate"] == {0: 1, -5: 1},
            f"rows returned: {result['degenerate']} -- the break test runs after the append",
        ),
        practice.Check(
            "CONTROL: max_rows=5 reads exactly 5, via a streaming load_dataset call",
            (five, pulled) == (5, 5)
            and result["call"] == {"path": "c4", "split": "train", "streaming": True},
            f"returned {five}, pulled {pulled}; call was {result['call']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
