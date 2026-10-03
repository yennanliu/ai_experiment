"""Exercise 3 — one snapshot at the end names the wrong line.

    Use `tracemalloc` to find which line in your data loading pipeline allocates the
    most memory.

Reading of the exercise: the pipeline is the usual three lines -- split a CSV text
into lines, parse every field to a Python float, stack into a float32 array --
over 2000 rows x 32 columns of generated CSV. "Allocates the most" is measured two
ways: the lesson's own recipe (start, run, one `take_snapshot()`, top of
`statistics("lineno")`), and a per-line peak from `get_traced_memory()` with
`reset_peak()` before each line. The lesson's own `demo_memory_tracking` is run
too, on its no-torch path.

**ANSWER: the float-parsing line, at about 2.1 MiB** -- 6.8x the `np.array`
line's 313 KiB and 3.6x the 594 KiB of CSV text it parses. Every value becomes a
24-byte float object plus an 8-byte list slot, against 4 bytes in float32.

**FINDING: the lesson's recipe names a different line.** One snapshot after the
pipeline returns sees only what is still alive, so its top line is
`np.array(...)` at 250 KiB; the parsed lists were freed on return, and the
parsing line shows 6.6 KiB of its 2.1 MiB. The process peak (3.05 MiB) is 12x
what the top line shows.

**FINDING: the lesson's own demo is a tie.** On its no-torch path the two lines
allocate 100 x `bytearray(4*100*100)` and one `bytearray(4*1000*1000)`: exactly
4,000,000 payload bytes each. The listing ranks the list first by 0.2%, which
is the 100 object headers and the list, not data.

**CONTROL:** with the pipeline's intermediates kept alive, the end snapshot's top
line is the parsing line, matching the per-line peak.

Structure: `make_csv` is the fixture; `PIPELINE` is the three lines;
`per_line_peak` and `end_snapshot` are the two measurements.
"""

from __future__ import annotations

import contextlib
import io
import random
import re
import tracemalloc

import numpy as np

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "12-debugging-and-profiling"
ROWS, COLS = 2000, 32
UNITS = {"B": 1, "KiB": 2**10, "MiB": 2**20}
PIPELINE = (
    ("lines = text.splitlines()", lambda s: s["text"].splitlines()),
    ("rows = [[float(v) ...]]", lambda s: [[float(v) for v in ln.split(",")] for ln in s["lines"]]),
    ("data = np.array(rows, float32)", lambda s: np.array(s["rows"], dtype=np.float32)),
)


def make_csv():
    rng = random.Random(0)
    return "\n".join(",".join(f"{rng.gauss(0, 1):.6f}" for _ in range(COLS)) for _ in range(ROWS))


def run_pipeline(text, keep=False, probe=None):
    state = {"text": text}
    for (label, line), key in zip(PIPELINE, ("lines", "rows", "data")):
        start = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        state[key] = line(state)
        if probe is not None:
            probe[label] = tracemalloc.get_traced_memory()[1] - start
    return state if keep else state["data"]


def per_line_peak(text):
    probe = {}
    tracemalloc.start()
    try:
        run_pipeline(text, probe=probe)
    finally:
        tracemalloc.stop()
    return probe


def end_snapshot(text, keep):
    """The lesson's recipe: start, run, one snapshot. Returns {line: bytes} in rank order, peak."""
    tracemalloc.start()
    try:
        alive = run_pipeline(text, keep)
        stats = tracemalloc.take_snapshot().statistics("lineno")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    del alive
    return {s.traceback[0].lineno: s.size for s in stats}, peak


def lesson_demo(ref):
    """The lesson's own demo on its no-torch path, as (line, bytes) of its top rows."""
    ref.HAS_TORCH, out = False, io.StringIO()
    with contextlib.redirect_stdout(out):
        ref.demo_memory_tracking()
    found = re.findall(r"debug_tools\.py:(\d+): size=([\d.]+) (B|KiB|MiB)", out.getvalue())
    return [(int(ln), float(v) * UNITS[unit]) for ln, v, unit in found]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "debug_tools")
    text = make_csv()
    lines = {label: run.__code__.co_firstlineno for label, run in PIPELINE}
    return {
        "text": len(text),
        "peak": per_line_peak(text),
        "freed": end_snapshot(text, keep=False),
        "kept": end_snapshot(text, keep=True),
        "lines": lines,
        "lesson": lesson_demo(ref),
    }


def verify(result):
    peak, lines = result["peak"], result["lines"]
    parse, stack = PIPELINE[1][0], PIPELINE[2][0]
    top = max(peak, key=peak.get)
    (freed, peak_bytes), (kept, _) = result["freed"], result["kept"]
    freed_top, kept_top = next(iter(freed.items())), next(iter(kept.items()))
    left = freed.get(lines[parse], 0)
    (line_a, size_a), (line_b, size_b) = result["lesson"][:2]
    return [
        practice.Check(
            "ANSWER: the float-parsing line allocates the most",
            top == parse and peak[parse] > 5 * peak[stack],
            "per-line peak: " + "; ".join(f"{k} {v / 2**10:.0f} KiB" for k, v in peak.items())
            + f"; parsing is {peak[parse] / peak[stack]:.1f}x the array and "
            f"{peak[parse] / result['text']:.1f}x the {result['text'] / 2**10:.0f} KiB of text",
        ),
        practice.Check(
            "FINDING: one snapshot at the end names np.array, not the parsing line",
            freed_top[0] == lines[stack] and left < 0.01 * peak[parse]
            and peak_bytes > 5 * freed_top[1],
            f"end snapshot's top line is {stack!r} at {freed_top[1] / 2**10:.0f} KiB; the "
            f"parsing line shows {left / 2**10:.1f} KiB of its {peak[parse] / 2**10:.0f} KiB; the "
            f"process peak was {peak_bytes / 2**20:.2f} MiB, {peak_bytes / freed_top[1]:.0f}x it",
        ),
        practice.Check(
            "FINDING: the lesson's own demo is a 0.2% tie between two 4,000,000-byte lines",
            abs(size_a / size_b - 1) < 0.01 and min(size_a, size_b) > 3.9e6,
            f"demo_memory_tracking (no-torch path): line {line_a} {size_a / 2**10:.0f} KiB, "
            f"line {line_b} {size_b / 2**10:.0f} KiB, {size_a / size_b - 1:+.2%} apart",
        ),
        practice.Check(
            "CONTROL: with intermediates kept alive, the snapshot agrees with the peak",
            kept_top[0] == lines[parse],
            f"top line {kept_top[0]} ({parse!r}) at {kept_top[1] / 2**10:.0f} KiB",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
