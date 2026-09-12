"""Exercise 3 — the sweep ends exactly where real attention stops fitting.

    **Hard.** Port the attention-style reduction to PyTorch on GPU. Time both as
    you sweep sequence length from 64 to 65,536. Plot and explain the curve
    shape.

Reading of the exercise: `torch` and `jax` both return None from `find_spec` and
the host is arm64 with no CUDA device, so "PyTorch on GPU" is unbuildable twice
over. The substitution is numpy: one C-level reduction with no Python work per
element, which is the same dependency-graph argument as a GPU kernel at a smaller
constant, and it keeps the thing being measured -- overhead versus bandwidth --
intact. Both arms are swept over the exact range asked for, 2^6 to 2^16.

**ANSWER: a hockey stick, and the flat part is not parallelism.** Fit the
log-log slope and the shape is a number: numpy's mean runs at slope **-0.04**
from N=64 to 1024 -- 16x the data for the same microsecond -- and **0.73** from
8192 to 65536, for **0.32** over the whole sweep. The plateau is fixed per-call
dispatch cost, so the left half of the curve times the call and only the right
half times the reduction. A GPU has the same shape with a longer plateau: a
kernel launch is microseconds, so the knee moves right, and reading the flat
part as "the GPU is parallel" reads overhead.

**FINDING: nothing in the sweep is quadratic.** `attention_style` is
`sum(xs) / len(xs)` -- a mean. O(N) time, O(1) extra memory, no N-by-N anything.
The doc says the gap widens "until you hit the O(N^2) memory wall of attention",
and at the sweep's own endpoint a real N-by-N attention matrix is 65536^2 * 4 =
**16 GiB per head per layer**, against 512 KiB for the array being swept. The
exercise stops precisely where the wall it names begins, holding a proxy that
cannot feel it.

**FINDING: the recurrent arm has no curve at all.** `rnn_style` fits slope
**1.01** across the same 1024x range -- a ruler -- so every bend in the plot
belongs to the reduction's overhead and none of it to recurrence.

**CONTROL: "beats it at length >= 1,000" is off by a factor of 16.** Step 1
claims the attention-style reduction overtakes the RNN at N >= 1000. Measured in
the same pure Python, it already wins at N=64 by ~4x and the ratio only grows.

Structure: `best` is `benchmark()`'s best-of-three; `sweep` walks 2^6..2^16;
`slope` fits the log-log exponent; `plot` is the requested plot as a text
scatter on the same axes.
"""

from __future__ import annotations

import importlib.util
import math
import time

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "01-why-transformers"
LO, HI, ROWS, FLAT_HI, RAMP_LO = 6, 16, 14, 1024, 8192


def best(call, reps=5):
    """`benchmark()`'s protocol, one more rep because these are microseconds."""
    def once():
        start = time.perf_counter()
        call()
        return time.perf_counter() - start
    return min(once() for _ in range(reps))


def sweep(ref):
    """{N: (rnn, pure-Python mean, numpy mean)} over 2^LO .. 2^HI."""
    import numpy
    out = {}
    for power in range(LO, HI + 1):
        xs = [0.001 * (i % 17) for i in range(2 ** power)]
        array = numpy.array(xs)
        out[2 ** power] = (best(lambda: ref.rnn_style(xs)), best(lambda: ref.attention_style(xs)),
                           best(lambda: array.mean()))
    return out


def slope(timings, column, lo=0, hi=2 ** HI):
    """Least-squares exponent of t vs N in log-log: 1 is linear, 0 is a plateau."""
    xs = [math.log2(n) for n in timings if lo <= n <= hi]
    ys = [math.log2(row[column]) for n, row in timings.items() if lo <= n <= hi]
    n, sx, sy = len(xs), sum(xs), sum(ys)
    return (n * sum(x * y for x, y in zip(xs, ys)) - sx * sy) / (n * sum(x * x for x in xs) - sx ** 2)


def plot(timings, columns=(0, 2), marks="Rn"):
    """The requested plot: log-log text scatter, time up, sequence length right."""
    times = [row[c] for row in timings.values() for c in columns]
    top, bottom = math.log10(max(times)), math.log10(min(times))
    grid, axis = [[" "] * (HI - LO + 1) for _ in range(ROWS)], ""
    for column, mark in zip(columns, marks):
        for index, (_, row) in enumerate(sorted(timings.items())):
            depth = (top - math.log10(row[column])) / (top - bottom) * (ROWS - 1)
            grid[min(ROWS - 1, max(0, round(depth)))][index] = mark
    axis += "".join(str(power % 10) for power in range(LO, HI + 1))
    return "\n".join(["".join(line) for line in grid] + [axis + "   (x = log2 N, y = log10 s)"])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    timings = sweep(ref)
    return {
        "timings": timings, "plot": plot(timings),
        "flat": slope(timings, 2, hi=FLAT_HI), "ramp": slope(timings, 2, lo=RAMP_LO),
        "whole": slope(timings, 2), "rnn_slope": slope(timings, 0),  # log-log exponents
        "missing": [m for m in ("torch", "jax") if importlib.util.find_spec(m) is None],
        "python_wins": [n for n, row in timings.items() if row[1] < row[0]],
        "attention_bytes": (2 ** HI) ** 2 * 4, "array_bytes": 2 ** HI * 8,  # fp32 vs float64
    }


def verify(result):
    timings, span = result["timings"], 2 ** HI // 2 ** LO
    return [
        practice.Check(
            "CONTROL: 'PyTorch on GPU' is unbuildable twice over, so numpy is the arm",
            result["missing"] == ["torch", "jax"],
            f"find_spec is None for {result['missing']} and the host has no CUDA device, so both "
            "halves are absent. numpy's mean is one C-level reduction with no Python work per "
            "element -- a GPU kernel's dependency-graph claim at a smaller constant",
        ),
        practice.Check(
            "ANSWER: a hockey stick whose flat part is dispatch cost, not parallelism",
            abs(result["flat"]) < 0.25 and result["ramp"] > 0.5,
            f"log-log slope of numpy's mean: {result['flat']:+.2f} over N=64..{FLAT_HI} "
            f"({timings[2 ** LO][2] * 1e6:.2f} -> {timings[FLAT_HI][2] * 1e6:.2f} us, 16x the data "
            f"for one microsecond), {result['ramp']:+.2f} over {RAMP_LO}..{2 ** HI}, "
            f"{result['whole']:+.2f} overall. Left half times the call, right half the reduction",
        ),
        practice.Check(
            "FINDING: nothing in the swept quantity is quadratic",
            result["attention_bytes"] > 10_000 * result["array_bytes"],
            f"attention_style is sum(xs)/len(xs), a mean: O(N) time, O(1) extra memory. At the "
            f"sweep's endpoint the array is {result['array_bytes'] / 2 ** 10:.0f} KiB while a real "
            f"N-by-N attention matrix is {result['attention_bytes'] / 2 ** 30:.0f} GiB per "
            "head-layer -- the doc's 'O(N^2) wall' starts where this sweep stops",
        ),
        practice.Check(
            "FINDING: the recurrent arm is a straight line, so the bend is all overhead",
            0.85 < result["rnn_slope"] < 1.15,
            f"rnn_style fits slope {result['rnn_slope']:.2f} across the same {span}x range "
            f"({timings[2 ** LO][0] * 1e6:.2f} -> {timings[2 ** HI][0] * 1e6:.0f} us): an exponent "
            f"of 1 is a ruler, against {result['whole']:.2f} for the reduction, so every bend in "
            "the plot is the plateau and recurrence contributes none of it",
        ),
        practice.Check(
            "CONTROL: 'beats it at length >= 1,000' is off by a factor of 16",
            len(result["python_wins"]) == HI - LO + 1,
            f"Step 1 says the reduction overtakes the RNN at N >= 1000; it already wins at N=64 "
            f"by {timings[2 ** LO][0] / timings[2 ** LO][1]:.1f}x and at all "
            f"{len(result['python_wins'])} lengths swept, because sum() is C -- which has nothing "
            "to do with the dependency graph the lesson is selling",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
