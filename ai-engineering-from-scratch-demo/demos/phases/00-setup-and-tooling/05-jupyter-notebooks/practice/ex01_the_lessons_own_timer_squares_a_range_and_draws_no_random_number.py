"""Exercise 1 — the lesson's own timer squares a range and draws no random number.

    Open JupyterLab, create a notebook, and use `%timeit` to compare list
    comprehension vs numpy for creating an array of 100,000 random numbers

Reading of the exercise: `%timeit` is IPython's wrapper around the stdlib
`timeit` module -- it picks a loop count with `autorange` and repeats -- so the
same measurement runs here without a kernel. The two contestants are
`[random.random() for _ in range(n)]` and `np.random.rand(n)` at n = 100,000,
timed best-of-REPEATS. The lesson's own `timing_comparison` is then run and
read, because it is the code a learner would paste into that notebook cell.

**ANSWER: numpy is ~11x faster.** On the machine that wrote this, the list
comprehension takes **~2.1 ms** and `np.random.rand` **~0.18 ms**. The ratio is
wall-clock and varies by host, which is why this exercise is T1 and the check
asks only for numpy to win by 2x.

**FINDING: the lesson's own timer measures a different thing.**
`timing_comparison` squares `range(1_000_000)` against `np.arange(size) ** 2`:
its source contains no random draw at all, uses ten times the exercise's n, and
times each side with one `perf_counter` shot rather than `timeit`'s repeats.
It printed **13.0x** and **13.6x** on two runs here, while `timeit` on the
very same squaring workload gives **~22x**: the single shot understates even its
own question, and that question is not the exercise's.

**FINDING: the two sides do not build the same object.** The list holds
100,000 boxed floats -- **3.20 MB** by `sys.getsizeof`, exactly **4.0x** the
array's **0.80 MB** (24 bytes per float object plus an 8-byte pointer, against 8
bytes raw). If an array is what the next cell needs, the list route also pays
`np.array(...)`, which makes the honest comparison wider than the headline one.

**CONTROL: both contestants produce 100,000 floats in [0, 1).**

Structure: `load_lesson` imports the lesson with its plotting imports stubbed
when absent; `best` is `%timeit`'s measurement; `solve` times all three routes.
"""

from __future__ import annotations

import contextlib
import importlib.util
import inspect
import io
import re
import sys
import types

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "05-jupyter-notebooks"
N, REPEATS = 100_000, 5


def load_lesson():
    """The lesson's module; matplotlib and pandas are stubbed only if absent.

    The functions used here are numpy-only, and the stubs leave sys.modules after
    import so nothing else in the process sees them.
    """
    added = []
    if importlib.util.find_spec("matplotlib") is None:
        mpl = types.ModuleType("matplotlib")
        mpl.use, mpl.pyplot = (lambda *a, **k: None), types.ModuleType("matplotlib.pyplot")
        sys.modules.update({"matplotlib": mpl, "matplotlib.pyplot": mpl.pyplot})
        added += ["matplotlib", "matplotlib.pyplot"]
    if importlib.util.find_spec("pandas") is None:
        sys.modules["pandas"] = types.ModuleType("pandas")
        added.append("pandas")
    try:
        return parity.load_reference(PHASE, LESSON, "notebook_tips")
    finally:
        for name in added:
            sys.modules.pop(name, None)


def best(stmt, names):
    """Seconds per call, as %timeit measures it: autorange, then best of REPEATS."""
    import timeit
    timer = timeit.Timer(stmt, globals=names)
    loops, _ = timer.autorange()
    return min(timer.repeat(REPEATS, loops)) / loops


def solve():
    import random

    import numpy as np
    ref = load_lesson()
    random.seed(0)
    np.random.seed(0)
    names = {"random": random, "np": np, "n": N}
    listed = [random.random() for _ in range(N)]
    array = np.random.rand(N)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ref.timing_comparison()
    return {
        "list_s": best("[random.random() for _ in range(n)]", names),
        "numpy_s": best("np.random.rand(n)", names),
        "convert_s": best("np.array([random.random() for _ in range(n)])", names),
        "squares_ratio": best("[x ** 2 for x in range(1_000_000)]", names)
        / best("np.arange(1_000_000) ** 2", names),
        "source": inspect.getsource(ref.timing_comparison),
        "lesson_speedup": float(re.search(r"Speedup:\s+([\d.]+)x", out.getvalue()).group(1)),
        "list_bytes": sys.getsizeof(listed) + sum(sys.getsizeof(x) for x in listed),
        "array_bytes": array.nbytes,
        "ranges": [(len(listed), min(listed), max(listed)),
                   (array.size, float(array.min()), float(array.max()))],
    }


def verify(result):
    ratio = result["list_s"] / result["numpy_s"]
    source = result["source"]
    mem = result["list_bytes"] / result["array_bytes"]
    return [
        practice.Check(
            "ANSWER: numpy wins, by an order of magnitude",
            ratio > 2,
            f"timeit best-of-{REPEATS}: list comprehension {result['list_s'] * 1e3:.2f} ms, "
            f"np.random.rand {result['numpy_s'] * 1e3:.3f} ms, so numpy is {ratio:.1f}x faster "
            f"for {N:,} random numbers. The ratio is wall-clock and host-dependent",
        ),
        practice.Check(
            "FINDING: the lesson's own timer squares a range and draws no random number",
            "random" not in source and "range(size)" in source and "1_000_000" in source,
            "timing_comparison times [x ** 2 for x in range(1_000_000)] against "
            "np.arange(size) ** 2 with one perf_counter shot each, not timeit's repeats, at ten "
            f"times the exercise's n. It printed {result['lesson_speedup']:.1f}x here, while "
            f"timeit on that same squaring gives {result['squares_ratio']:.1f}x: one shot is not "
            "%timeit, and the squaring is not the exercise's question either",
        ),
        practice.Check(
            "FINDING: the two sides do not build the same object",
            3.9 < mem < 4.1 and result["convert_s"] > result["list_s"],
            f"the list holds {N:,} boxed floats, {result['list_bytes'] / 1e6:.2f} MB, which is "
            f"{mem:.2f}x the array's {result['array_bytes'] / 1e6:.2f} MB. If the next cell "
            f"needs an array the list route also pays np.array(...): "
            f"{result['convert_s'] * 1e3:.2f} ms in all, "
            f"{result['convert_s'] / result['numpy_s']:.0f}x numpy's time",
        ),
        practice.Check(
            "CONTROL: both contestants produce 100,000 floats in [0, 1)",
            all(n == N and 0 <= lo and hi < 1 for n, lo, hi in result["ranges"]),
            "(count, min, max) = "
            + str([(n, round(lo, 6), round(hi, 6)) for n, lo, hi in result["ranges"]]),
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
