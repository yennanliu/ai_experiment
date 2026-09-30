"""Exercise 1 -- 1F1B matches GPipe's bubble on 42 grids and holds min(N, M) microbatches instead of M.

    Implement 1F1B and verify the bubble fraction matches GPipe but activation memory is bounded.

Reading of the exercise: both schedules are written as a per-stage order of
ops and run through one dependency-driven simulator (F of a microbatch waits
for the previous stage's F, B waits for the next stage's B, each stage runs
one op at a time). GPipe's order is taken from the lesson's own
`gpipe_schedule`; 1F1B is the PipeDream-Flush order (warm up N-1-s
forwards, then alternate, then drain). "Bubble" is the lesson's measure:
idle slots over all slots. "Activation memory" is the peak number of
microbatches a stage has run forward on but not yet backward, since each one
holds its stashed activations.

**ANSWER: 1F1B's bubble equals GPipe's and the closed form on all 42 (N, M)
grids; its peak in-flight count is min(N, M), where GPipe's is M.** The
simulator reproduces the lesson's GPipe cycles exactly. At N=4 the first
stage holds 4 microbatches under 1F1B at every M >= 4, against 8 at M=8 and
64 at M=64 under GPipe.

**FINDING: the lesson's backward-costs-2x constants are never used.**
`FORWARD_UNITS = 1` and `BACKWARD_UNITS = 2` are defined and read nowhere;
`gpipe_schedule` is unit-time. Re-run with B = 2 units, both schedules still
give exactly (N-1)/(M+N-1), so the closed form survives the omission.

**FINDING: the doc's numbers use a different denominator from the Megatron
paper it cites, and "interleaved 1F1B" is a third schedule.** The lesson
divides the bubble by total time, 3/11 = 27.3% at N=4, M=8. Narayanan et
al. 2021 (https://arxiv.org/html/2104.04473, read 2026-09-29) divide by
ideal time, (p-1)/m = 37.5%. Their interleaved schedule with v chunks cuts
that by v, to 18.75% at v = 2, so it does not "match GPipe". The doc also
names `PipelineStage` and `Pipeline(stages, num_microbatches)`, and
`main.py` defines neither.

Expected output: three PASS checks.
"""

from __future__ import annotations

import inspect
import itertools

from harness import parity, practice

try:
    import torch  # noqa: F401  (the lesson module imports torch at load time)
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "79-pipeline-parallel"
GRID = [(n, m) for n in (1, 2, 3, 4, 6, 8) for m in (1, 2, 4, 7, 8, 16, 64)]


def gpipe_orders(ref, n, m):
    """Each stage's op order, read off the lesson's own GPipe schedule."""
    sched = sorted(ref.gpipe_schedule(n, m))
    return [[(ph, mb) for _, st, mb, ph in sched if st == s] for s in range(n)], sched


def one_f_one_b_orders(n, m):
    orders = []
    for s in range(n):
        warm = min(n - 1 - s, m)
        ops = [("F", i) for i in range(warm)]
        for i in range(m - warm):
            ops += [("F", warm + i), ("B", i)]
        orders.append(ops + [("B", i) for i in range(m - warm, m)])
    return orders


def step(op, s, n, times, tf, tb):
    """Start op on stage s if its dependency has finished; returns whether it started."""
    (ph, mb), (end, start, free) = op, times
    dep = (s - 1, "F", mb) if ph == "F" else ((s, "F", mb) if s == n - 1 else (s + 1, "B", mb))
    ready = 0 if dep[0] < 0 else end.get(dep)
    if ready is None:
        return False
    start[(s, ph, mb)] = t0 = max(ready, free[s])
    end[(s, ph, mb)] = free[s] = t0 + (tf if ph == "F" else tb)
    return True


def simulate(orders, tf=1, tb=1):
    """Earliest-start times for per-stage op orders; returns (makespan, idle fraction, starts)."""
    n, pos = len(orders), [0] * len(orders)
    times = ({}, {}, [0] * n)
    while any(p < len(o) for p, o in zip(pos, orders)):
        for s in range(n):
            if pos[s] < len(orders[s]) and step(orders[s][pos[s]], s, n, times, tf, tb):
                pos[s] += 1
    span = max(times[2])
    return span, 1 - sum(map(len, orders)) * (tf + tb) / 2 / (n * span), times[1]


def peak(order):
    """Most microbatches run forward but not yet backward, i.e. holding stashed activations."""
    return max(itertools.accumulate(1 if ph == "F" else -1 for ph, _ in order))


def close(a, b):
    return abs(a - b) < 1e-12


def grid_row(ref, n, m):
    g, sched = gpipe_orders(ref, n, m)
    f, closed = one_f_one_b_orders(n, m), ref.bubble_fraction(n, m)
    _, g_bub, starts = simulate(g)
    return {
        "n": n, "m": m, "exact": all(starts[(s, ph, mb)] == c for c, s, mb, ph in sched),
        "same": close(g_bub, closed) and close(simulate(f)[1], closed),
        "tb2": close(simulate(g, tb=2)[1], closed) and close(simulate(f, tb=2)[1], closed),
        "peaks": (max(map(peak, g)), max(map(peak, f))),
    }


def doc_facts(ref):
    src = inspect.getsource(ref)
    return {
        "unit_uses": [src.count(k) for k in ("FORWARD_UNITS", "BACKWARD_UNITS")],
        "lesson_48": round(ref.bubble_fraction(4, 8) * 100, 1), "megatron_48": 3 / 8 * 100,
        "interleaved_v2": 3 / 8 / 2 * 100,
        "doc_says_matches": "interleaved 1F1B schedule used in Megatron-LM" in parity.doc_text(PHASE, LESSON),
        "missing": [k for k in ("PipelineStage", "Pipeline") if not hasattr(ref, k)],
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = [grid_row(ref, n, m) for n, m in GRID]
    return {
        "grids": len(rows), "exact": all(r["exact"] for r in rows),
        "same": sum(r["same"] for r in rows), "tb2": sum(r["tb2"] for r in rows),
        "bounded": all(r["peaks"] == (r["m"], min(r["n"], r["m"])) for r in rows),
        "n4": {r["m"]: r["peaks"] for r in rows if r["n"] == 4}, **doc_facts(ref),
    }


def verify(r):
    return [
        practice.Check(
            "ANSWER: 1F1B's bubble equals GPipe's; its peak in-flight count is min(N, M), GPipe's is M",
            r["same"] == r["grids"] == 42 and r["bounded"] and r["exact"] and r["n4"][64] == (64, 4),
            f"bubble equal to (N-1)/(M+N-1) on {r['same']}/{r['grids']} grids; simulator reproduces the "
            f"lesson's GPipe cycles: {r['exact']}; N=4 peak in-flight (GPipe, 1F1B) by M: {r['n4']}",
        ),
        practice.Check(
            "FINDING: the lesson's backward-costs-2x constants are never used",
            r["unit_uses"] == [1, 1] and r["tb2"] == 42,
            f"FORWARD_UNITS/BACKWARD_UNITS occurrences in main.py {r['unit_uses']} (the definitions); "
            f"with B = 2 units both schedules still hit the closed form on {r['tb2']}/42 grids",
        ),
        practice.Check(
            "FINDING: the doc's bubble uses a different denominator from Megatron, and interleaving is a third schedule",
            (r["lesson_48"], r["megatron_48"], r["interleaved_v2"]) == (27.3, 37.5, 18.75)
            and r["doc_says_matches"] and r["missing"] == ["PipelineStage", "Pipeline"],
            f"N=4, M=8: lesson {r['lesson_48']}% of total, Megatron (p-1)/m {r['megatron_48']}% of "
            f"ideal, interleaved v=2 {r['interleaved_v2']}%; doc-named symbols absent: {r['missing']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
