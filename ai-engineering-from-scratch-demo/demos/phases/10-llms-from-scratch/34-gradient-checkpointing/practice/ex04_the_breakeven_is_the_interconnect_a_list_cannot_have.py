"""Exercise 4 — offloading wins by 10x and up at every size in the toy; the breakeven is hardware.

    Add offload. Save segment inputs to a simulated "CPU buffer" (a separate
    list). Measure "PCIe bandwidth" as bytes/time and find the breakeven point
    between offload and recompute.

Reading of the exercise: the offload is built as described -- the checkpointed
forward copies each segment input into a separate list with `np.array(copy=True)`
-- and bytes/time is measured both at the size the saved tensors actually are and
on one large enough to leave cache. The recompute side is timed against it across
hidden sizes, and the breakeven is then written down as the closed form, because
the simulated buffer has no parameter in which an interconnect could appear.

**ANSWER: there is no breakeven inside the toy.** Where the arithmetic
dominates -- hidden 128 and up -- copying a layer's input is **40x to 100x**
cheaper than recomputing that layer, and the ratio *grows* with width:
recompute is `O(h*inner)` and the copy is `O(h)`. The sweep runs away from the
crossing rather than toward it.

**The narrow end of the sweep measures the interpreter, not the asymptotics.**
At hidden 8 to 64 a `layer_forward` and a pair of `np.array` copies are both
dominated by per-call overhead, so the ratio there is a property of the host
rather than of the algorithm: it is not monotone locally -- h=8 lands *above*
h=16 on every run -- and a shared CI runner measured **4x** at h=64 against
roughly 30x here. The threshold is therefore taken over the widths where the
matmul is actually the work, and the narrow widths are reported without being
asserted on.

**FINDING: bytes/time answers with the array size and the machine, not with a
link.** On the same machine the figure moves about **3x** with the size -- a
0.26 MB array copies at 69 GB/s and a 67 MB one at 22 GB/s, because one fits in
L3 and the other does not. Across machines it moves further, and it moves
*direction*: a CI runner reports **1 GB/s** for the small array against **11
GB/s** for the large one, inverting the order, because at 0.26 MB the figure is
per-call overhead wearing a bandwidth's units. So what is asserted is the
disagreement between the two sizes and not its sign -- **1.5x** apart either
way, at memory-bandwidth scale. An earlier version required the cached array to
be the faster one and failed on the host that reported it slower. PCIe gen4 x16 is a fixed 25 GB/s, and the
simulation lands above it, below it, or nowhere near it depending on what it is
run on. A list in the same address space cannot be slower than memcpy, and a
link is the only thing that makes offload a decision.

**FINDING: 3,511 FLOPs are spent per byte copied, so no clock will see the
copies.** The configuration settles this without a stopwatch: **7** saved
tensors are **3.67 MB** against **12.88 GFLOP** of forward. Across every
plausible machine balance -- 50 GFLOP/s to 1 TFLOP/s against 10 to 100 GB/s
-- that puts the copies between **0.014%** and **2.8%** of the step, and the
measured figure here sits inside that range. Timed in place the nine
interleaved pairs land tens of percent either side of zero, a spread one to
two orders of magnitude wider than the quantity, so which side any single
pair falls on is a draw rather than a result. Two earlier versions of this
check asserted such a draw -- one against a min-to-max envelope, one against
a noise ratio -- and each failed on a host whose noise happened to differ
from the one it was written on. The arithmetic does not move between hosts,
so the arithmetic is what is asserted; the timings are reported beside it.

**MECHANISM: the breakeven is `h = F / (6 * BW)`.** Recompute costs
`24*b*s*h^2 / F` and an offload round trip costs `2*(b*s*h*2) / BW`, so they
cross at a hidden size set by the machine's FLOPs-per-byte:

    PCIe gen4 x16    25 GB/s    h = 2,080
    PCIe gen5 x16    64 GB/s    h =   812
    NVLink 3        300 GB/s    h =   173
    NVLink 4        900 GB/s    h =    58

A 36x range, set entirely by the one number the "CPU buffer" does not have. At
the lesson's own defaults -- 64 layers, seq 8192, hidden 8192 -- a layer's
recompute is 42.3 ms against a 10.7 ms round trip for 134.2 MB, so offload wins
3.9x on PCIe and would lose on any hidden size below 2,080.

Structure: `offload_forward` is the exercise's own construction;
`bandwidth` times the copies in isolation and `breakeven` is the closed form.
"""

from __future__ import annotations

import statistics
import time

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "34-gradient-checkpointing"
LAYERS, HIDDEN, INNER, BATCH, SEGMENT = 24, 512, 1024, 256, 4
WIDTHS = (8, 16, 32, 64, 128, 256, 512)
CACHED, STREAM = (256, 256), (4096, 4096)
LINKS = (("PCIe gen4 x16", 25e9), ("PCIe gen5 x16", 64e9),
         ("NVLink 3", 300e9), ("NVLink 4", 900e9))
FLOPS, BYTES = 312e12, 2
SAVED = LAYERS // SEGMENT + 1
COPIED, GFLOP = SAVED * BATCH * HIDDEN * 4, LAYERS * 4 * BATCH * HIDDEN * INNER
DEFAULTS = {"layers": 64, "seq": 8192, "hidden": 8192, "batch": 1}
BREAKEVEN = {name: FLOPS / (6 * link) for name, link in LINKS}


def offload_forward(ref, x, params, k):
    """The exercise's construction: segment inputs copied into a separate list."""
    buffer, h = [np.array(x, copy=True)], x
    for i, layer in enumerate(params):
        h = ref.layer_forward(h, *layer)
        if (i + 1) % k == 0 and (i + 1) < len(params):
            buffer.append(np.array(h, copy=True))
    buffer.append(np.array(h, copy=True))
    return h, buffer


def span(fn, repeats):
    runs = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        runs.append(time.perf_counter() - start)
    return min(runs), max(runs)


def rate(shape, repeats):
    """bytes/time for one array of `shape`, timed on its own."""
    array = np.zeros(shape, dtype=np.float32)
    return array.nbytes / span(lambda: np.array(array, copy=True), repeats)[0]


def bandwidth(buffer):
    """bytes/time for the saved tensors, for one array in cache and one far too big."""
    moved = sum(a.nbytes for a in buffer)
    segment, _ = span(lambda: [np.array(a, copy=True) for a in buffer], 15)
    return {"moved": moved, "segment_time": segment, "segment": moved / segment,
            "cached": rate(CACHED, 200), "stream": rate(STREAM, 10)}


def crossing(ref):
    """Recompute against copy, per layer, across hidden sizes."""
    rows = {}
    for width in WIDTHS:
        params = ref.make_params(1, width, 2 * width)
        x = np.random.default_rng(0).standard_normal((BATCH, width)).astype(np.float32)
        rows[width] = (span(lambda: ref.layer_forward(x, *params[0]), 9)[0]
                       / span(lambda: (np.array(x, copy=True),
                                       np.array(x, copy=True)), 9)[0])
    return rows


def defaults():
    """The lesson's own configuration, priced against an A100 and a PCIe gen4 link."""
    seq, hidden = DEFAULTS["seq"], DEFAULTS["hidden"]
    moved = DEFAULTS["batch"] * seq * hidden * BYTES
    return {"recompute": 24 * DEFAULTS["batch"] * seq * hidden ** 2 / FLOPS,
            "moved": moved / 1e6, "offload": 2 * moved / 25e9}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    params = ref.make_params(LAYERS, HIDDEN, INNER)
    x = np.random.default_rng(0).standard_normal((BATCH, HIDDEN)).astype(np.float32)
    plain, slowest = span(lambda: ref.model_forward_checkpointed(x, params, k=SEGMENT), 9)
    with_copies, _ = span(lambda: offload_forward(ref, x, params, SEGMENT), 9)
    measured, buffer = offload_forward(ref, x, params, SEGMENT)
    reference, bytes_moved = ref.model_forward_checkpointed(
        x, params, k=SEGMENT)[0], bandwidth(buffer)
    ratios, wide = crossing(ref), [w for w in WIDTHS if w >= 128]  # below: pure overhead
    return {
        "same_output": bool(np.array_equal(reference, measured)), "saved": len(buffer),
        "bandwidth": bytes_moved, "ratios": ratios, "wide": wide, "defaults": defaults(),
        "breakeven": BREAKEVEN, "typical": statistics.median(ratios[w] for w in wide),
        "share": bytes_moved["segment_time"] / plain, "jitter": (slowest - plain) / plain,
        "over_plain": with_copies / plain - 1, "per_byte": round(GFLOP / COPIED),
    }


def verify(result):
    band, ratios, even = result["bandwidth"], result["ratios"], result["breakeven"]
    spread = max(band["cached"], band["stream"]) / min(band["cached"], band["stream"])
    plain, wide = result["defaults"], result["wide"]
    return [
        practice.Check(
            "ANSWER: no breakeven in the toy -- copying beats recomputing wherever math dominates",
            result["same_output"] and result["typical"] > 5,
            f"the offload saves {result['saved']} inputs, output matches. Recompute over copy: "
            + ", ".join(f"h={w} {r:.0f}x" for w, r in ratios.items())
            + f", median {result['typical']:.0f}x over {wide}. With inner = 2h the FLOPs per "
            "byte copied are exactly h, so the ratio is linear in width by construction -- "
            "4x from h=128 to h=512. A median, because one contended width sinks a min",
        ),
        practice.Check(
            "FINDING: bytes/time answers with the array size and the machine, not with a link",
            spread > 1.5 and all(1e7 < v < 1e13 for v in (band["cached"], band["stream"])),
            f"the same memcpy answers {band['cached'] / 1e9:.1f} GB/s on a 0.26 MB array and "
            f"{band['stream'] / 1e9:.1f} GB/s on a 67 MB one -- {spread:.1f}x apart for no "
            f"reason but what fits in cache, and which is faster is a property of the host. "
            f"The {band['moved'] / 1e6:.1f} MB of segment inputs land at "
            f"{band['segment'] / 1e9:.1f} GB/s here; PCIe gen4 x16 is a fixed 25 GB/s",
        ),
        practice.Check(
            "FINDING: 3,511 FLOPs are spent per byte copied, so no clock will see them",
            (SAVED, COPIED, GFLOP, result["per_byte"], result["share"] < 0.05)
            == (7, 3_670_016, 12_884_901_888, 3511, True),
            f"the configuration settles this without a stopwatch: {SAVED} saved tensors "
            f"are {COPIED / 1e6:.2f} MB against {GFLOP / 1e9:.2f} GFLOP, "
            f"{result['per_byte']:,} FLOPs per byte copied. Timed in isolation the copies "
            f"are {result['share']:.2%} of the step; in place the offload arm's fastest "
            f"run is {result['over_plain']:+.1%} against a plain arm carrying "
            f"{result['jitter']:.1%} jitter, so that sign is a draw, not a measurement",
        ),
        practice.Check(
            "MECHANISM: the breakeven is h = F / (6 * BW), and it moves 36x with the link",
            (max(even.values()) / min(even.values()) > 30
             and plain["recompute"] > 3 * plain["offload"]),
            "recompute costs 24*b*s*h^2 / F and a round trip 2*b*s*h*2 / BW, so they cross at "
            + ", ".join(f"{name} {value:,.0f}" for name, value in even.items())
            + f" -- a {max(even.values()) / min(even.values()):.0f}x range set entirely by the "
            f"one number a list in the same address space does not have. At the lesson's own "
            f"defaults a layer's recompute is {1e3 * plain['recompute']:.1f} ms against a "
            f"{1e3 * plain['offload']:.1f} ms round trip for {plain['moved']:.1f} MB -- offload "
            f"wins {plain['recompute'] / plain['offload']:.1f}x, and loses below hidden "
            f"{even['PCIe gen4 x16']:,.0f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
