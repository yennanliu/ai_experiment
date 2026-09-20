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
link.** On the same machine the figure moves about **3-4x** with the size -- the
6.8 MB of segment inputs copy at 75 GB/s and a single 67 MB array at 20 GB/s,
because one fits in L3 and the other does not. Across machines the same
measurement moves again: a CI runner reports 15 GB/s and 4.9 GB/s for the same
two copies, a fifth of this machine's. PCIe gen4 x16 is a fixed 25 GB/s, and the
simulation lands above it, below it, or nowhere near it depending on what it is
run on. A list in the same address space cannot be slower than memcpy, and a
link is the only thing that makes offload a decision.

**FINDING: the traffic is unmeasurable inside the step anyway.** Timed in
isolation the copies are well under **1%** of the checkpointed forward; timed in
place they disappear into run-to-run jitter, which runs from **2%** on a quiet
CI runner to **35%** on a busy laptop. Adding the copies leaves the fastest run
inside the plain forward's own min-to-max envelope, and on a quiet host the
offload arm comes out *faster* than the plain one -- which a real cost cannot
do. Timing them in place, as "add offload, then measure", returns noise; they
have to be timed in isolation, which is how the number above was obtained.

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
DEFAULTS = {"layers": 64, "seq": 8192, "hidden": 8192, "batch": 1}


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
    fastest, _ = span(lambda: np.array(array, copy=True), repeats)
    return array.nbytes / fastest


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
        recompute, _ = span(lambda: ref.layer_forward(x, *params[0]), 9)
        copy, _ = span(lambda: (np.array(x, copy=True), np.array(x, copy=True)), 9)
        rows[width] = recompute / copy
    return rows


def breakeven():
    """Where 24 b s h^2 / F equals 2 * b s h * bytes / BW, per link."""
    return {name: FLOPS / (6 * link) for name, link in LINKS}


def defaults():
    """The lesson's own configuration, priced against an A100 and a PCIe gen4 link."""
    batch, seq, hidden = DEFAULTS["batch"], DEFAULTS["seq"], DEFAULTS["hidden"]
    moved = batch * seq * hidden * BYTES
    return {"recompute": 24 * batch * seq * hidden ** 2 / FLOPS,
            "moved": moved / 1e6, "offload": 2 * moved / 25e9}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    params = ref.make_params(LAYERS, HIDDEN, INNER)
    x = np.random.default_rng(0).standard_normal((BATCH, HIDDEN)).astype(np.float32)
    plain, slowest = span(lambda: ref.model_forward_checkpointed(x, params, k=SEGMENT), 9)
    with_copies, _ = span(lambda: offload_forward(ref, x, params, SEGMENT), 9)
    reference = ref.model_forward_checkpointed(x, params, k=SEGMENT)[0]
    measured, buffer = offload_forward(ref, x, params, SEGMENT)
    bytes_moved = bandwidth(buffer)
    ratios, wide = crossing(ref), [w for w in WIDTHS if w >= 128]  # below: pure overhead
    return {
        "same_output": bool(np.array_equal(reference, measured)), "saved": len(buffer),
        "bandwidth": bytes_moved, "ratios": ratios, "wide": wide, "defaults": defaults(),
        "breakeven": breakeven(), "floor": min(ratios[w] for w in wide),
        "share": bytes_moved["segment_time"] / plain, "jitter": (slowest - plain) / plain,
        "inside_noise": with_copies <= slowest, "over_plain": with_copies / plain - 1,
    }


def verify(result):
    band, ratios, even = result["bandwidth"], result["ratios"], result["breakeven"]
    plain, wide = result["defaults"], result["wide"]
    return [
        practice.Check(
            "ANSWER: no breakeven in the toy -- offload wins by 10x and up where the math dominates",
            result["same_output"] and result["floor"] > 10
            and ratios[max(wide)] > ratios[min(wide)],
            f"the offload saves {result['saved']} inputs, output matches. Recompute over copy: "
            + ", ".join(f"h={w} {r:.0f}x" for w, r in ratios.items())
            + f". Over {wide} it grows -- recompute O(h*inner), copy O(h); below it both sides "
            "are per-call overhead, 4x at h=64 on a shared runner",
        ),
        practice.Check(
            "FINDING: bytes/time answers with the array size and the machine, not with a link",
            band["cached"] > 1.5 * band["stream"] and 1e8 < band["stream"] < 1e12,
            f"the same memcpy answers {band['cached'] / 1e9:.0f} GB/s on a 0.26 MB array and "
            f"{band['stream'] / 1e9:.0f} GB/s on a 67 MB one -- "
            f"{band['cached'] / band['stream']:.1f}x apart on one machine, for no reason but "
            f"what fits in cache. The exercise's own {band['moved'] / 1e6:.1f} MB of segment "
            f"inputs land at {band['segment'] / 1e9:.0f} GB/s here and 15 GB/s on a CI runner. "
            "PCIe gen4 x16 is a fixed 25 GB/s; this is above it, below it or neither, by host",
        ),
        practice.Check(
            "FINDING: the traffic is unmeasurable in place -- it hides inside the jitter",
            result["inside_noise"] and result["jitter"] > result["share"],
            f"timed in isolation the copies are {result['share']:.2%} of the forward, "
            f"against {result['jitter']:.1%} jitter on it. Timed in place they vanish: the "
            f"offload arm's fastest run is {result['over_plain']:+.1%} against the plain "
            f"arm's and lands inside its min-to-max envelope -- on a quiet host that goes "
            "negative, which a real cost cannot do",
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
