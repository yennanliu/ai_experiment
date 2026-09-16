"""Exercise 4 — offloading wins by 15x to 120x at every size in the toy; the breakeven is hardware.

    Add offload. Save segment inputs to a simulated "CPU buffer" (a separate
    list). Measure "PCIe bandwidth" as bytes/time and find the breakeven point
    between offload and recompute.

Reading of the exercise: the offload is built as described -- the checkpointed
forward copies each segment input into a separate list with `np.array(copy=True)`
-- and bytes/time is measured both at the size the saved tensors actually are and
on one large enough to leave cache. The recompute side is timed against it across
hidden sizes, and the breakeven is then written down as the closed form, because
the simulated buffer has no parameter in which an interconnect could appear.

**ANSWER: there is no breakeven inside the toy.** Copying a layer's input is
15x to 120x cheaper than recomputing that layer, at every hidden size from 8 to
512, and the ratio *grows* with width: recompute is `O(h*inner)` and the copy is
`O(h)`. The sweep runs away from the crossing rather than toward it.

**FINDING: bytes/time measures this machine's cache hierarchy, and the number
changes about 4x with the array size.** The 6.8 MB of segment inputs copy at
about **75 GB/s**; a single 67 MB array copies at about **20 GB/s**. PCIe gen4 x16 is
25 GB/s, so the simulation straddles the quantity it claims to measure and lands
on either side of it depending on what fits in L3. A list in the same address
space cannot be slower than memcpy, and a link is the only thing that makes
offload a decision.

**FINDING: the traffic is unmeasurable inside the step anyway.** The copies are
**0.23%** of the checkpointed forward's time, against **8.3%** run-to-run
jitter on the forward itself -- a factor of 36. Timing them in place, as "add
offload, then measure", returns noise; they have to be timed in isolation, which
is how the number above was obtained.

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

import math
import time

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "34-gradient-checkpointing"
LAYERS, HIDDEN, INNER, BATCH, SEGMENT = 24, 512, 1024, 256, 4
WIDTHS = (8, 16, 32, 64, 128, 256, 512)
BIG = (4096, 4096)
LINKS = (("PCIe gen4 x16", 25e9), ("PCIe gen5 x16", 64e9),
         ("NVLink 3", 300e9), ("NVLink 4", 900e9))
FLOPS, BYTES = 312e12, 2
DEFAULTS = {"layers": 64, "seq": 8192, "hidden": 8192, "batch": 1}


def offload_forward(ref, x, params, k):
    """The exercise's construction: segment inputs copied into a separate list."""
    buffer, h = [np.array(x, copy=True)], x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = ref.layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0 and (i + 1) < len(params):
            buffer.append(np.array(h, copy=True))
    buffer.append(np.array(h, copy=True))
    return h, buffer


def span(fn, repeats):
    fastest, slowest = math.inf, 0.0
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        fastest, slowest = min(fastest, elapsed), max(slowest, elapsed)
    return fastest, slowest


def bandwidth(buffer):
    """bytes/time for the saved tensors, and for one array too big for cache."""
    moved = sum(a.nbytes for a in buffer)
    small, _ = span(lambda: [np.array(a, copy=True) for a in buffer], 15)
    big = np.zeros(BIG, dtype=np.float32)
    large, _ = span(lambda: np.array(big, copy=True), 10)
    return {"moved": moved, "small_time": small, "small": moved / small,
            "large": big.nbytes / large, "large_mb": big.nbytes / 1e6}


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
    batch, seq, hidden = DEFAULTS["batch"], DEFAULTS["seq"], DEFAULTS["hidden"]
    moved = batch * seq * hidden * BYTES
    return {"recompute": 24 * batch * seq * hidden ** 2 / FLOPS,
            "offload": 2 * moved / 25e9, "moved": moved / 1e6}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    params = ref.make_params(LAYERS, HIDDEN, INNER)
    x = np.random.default_rng(0).standard_normal((BATCH, HIDDEN)).astype(np.float32)
    plain, jitter = span(lambda: ref.model_forward_checkpointed(x, params, k=SEGMENT), 9)
    _, buffer = offload_forward(ref, x, params, SEGMENT)
    reference, _ = ref.model_forward_checkpointed(x, params, k=SEGMENT)
    measured, _ = offload_forward(ref, x, params, SEGMENT)
    bytes_moved = bandwidth(buffer)
    return {
        "same_output": bool(np.array_equal(reference, measured)),
        "saved": len(buffer),
        "bandwidth": bytes_moved,
        "share": bytes_moved["small_time"] / plain,
        "jitter": (jitter - plain) / plain,
        "ratios": crossing(ref),
        "breakeven": breakeven(),
        "defaults": defaults(),
    }


def verify(result):
    band, ratios, even = result["bandwidth"], result["ratios"], result["breakeven"]
    plain = result["defaults"]
    return [
        practice.Check(
            "ANSWER: there is no breakeven in the toy -- offload wins 15x to 120x at every width",
            result["same_output"] and min(ratios.values()) > 5
            and ratios[max(WIDTHS)] > ratios[min(WIDTHS)],
            f"the offload forward saves {result['saved']} segment inputs and returns the same "
            f"output as the lesson's own ({result['same_output']}). Recompute over copy is "
            + ", ".join(f"h={width} {ratio:.0f}x" for width, ratio in ratios.items())
            + ". The ratio grows with width because recompute is O(h*inner) and the copy is "
            "O(h), so the sweep runs away from the crossing rather than toward it",
        ),
        practice.Check(
            "FINDING: bytes/time measures this machine's cache, and moves about 4x with the size",
            band["small"] > band["large"] and 5e9 < band["large"] < 2e11,
            f"the {band['moved'] / 1e6:.1f} MB of segment inputs copy at "
            f"{band['small'] / 1e9:.1f} GB/s; a single {band['large_mb']:.0f} MB array copies at "
            f"{band['large'] / 1e9:.1f} GB/s, a factor of {band['small'] / band['large']:.1f}. "
            "PCIe gen4 x16 is 25 GB/s, so the simulation straddles the quantity it claims to "
            "measure and lands on either side of it depending on what fits in L3",
        ),
        practice.Check(
            "FINDING: the traffic is unmeasurable in place -- 0.2% against 8% jitter",
            result["jitter"] > 5 * result["share"],
            f"the copies are {result['share']:.2%} of the checkpointed forward's time against "
            f"{result['jitter']:.1%} run-to-run jitter on the forward itself, a factor of "
            f"{result['jitter'] / result['share']:.0f}. 'Add offload, then measure bytes/time' "
            "returns noise if the copies are timed in place; the figure above comes from timing "
            "them in isolation",
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
            f"{1e3 * plain['offload']:.1f} ms round trip for {plain['moved']:.1f} MB, so offload "
            f"wins {plain['recompute'] / plain['offload']:.1f}x on PCIe and would lose below "
            f"hidden {even['PCIe gen4 x16']:,.0f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
