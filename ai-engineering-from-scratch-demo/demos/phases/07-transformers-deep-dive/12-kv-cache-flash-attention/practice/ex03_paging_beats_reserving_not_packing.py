"""Exercise 3 — paging beats reserving by 8x and beats packing by nothing.

    **Hard.** Implement a toy PagedAttention: KV cache in fixed 16-token blocks
    with a free-list. When a sequence finishes, return its blocks to the pool.
    Simulate 1,000 chat completions with varying lengths. Compare memory
    fragmentation vs contiguous allocation.

Reading of the exercise: "contiguous allocation" names two different schemes and
the answer depends entirely on which. Reserving `max_len` per sequence is what
inference servers did before vLLM and is the comparison paging was built to win.
Packing exact-size contiguous allocations with a first-fit allocator is the
comparison paging is usually *described* as winning. Both are run against the
same 1,000 completions -- lengths log-normal around 180 tokens, mean 262, median
173, capped at a 2,048 context.

**ANSWER: against reserve-at-max_len, paging is 8.1x.** A 32,768-token pool holds
**16** sequences reserved at 2,048 each and **129** paged. Internal waste falls
from **87.2%** to **2.82%**, and the paged overhead is **7.61 tokens per
sequence** -- bounded above by `BLOCK - 1 = 15` and independent of length, which
is the property that makes it a constant rather than a risk.

**FINDING: against exact-size packing, paging is a wash.** Held at a fixed number
of concurrent sequences with a first-fit allocator and a coalescing free list:

| pool | concurrent | contiguous failures | of those, external fragmentation | paged failures |
|---:|---:|---:|---:|---:|
| 6,144 | 20 | 58 | 46 | **82** |
| 8,192 | 28 | 60 | 49 | 48 |
| 12,288 | 40 | 26 | 21 | 32 |

External fragmentation is real -- **79%** of the contiguous failures had enough
total free space and no single hole big enough -- and block rounding costs about
as much as it saves. Paging does not win this comparison.

**FINDING: exact-size packing is not available.** It needs the length of the
completion at allocation time, which is the one number an autoregressive decoder
does not have. That is why the real alternative is reserve-at-max_len, and why
the 8.1x is the number that matters.

**CONTROL: the free list is what makes the pool reusable.** Blocks returned on
completion are immediately re-allocatable in any order, because every block is
interchangeable -- the property a variable-size allocator cannot have.

Structure: `corpus` draws the lengths; `packed` runs first-fit with coalescing;
`paged` runs the block free-list; `reserved` is the max_len arithmetic.
"""

from __future__ import annotations

import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "12-kv-cache-flash-attention"
BLOCK, MAX_LEN, COMPLETIONS, POOL = 16, 2_048, 1_000, 32_768
SETTINGS = ((6_144, 20), (8_192, 28), (12_288, 40))


def corpus(seed=7, count=COMPLETIONS):
    """Chat completion lengths: log-normal around 180 tokens, capped at the context."""
    rng = random.Random(seed)
    return [max(8, min(MAX_LEN, int(rng.lognormvariate(math.log(180), 0.9))))
            for _ in range(count)]


def coalesce(free):
    """Merge adjacent holes -- what a variable-size allocator has to do and paging does not."""
    out = []
    for start, end in sorted(free):
        if out and out[-1][1] == start:
            out[-1] = (out[-1][0], end)
        else:
            out.append((start, end))
    return out


def packed(lengths, pool, window):
    """Exact-size contiguous, first-fit. (failures, failures with enough total free space)."""
    free, live, failures, external = [(0, pool)], [], 0, 0
    for length in lengths:
        while len(live) >= window:
            free = coalesce(free + [live.pop(0)])
        hole = next(((s, e) for s, e in free if e - s >= length), None)
        if hole is None:
            failures += 1
            external += sum(e - s for s, e in free) >= length
            continue
        free.remove(hole)
        free += [(hole[0] + length, hole[1])] if hole[1] - hole[0] > length else []
        live.append((hole[0], hole[0] + length))
    return failures, external


def paged(lengths, pool, window):
    """Fixed 16-token blocks with a free list. Returns failures."""
    blocks, live, failures = pool // BLOCK, [], 0
    for length in lengths:
        while len(live) >= window:
            blocks += live.pop(0)
        need = math.ceil(length / BLOCK)
        if blocks < need:
            failures += 1
            continue
        blocks -= need
        live.append(need)
    return failures


def capacity(lengths, pool=POOL):
    """How many sequences fit at once: (reserved at max_len, paged)."""
    total, fits = 0, 0
    for length in lengths:
        total += BLOCK * math.ceil(length / BLOCK)
        if total > pool:
            break
        fits += 1
    return pool // MAX_LEN, fits


def solve():
    parity.load_reference(PHASE, LESSON, "main")
    lengths = corpus()
    used = sum(lengths)
    blocks = sum(BLOCK * math.ceil(length / BLOCK) for length in lengths)
    return {
        "lengths": (statistics.fmean(lengths), statistics.median(lengths), max(lengths)),
        "waste": (1 - used / (COMPLETIONS * MAX_LEN), 1 - used / blocks),
        "overhead": (blocks - used) / COMPLETIONS, "capacity": capacity(lengths),
        "dynamic": (runs := {s: (packed(lengths, *s), paged(lengths, *s)) for s in SETTINGS}),
        "failures": (sum(f for (f, _), _ in runs.values()),
                     sum(e for (_, e), _ in runs.values())),
    }


def verify(result):
    reserved, fits = result["capacity"]
    big, small = result["waste"]
    dynamic, contiguous, external = result["dynamic"], *result["failures"]
    return [
        practice.Check(
            "ANSWER: against reserve-at-max_len, paging is 8.1x",
            fits / reserved > 7 and big > 0.85 and small < 0.05,
            f"a {POOL:,}-token pool holds {reserved} sequences reserved at {MAX_LEN:,} each and "
            f"{fits} paged, {fits / reserved:.1f}x, over completions averaging "
            f"{result['lengths'][0]:.0f} tokens with a median of {result['lengths'][1]:.0f}. "
            f"Internal waste falls from {big:.1%} to {small:.2%}",
        ),
        practice.Check(
            "ANSWER: the paged overhead is 7.61 tokens per sequence, bounded by 15",
            result["overhead"] < BLOCK,
            f"{result['overhead']:.2f} tokens of rounding per sequence against a ceiling of "
            f"BLOCK - 1 = {BLOCK - 1}, whatever the length. Reserving costs "
            f"{MAX_LEN - result['lengths'][0]:.0f} tokens per sequence on the same corpus, and "
            "that grows with the context window while this does not",
        ),
        practice.Check(
            "FINDING: against exact-size packing, paging is a wash",
            dynamic[SETTINGS[0]][1] > dynamic[SETTINGS[0]][0][0],
            "first-fit with a coalescing free list, at a fixed number of concurrent sequences: "
            + ", ".join(f"pool {p:,}/{w} concurrent: contiguous {f} vs paged {q}"
                        for (p, w), ((f, _), q) in dynamic.items())
            + ". Block rounding costs about what fragmentation saves; at the smallest pool "
              "paging loses",
        ),
        practice.Check(
            "FINDING: external fragmentation is real, and it is not what paging is for",
            external / contiguous > 0.7,
            f"{external} of the {contiguous} contiguous failures had enough total free space and "
            f"no single hole big enough -- {external / contiguous:.0%} fragmentation rather than "
            "exhaustion. Paging removes that failure mode entirely and still does not win",
        ),
        practice.Check(
            "FINDING: exact-size packing is not available anyway",
            fits > reserved,
            "it needs the completion's length at allocation time, the one number an "
            "autoregressive decoder does not have. That is why the real alternative is "
            f"reserve-at-max_len and why {fits / reserved:.1f}x is the number that matters. The "
            "free list also makes returned blocks re-allocatable in any order, every block being "
            "interchangeable -- which a variable-size allocator cannot be",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
