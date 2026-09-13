"""Exercise 1 — there is no op-count difference to note.

    **Easy.** Run `code/main.py`. Confirm the naive and cached decoders produce
    the same output; note the op-count difference.

Reading of the exercise: both halves are checked. The outputs are compared
elementwise rather than to a tolerance, and the two op counters are compared
against each other rather than against their labels.

**ANSWER: the outputs are identical to 0.0, and so are the op counts.**
`decode_naive` adds `t + 1` per step; `decode_cached` adds `len(cache)`, which is
`t + 1`. Both come to **55** at N=10, `N(N+1)/2` exactly, and `main()` prints
them on two lines labelled `O(N^2)` and `O(N)`. There is nothing to note.

**FINDING: the saving is real, and it is in a quantity neither counter counts.**
Recomputing the prefix means recomputing its K and V projections: `N(N+1)/2` of
them against the cache's `N`, a factor of `(N+1)/2`.

| N | attention ops, both | K,V projections naive | cached | saving |
|---:|---:|---:|---:|---:|
| 10 | 55 | 55 | 10 | 5.5x |
| 100 | 5,050 | 5,050 | 100 | 50.5x |
| 1,000 | 500,500 | 500,500 | 1,000 | **500.5x** |

The lesson's own comment says as much two lines further down -- "counting K,V
recomputes would make naive O(N^2) in matmuls" -- and then prints the counter
that does not.

**FINDING: the tiled softmax is not bit-identical, it is one ULP.** The module
docstring promises "a running-max softmax that yields bit-identical output
tile-by-tile". Measured against the standard path it differs by **one to two
units in the last place** at every tile size from 1 to 16. Even a single tile
covering the whole sequence disagrees, because the running form divides by a
differently-accumulated denominator.

The absolute figures move between platforms -- `math.exp` is not bit-identical
across libms, so the same code lands on 1.1e-16 where it lands on 2.2e-16
elsewhere -- which is why everything here is stated and graded in ULP.

**CONTROL: the disagreement does not grow with the number of tiles.** Tile 2
splits the sequence five ways and tile 16 not at all, and the two land within
one ULP of each other, with no tiling worse than two. The error is in the
formulation, not in the accumulation.

Structure: `projections` is the count the lesson does not keep; `tiles` sweeps
the tile size against the standard path.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "12-kv-cache-flash-attention"
WIDTH, TOKENS, SIZES, TILES = 8, 10, (10, 100, 1_000), (1, 2, 4, 7, 10, 16)
ULP = math.ulp(1.0)          # the scale every disagreement below is quoted in


def stream(rng, count=TOKENS, width=WIDTH):
    """One list of `count` random vectors."""
    return [[rng.gauss(0, 1) for _ in range(width)] for _ in range(count)]


def projections(n):
    """(naive, cached) K and V projection calls for an n-token decode."""
    return n * (n + 1) // 2, n


def tiles(ref, query, keys, values):
    """{tile size: worst disagreement with the standard softmax path}."""
    standard = ref.attention_full(query, keys, values)
    return {size: max(abs(a - b) for a, b in
                      zip(standard, ref.tiled_softmax_dot(query, keys, values, size)))
            for size in TILES}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = random.Random(42)
    queries, keys, values = stream(rng), stream(rng), stream(rng)
    naive, naive_ops = ref.decode_naive(keys, values, queries)
    cached, cached_ops = ref.decode_cached(keys, values, queries)
    return {
        "ops": (naive_ops, cached_ops), "formula": TOKENS * (TOKENS + 1) // 2,
        "gap": max(abs(a - b) for u, v in zip(naive, cached) for a, b in zip(u, v)),
        "projections": {n: projections(n) for n in SIZES},
        "tiles": tiles(ref, queries[-1], keys, values),
    }


def verify(result):
    naive_ops, cached_ops = result["ops"]
    counts, gaps = result["projections"], result["tiles"]
    return [
        practice.Check(
            "ANSWER: the outputs are identical to 0.0 and so are the op counts",
            result["gap"] == 0.0 and naive_ops == cached_ops == result["formula"],
            f"decode_naive adds t + 1 per step and decode_cached adds len(cache), which is t + 1: "
            f"both reach {naive_ops} at N={TOKENS}, exactly N(N+1)/2. main() prints them on two "
            f"lines labelled O(N^2) and O(N). The outputs differ by {result['gap']}",
        ),
        practice.Check(
            "FINDING: the saving is in a quantity neither counter counts",
            all(naive == cached * (n + 1) // 2 for n, (naive, cached) in counts.items()),
            "recomputing the prefix recomputes its K and V projections: "
            + ", ".join(f"N={n} naive {a:,} vs cached {b:,} ({a / b:.1f}x)"
                        for n, (a, b) in counts.items())
            + ". The lesson's own comment says so two lines further down and then prints the "
              "counter that does not",
        ),
        practice.Check(
            "FINDING: the tiled softmax is one ULP off, not bit-identical",
            0 < gaps[4] < 1e-15 and gaps[TILES[-1]] > 0,
            f"the module docstring promises output that is bit-identical tile-by-tile. Measured: "
            + ", ".join(f"tile {t} {gaps[t] / ULP:.1f} ULP" for t in TILES)
            + f". Even tile {TILES[-1]}, one block covering the whole sequence, disagrees -- the "
              "running form divides by a differently-accumulated denominator",
        ),
        practice.Check(
            "CONTROL: the disagreement does not grow with the number of tiles",
            max(gaps.values()) <= 2 * ULP and gaps[2] <= gaps[TILES[-1]] + ULP,
            f"tile 2 splits {TOKENS} keys five ways and tile {TILES[-1]} not at all: "
            f"{gaps[2] / ULP:.1f} and {gaps[TILES[-1]] / ULP:.1f} units in the last place, within "
            f"one ULP of each other, and no tiling exceeds {max(gaps.values()) / ULP:.1f}. The "
            "error is in the formulation rather than in the accumulation -- an error that grew "
            "with the block count would be the accumulation",
        ),
        practice.Check(
            "CONTROL: 'same output' is a stronger claim here than it usually is",
            result["gap"] == 0.0,
            "decode_cached passes the same key and value objects to the same attention_full in "
            "the same order, so the two paths are the same arithmetic and the difference is "
            "exactly zero rather than small. The tiled path, which is genuinely different "
            "arithmetic, is not",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
