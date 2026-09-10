"""Exercise 1 — the requested grid cannot vary.

    **Easy.** Build a NIAH with 3 depths (0.25, 0.5, 0.75) × 3 lengths (1k, 4k,
    16k). Run on any model. Plot pass rate as a 3×3 heatmap.

Reading of the exercise: build exactly the grid asked for, on the only model the
lesson ships -- `mock_retrieval_model`, whose effective capacity `run_niah_grid`
hard-codes at 20,000 words -- and ask what the heatmap would show. All **nine
cells score 1**. The pass rate is 1.0 in every cell, the variance across the
grid is zero, and there is no surface to plot.

The mock retrieves the needle exactly when `int(length * depth) <= 20000`. The
deepest cell the exercise asks for puts the needle at word **12,000** of 16,000
-- 8,000 words inside the cutoff -- so no cell in the grid is capable of failing.
Binary-searching the boundary per depth, the shortest failing context is
**80,004** words at depth 0.25, **40,002** at depth 0.5 and **26,668** at depth
0.75. The exercise's longest context is 16k, so its most demanding corner is
still 1.7x short of the first length that could produce a zero.

"Run on any model" has no second reading available here: `openai`, `anthropic`,
`transformers`, `torch` and `tiktoken` are all absent, and so is `matplotlib`
-- **6 of 6 probed modules missing** -- so both the model and the heatmap resolve
to what the lesson ships, a mock and a printed text grid.

Widening the length axis does produce variation, but never the shape the lesson's
own Pitfalls section warns about. Sweeping depth 0 to 1.0 gives **[1, 1, 1, 1,
1]** at 20k, **[1, 1, 1, 0, 0]** at 30k and **[1, 1, 0, 0, 0]** at 60k: a
staircase that only ever falls. "Lost in the middle" is a U -- middle worse than
both ends -- and a hard positional cutoff cannot express one at any length.

Which is why a predictor that never reads the haystack reproduces the model
exactly. `int(length * depth) <= 20000` agrees with the mock on **50 of 50**
cells over ten lengths and five depths. The mock also ignores its `question`
argument outright: asking "What is the capital of France?" of a pineapple
haystack returns `pineapple`. And `run_niah_grid` prints its rows and returns
`None`, so the pass rates the exercise asks you to plot are not values the
shipped function will hand back.

Structure: `cell` scores one depth-by-length point through the lesson's own
`score_single_needle`; `grid` tiles it; `flip_length` binary-searches the first
failing length for a depth; `depth_rows` sweeps five depths at fixed lengths;
`agreement` compares the mock against `position_only`, the baseline that reads
nothing; `absent` probes for the named libraries without importing them.
"""

from __future__ import annotations

import importlib.util

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "28-long-context-evaluation"

CAPACITY = 20000
NEEDLE, EXPECTED = "the magic word is pineapple", "pineapple"
SPEC_DEPTHS = (0.25, 0.5, 0.75)
SPEC_LENGTHS = (1000, 4000, 16000)
DEPTH_SWEEP = (0.0, 0.25, 0.5, 0.75, 1.0)
STAIR_LENGTHS = (20000, 30000, 60000)
WIDE_LENGTHS = (500, 1000, 4000, 16000, 26000, 27000, 40000, 64000, 80000, 100000)
PROBED = ("openai", "anthropic", "transformers", "torch", "tiktoken", "matplotlib")


def absent(names):
    """The probed modules that are not installed — probed by spec, never imported."""
    return tuple(n for n in names if importlib.util.find_spec(n) is None)


def cell(ref, length, depth):
    """One NIAH cell: 1 when the mock retrieves the needle at that depth and length."""
    haystack = ref.insert_needle(ref.make_filler(length, seed=length), NEEDLE, depth)
    return ref.score_single_needle(haystack, EXPECTED, CAPACITY)


def grid(ref, lengths, depths):
    """The depth-by-length pass matrix, as {(depth, length): 0 or 1}."""
    return {(d, n): cell(ref, n, d) for d in depths for n in lengths}


def flip_length(ref, depth, lo=1000, hi=400000):
    """The shortest context at which the needle at that depth falls out of reach."""
    while lo < hi:
        mid = (lo + hi) // 2
        if cell(ref, mid, depth):
            lo = mid + 1
        else:
            hi = mid
    return lo


def depth_rows(ref, lengths):
    """Pass value at every depth in DEPTH_SWEEP, one row per length."""
    return {n: [cell(ref, n, d) for d in DEPTH_SWEEP] for n in lengths}


def position_only(length, depth):
    """The baseline that never reads the haystack: pass iff the offset fits the capacity."""
    return 1 if int(length * depth) <= CAPACITY else 0


def falls_only(row):
    """True when accuracy never recovers as the needle moves deeper — no U, no middle dip."""
    return all(a >= b for a, b in zip(row, row[1:]))


def agreement(ref):
    """Cells where position_only matches the mock, and how many cells were compared."""
    cells = [(n, d, cell(ref, n, d)) for n in WIDE_LENGTHS for d in DEPTH_SWEEP]
    return sum(1 for n, d, got in cells if got == position_only(n, d)), len(cells)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    haystack = ref.insert_needle(ref.make_filler(1000, seed=1), NEEDLE, 0.5)
    with parity.quiet():
        returned = ref.run_niah_grid(SPEC_LENGTHS, SPEC_DEPTHS)
    agree, n_cells = agreement(ref)
    return {
        "spec": grid(ref, SPEC_LENGTHS, SPEC_DEPTHS),
        "deepest": max(int(n * d) for d in SPEC_DEPTHS for n in SPEC_LENGTHS),
        "flips": {d: flip_length(ref, d) for d in SPEC_DEPTHS},
        "rows": depth_rows(ref, STAIR_LENGTHS),
        "agree": agree,
        "n_cells": n_cells,
        "absent": absent(PROBED),
        "n_probed": len(PROBED),
        "asked": ref.mock_retrieval_model(haystack, "What is the magic word?", CAPACITY),
        "unasked": ref.mock_retrieval_model(haystack, "What is the capital of France?", CAPACITY),
        "returned": returned,
    }


def verify(result):
    spec, flips, rows = result["spec"], result["flips"], result["rows"]
    return [
        practice.Check(
            "ANSWER: every one of the nine requested cells passes, so the heatmap is flat",
            sum(spec.values()) == len(spec),
            f"the grid the exercise specifies scores {sum(spec.values())}/{len(spec)} on the "
            f"lesson's mock at its own capacity of {CAPACITY} words: pass rate 1.0 in every "
            "cell, zero variance across depth and across length, nothing to plot",
        ),
        practice.Check(
            "MECHANISM: the deepest cell sits 8,000 words inside a hard positional cutoff",
            result["deepest"] < CAPACITY and min(flips.values()) > max(SPEC_LENGTHS),
            f"the mock passes iff int(length * depth) <= {CAPACITY}; the deepest requested cell "
            f"puts the needle at word {result['deepest']}. Shortest failing length per depth: "
            f"{flips} — the exercise's longest context is {max(SPEC_LENGTHS)}",
        ),
        practice.Check(
            "FINDING: 'run on any model' and 'plot a heatmap' have no library to run on",
            len(result["absent"]) == result["n_probed"],
            f"{len(result['absent'])} of {result['n_probed']} probed modules are missing: "
            f"{list(result['absent'])}. Both the model and the plot resolve to what the lesson "
            "ships — a regex mock and the text grid run_niah_grid prints",
        ),
        practice.Check(
            "FINDING: the depth axis only ever falls, so 'lost in the middle' is unreachable",
            all(falls_only(row) for row in rows.values()),
            f"sweeping depth {list(DEPTH_SWEEP)}: {rows[20000]} at 20k, {rows[30000]} at 30k, "
            f"{rows[60000]} at 60k. A staircase, never a U. The Pitfalls section asks for the "
            "depth-bias effect; a hard positional cutoff cannot express one at any length",
        ),
        practice.Check(
            "CONTROL: a predictor that never reads the haystack reproduces the model exactly",
            result["agree"] == result["n_cells"],
            f"int(length * depth) <= {CAPACITY} agrees with the mock on {result['agree']}/"
            f"{result['n_cells']} cells over ten lengths and five depths. The grid measures the "
            "fixture's arithmetic, not retrieval",
        ),
        practice.Check(
            "CONTROL: the question is ignored, and run_niah_grid returns nothing to plot",
            result["asked"] == result["unasked"] and result["returned"] is None,
            f"asking 'What is the capital of France?' of a pineapple haystack returns "
            f"'{result['unasked']}', the same answer as the real question. And run_niah_grid "
            f"printed its rows and returned {result['returned']}: the pass rates are not values "
            "the shipped function hands back",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
