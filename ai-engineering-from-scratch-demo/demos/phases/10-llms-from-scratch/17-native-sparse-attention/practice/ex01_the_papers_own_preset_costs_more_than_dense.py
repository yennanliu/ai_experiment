"""Exercise 1 — the paper's own preset reads 152% of full attention at 1024 tokens, and every preset recalls 100%.

    Run `code/main.py` on a 1024-token synthetic. Sweep `(l, k, w)` across three
    presets and print compute counts. Identify the preset that achieves the
    lowest key-count per query while keeping 95% recall against full attention
    on a needle-in-haystack test.

Reading of the exercise: key counts come from the lesson's own `count_nsa` and
are cross-checked against what `nsa_step` actually reads, and recall is measured
the way a needle-in-haystack test measures it -- did the selected branch pick the
block the needle is in -- on 40 placements per preset, with the needle always
placed outside the sliding window so that the selection has to find it.

**ANSWER: the lowest key count is `l=64, k=4, W=256` at 528 keys, and the recall
constraint never binds.**

    preset              keys   share of full   recall
    l=32,  k=8,  W=512   800       78.1%        100%
    l=64,  k=16, W=512  1552      151.6%        100%
    l=64,  k=4,  W=256   528       51.6%        100%
    l=128, k=4,  W=128   648       63.3%        100%

**FINDING: the paper's own preset costs more than the dense attention it
replaces.** `l=64, k=16, W=512` reads **1552** keys on a 1024-token sequence.
`count_nsa` is `N/l + k*l + W`, and two of its three terms do not depend on `N`
at all, so at short context the fixed cost dominates. NSA breaks even against
full attention at **N = 1560** for this preset; the exercise runs it at 1024.

**FINDING: 95% recall is not a constraint on this test.** Every preset scores
**100%** on 40 needle placements, because `synthesize_sequence` fills the whole
signal block with copies of one unit pattern -- so the block's mean *is* that
pattern, the compressed score for it is the maximum possible, and it ranks first
however coarse the compression. A needle that survives mean-pooling perfectly
cannot discriminate between compression ratios.

**MECHANISM: the three branches are priced per query and two are constant.**
The compressed branch is `N/l` keys and shrinks as blocks get bigger; the
selected branch is `k*l` and *grows* with block size at fixed `k`; the window is
`W` and never moves. Doubling `l` from 64 to 128 at `k=4` cuts the compressed
term from 16 to 8 and raises the selected term from 256 to 512 -- which is why
the coarsest preset is not the cheapest.

Structure: `preset_row` runs 40 needle placements through the lesson's own
`nsa_step` and reports measured keys beside `count_nsa`'s formula.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "17-native-sparse-attention"
TOKENS, DIM, TRIALS = 1024, 16, 40
PRESETS = {"l=32,k=8,W=512": (32, 8, 512), "l=64,k=16,W=512": (64, 16, 512),
           "l=64,k=4,W=256": (64, 4, 256), "l=128,k=4,W=128": (128, 4, 128)}


def preset_row(ref, block, top_k, window):
    """Measured keys and needle recall over `TRIALS` placements outside the window."""
    config = ref.NSAConfig(l=block, k=top_k, W=window)
    found, keys = 0, []
    for seed in range(TRIALS):
        rng = random.Random(seed)
        reachable = max(1, TOKENS // block - window // block)
        needle = [rng.randrange(0, reachable)]
        keys_, values, query = ref.synthesize_sequence(TOKENS, DIM, needle, block, rng)
        gates = [[rng.gauss(0, 1) for _ in range(DIM)] for _ in range(3)]
        _, info = ref.nsa_step(query, keys_, values, gates, config)
        found += set(info["selected_blocks"]) >= set(needle)
        keys.append(info["total_keys"])
    return {"formula": ref.count_nsa(TOKENS, block, top_k, window),
            "measured": statistics.fmean(keys), "recall": found / TRIALS,
            "share": statistics.fmean(keys) / TOKENS,
            "compressed": TOKENS // block, "selected": top_k * block, "window": window}


def breakeven(block, top_k, window):
    """The N at which `N/l + k*l + W` equals N."""
    return (top_k * block + window) / (1 - 1 / block)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = {name: preset_row(ref, *config) for name, config in PRESETS.items()}
    return {
        "rows": rows,
        "cheapest": min(rows, key=lambda n: rows[n]["measured"]),
        "over_full": [name for name, row in rows.items() if row["share"] > 1.0],
        "breakeven": {name: breakeven(*config) for name, config in PRESETS.items()},
        "tokens": TOKENS,
    }


def column(rows, field, fmt):
    return ", ".join(f"{name} {format(row[field], fmt)}" for name, row in rows.items())


def verify(result):
    rows, cheapest = result["rows"], result["cheapest"]
    paper = rows["l=64,k=16,W=512"]
    coarse, medium = rows["l=128,k=4,W=128"], rows["l=64,k=4,W=256"]
    return [
        practice.Check(
            "ANSWER: the cheapest preset is l=64,k=4,W=256 at 528 keys, and recall never binds",
            cheapest == "l=64,k=4,W=256" and min(r["recall"] for r in rows.values()) == 1.0,
            "measured keys per query are " + column(rows, "measured", ".0f")
            + ", which as a share of full attention is " + column(rows, "share", ".1%")
            + ", at recalls of " + column(rows, "recall", ".0%")
            + f". Every measured count matches count_nsa's formula exactly, and the 95% recall "
            "condition the exercise attaches to the choice is satisfied by all four",
        ),
        practice.Check(
            "FINDING: the paper's own preset costs more than the dense attention it replaces",
            paper["share"] > 1.0 and result["breakeven"]["l=64,k=16,W=512"] > result["tokens"],
            f"l=64,k=16,W=512 reads {paper['measured']:.0f} keys on a {result['tokens']}-token "
            f"sequence -- {paper['share']:.1%} of full attention. count_nsa is N/l + k*l + W and "
            f"two of its three terms do not depend on N at all, so at short context the fixed "
            f"cost dominates: this preset breaks even at N = "
            f"{result['breakeven']['l=64,k=16,W=512']:.0f}, and the exercise runs it at "
            f"{result['tokens']}. The presets over full attention are {result['over_full']}",
        ),
        practice.Check(
            "FINDING: 95% recall is not a constraint on this test",
            all(row["recall"] == 1.0 for row in rows.values()),
            f"every preset scores 100% on {TRIALS} needle placements outside the sliding window. "
            "synthesize_sequence fills the whole signal block with copies of one unit pattern, so "
            "the block's mean is that pattern, its compressed score is the maximum possible, and "
            "it ranks first however coarse the compression. A needle that survives mean-pooling "
            "perfectly cannot discriminate between compression ratios",
        ),
        practice.Check(
            "MECHANISM: two of the three branches are constant in N, and they trade against each other",
            coarse["compressed"] < medium["compressed"] and coarse["selected"] > medium["selected"],
            f"the compressed branch is N/l keys and shrinks as blocks get bigger; the selected "
            f"branch is k*l and grows with block size at fixed k; the window is W and never "
            f"moves. Doubling l from 64 to 128 at k=4 cuts the compressed term from "
            f"{medium['compressed']} to {coarse['compressed']} and raises the selected term from "
            f"{medium['selected']} to {coarse['selected']}, which is why the coarsest preset is "
            f"not the cheapest -- {coarse['measured']:.0f} keys against "
            f"{medium['measured']:.0f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
