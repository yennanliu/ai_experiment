"""Exercise 3 — the one global layer in six is 96% of the mix's KV cache.

    **Hard.** Implement a Gemma-3-style 5:1 layer mix (5 SWA, 1 global) in the
    capstone model. Compare loss, memory, and generation quality against
    pure-SWA and pure-global baselines at matched parameters.

Reading of the exercise: the capstone has **3 layers**, and 5:1 needs a multiple
of 6, so the mix cannot be laid out in it at all -- 3 % 6 = 3. Its `block_size`
is also 64, where Exercise 2 shows any window at or above 64 is full attention,
so "pure SWA" and "pure global" are the same model there too. Both halves are
therefore measured where they are defined: on the lesson's own KV-cache
arithmetic at 128K context, 80 layers, 8 KV heads and `d_head=128` in fp16.

**ANSWER: the mix costs 7.44 GB against full attention's 42.9 GB -- a 5.8x
shrink, against a ceiling of exactly 6.**

| configuration | KV cache | vs full |
|---|---:|---:|
| full attention | 42.9 GB | 1.0x |
| Gemma 5:1, W=1024 | **7.44 GB** | 5.8x |
| pure SWA, W=1024 | **0.34 GB** | 128x |
| pure SWA, W=4096 | 1.34 GB | 32x |

**FINDING: 96.2% of the mix's cache is the single global layer.** The five SWA
layers contribute **3.76%** of it. So the 5:1 mix is, to within 4%, "one sixth of
full attention": quadrupling the window from 1024 to 4096 moves the total only
from 7.44 GB to 8.28 GB, 11%. The design's ceiling is
`n_layers / global_layers = 6`, and it reaches 5.8 of it.

**FINDING: the mix pays 22x the pure-SWA cache to keep one global layer.** 7.44
GB against 0.34 GB. That is the price of retrieval over the full context, and it
is worth stating as a price rather than as a shrink: against pure SWA the mix is
not an optimisation, it is a 22x regression bought deliberately.

**FINDING: the capstone cannot express the ratio.** 3 layers, and 5:1 wants 6.
The nearest expressible mixes are 2:1 and 1:2, whose ceilings are 3 and 1.5, so
the comparison the exercise describes cannot be run in the model it names --
before considering that `block_size=64` makes every window global anyway.

**CONTROL: the loss and generation-quality comparisons are undefined here.** With
`block_size=64` the pure-SWA and pure-global baselines are bit-identical models,
so their losses match exactly and there is nothing to compare -- which is
Exercise 2's finding arriving one exercise later.

Structure: `cache` is the lesson's own byte formula; `mix` composes the 5:1
total; `expressible` checks the layer count against the ratio.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "15-attention-variants"
LAYERS, KV_HEADS, HEAD_DIM, CONTEXT = 80, 8, 128, 131_072
SWA_LAYERS, GLOBAL_LAYERS, WINDOWS, CAPSTONE = 5, 1, (4_096, 1_024), 3


def mix(ref, window, swa=SWA_LAYERS, glob=GLOBAL_LAYERS):
    """KV bytes for a swa:global layer mix, as main() composes it."""
    full = ref.kv_cache_bytes(LAYERS, KV_HEADS, HEAD_DIM, CONTEXT)
    total = swa + glob
    return full * (swa / total) * (window / CONTEXT) + full * (glob / total)


def expressible(layers, swa=SWA_LAYERS, glob=GLOBAL_LAYERS):
    """Can `layers` be laid out in this ratio, and what ceiling does the ratio imply?"""
    return layers % (swa + glob) == 0, (swa + glob) / glob


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    full = ref.kv_cache_bytes(LAYERS, KV_HEADS, HEAD_DIM, CONTEXT)
    totals = {w: mix(ref, w) for w in WINDOWS}
    pure = {w: full * (w / CONTEXT) for w in WINDOWS}
    share = full * (GLOBAL_LAYERS / (SWA_LAYERS + GLOBAL_LAYERS))
    return {
        "full": full, "mix": totals, "pure": pure, "global_part": share,
        "shrink": {w: full / t for w, t in totals.items()},
        "ceiling": expressible(LAYERS)[1],
        "capstone": expressible(CAPSTONE), "layers": CAPSTONE,
        "cost": totals[1_024] / pure[1_024],
        "window_effect": abs(totals[4_096] - totals[1_024]) / totals[1_024],
    }


def verify(result):
    totals, pure, shrink = result["mix"], result["pure"], result["shrink"]
    return [
        practice.Check(
            "ANSWER: 7.44 GB, a 5.8x shrink against a ceiling of exactly 6",
            5.5 < shrink[1_024] < result["ceiling"],
            f"full attention is {result['full'] / 1e9:.1f} GB; the 5:1 mix at W=1024 is "
            f"{totals[1_024] / 1e9:.2f} GB, {shrink[1_024]:.1f}x. The ceiling is "
            f"n_layers / global_layers = {result['ceiling']:.0f}, so the design reaches "
            f"{shrink[1_024] / result['ceiling']:.0%} of what its ratio allows",
        ),
        practice.Check(
            "FINDING: 96% of the mix's cache is the single global layer",
            result["global_part"] / totals[1_024] > 0.95,
            f"the global sixth is {result['global_part'] / 1e9:.2f} GB of the "
            f"{totals[1_024] / 1e9:.2f} GB total -- "
            f"{result['global_part'] / totals[1_024]:.1%} -- and the five SWA layers contribute "
            f"{1 - result['global_part'] / totals[1_024]:.2%}. The window barely enters: moving W "
            f"from 4096 to 1024 changes the total by {result['window_effect']:.1%}",
        ),
        practice.Check(
            "FINDING: the mix pays 22x the pure-SWA cache to keep one global layer",
            result["cost"] > 15,
            f"{totals[1_024] / 1e9:.2f} GB against pure SWA's {pure[1_024] / 1e9:.2f} GB at the "
            f"same window, {result['cost']:.0f}x. Against full attention the mix is a 5.8x win; "
            "against pure SWA it is a deliberate regression, bought to keep retrieval over the "
            "whole context. Both statements are about the same number",
        ),
        practice.Check(
            "FINDING: the capstone cannot express the ratio",
            not result["capstone"][0] and result["layers"] % (SWA_LAYERS + GLOBAL_LAYERS) != 0,
            f"the capstone has {result['layers']} layers and 5:1 wants a multiple of "
            f"{SWA_LAYERS + GLOBAL_LAYERS}: {result['layers']} % "
            f"{SWA_LAYERS + GLOBAL_LAYERS} = {result['layers'] % (SWA_LAYERS + GLOBAL_LAYERS)}. "
            "The nearest expressible mixes in 3 layers are 2:1 and 1:2, with ceilings of 3 and 1.5",
        ),
        practice.Check(
            "CONTROL: the loss and quality comparisons are undefined at block_size=64",
            result["layers"] == CAPSTONE,
            "Exercise 2 shows any window at or above 64 is full causal attention on a 64-token "
            "block, so the pure-SWA and pure-global baselines are bit-identical models there. "
            "Their losses match exactly and their samples match exactly, which is why this "
            "exercise is answered on the cache arithmetic rather than on a training run",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
