"""Exercise 1 — half of the change costs exactly zero parameters.

    Modify the model to use 24 layers and 16 heads instead of 12/12. Count the
    parameters. How does doubling the depth compare to doubling the width
    (embedding dimension)?

Reading of the exercise: "12/12" is `MiniGPT`'s own default -- 12 layers, 12
heads, 768 dimensions, the GPT-2 Small configuration the lesson's
`parameter_breakdown` prints -- and the counts are the model's own
`count_parameters`, not a formula. The asked-for change moves two knobs at once,
so each is also moved alone. "Doubling the width" is ambiguous about `ff_dim`,
which the lesson keeps at `4 * embed_dim`, so both readings are counted.

**ANSWER: 124,402,944 -> 209,420,544, up 68.3%.** And the baseline is exactly
GPT-2 Small's 124M, which is the check that the counter is right.

**FINDING: the head count contributes none of it.** 12 heads to 16 at the same
768 dimensions is a delta of **0 parameters**. `head_dim = embed_dim //
num_heads`, so the four projection matrices stay `768 x 768` however the heads
divide them up: heads are a reshape. The whole +68.3% is the depth change, and
the exercise moves two knobs where one of them is free.

**ANSWER to the comparison: depth is 3.5x cheaper than width.** Doubling depth
adds 68.3%; doubling `embed_dim` to 1536 with `ff_dim` following it adds
**236.5%**. Per-block matrices are `O(d^2)` and there are six of them, so width
squares what depth multiplies -- and width also doubles the 38.6M embedding
table, which depth leaves alone.

**FINDING: the ambiguity is worth 113M parameters.** Hold `ff_dim` at 3072
while doubling `embed_dim` and the answer is +145.5% instead of +236.5%. The
lesson's constructor defaults `ff_dim=3072` independently of `embed_dim`, so
"doubling the width" has two defensible readings **113,283,072** parameters
apart -- nearly a whole GPT-2 Small -- and the exercise names neither.

Structure: `count` builds one `MiniGPT` at a named configuration and returns its
own count; every row below is one call.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "04-pre-training-mini-gpt"
BASE = dict(vocab_size=50257, embed_dim=768, num_heads=12, num_layers=12,
            max_seq_len=1024, ff_dim=3072)
CONFIGS = {
    "baseline": {},
    "asked": dict(num_layers=24, num_heads=16),
    "heads": dict(num_heads=16),
    "depth": dict(num_layers=24),
    "width": dict(embed_dim=1536, ff_dim=6144),
    "width_ff_fixed": dict(embed_dim=1536),
}


def count(ref, **overrides):
    """One MiniGPT at `BASE` plus `overrides`, counted by its own method."""
    return ref.MiniGPT(**dict(BASE, **overrides)).count_parameters()


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    counts = {name: count(ref, **cfg) for name, cfg in CONFIGS.items()}
    base = counts["baseline"]
    return {
        "counts": counts,
        "growth": {name: n / base - 1 for name, n in counts.items()},
        "embedding": ref.MiniGPT(**BASE).embedding.token_embed.size,
        "head_dims": (BASE["embed_dim"] // BASE["num_heads"], BASE["embed_dim"] // 16),
    }


def verify(result):
    counts, growth = result["counts"], result["growth"]
    return [
        practice.Check(
            "ANSWER: 124,402,944 -> 209,420,544, up 68.3%, and the baseline is GPT-2 Small",
            counts["baseline"] == 124_402_944 and counts["asked"] == 209_420_544,
            f"the lesson's own count_parameters gives {counts['baseline']:,} at 12 layers, 12 "
            f"heads and 768 dimensions -- GPT-2 Small's published 124M, which is the check that "
            f"the counter is right -- and {counts['asked']:,} at 24 layers and 16 heads, "
            f"{100 * growth['asked']:+.1f}%",
        ),
        practice.Check(
            "FINDING: the head count contributes exactly none of that",
            counts["heads"] == counts["baseline"] and counts["asked"] == counts["depth"],
            f"12 heads to 16 at the same width is a delta of {counts['heads'] - counts['baseline']} "
            f"parameters. head_dim = embed_dim // num_heads, so the four projections stay "
            f"{BASE['embed_dim']}x{BASE['embed_dim']} whether they are cut into "
            f"{result['head_dims'][0]}-wide heads or {result['head_dims'][1]}-wide ones -- heads "
            "are a reshape of the same matrix. The exercise moves two knobs and one of them is "
            "free, so the whole change is the depth",
        ),
        practice.Check(
            "ANSWER to the comparison: depth is 3.5x cheaper than width",
            counts["depth"] == counts["asked"] and growth["width"] > 3 * growth["depth"],
            f"doubling the layers adds {100 * growth['depth']:.1f}% and doubling embed_dim to "
            f"1536 with ff_dim following adds {100 * growth['width']:.1f}%, a ratio of "
            f"{growth['width'] / growth['depth']:.1f}. Per-block matrices are O(d^2) and there "
            f"are six of them, so width squares what depth multiplies -- and width also doubles "
            f"the {result['embedding']:,}-parameter embedding table, which is "
            f"{100 * result['embedding'] / counts['baseline']:.0f}% of the baseline and which "
            "depth leaves untouched",
        ),
        practice.Check(
            "FINDING: 'doubling the width' has two readings 113M parameters apart",
            counts["width"] - counts["width_ff_fixed"] > counts["baseline"] * 0.9,
            f"MiniGPT's constructor defaults ff_dim=3072 independently of embed_dim, so doubling "
            f"embed_dim while leaving ff_dim alone gives {counts['width_ff_fixed']:,} "
            f"({100 * growth['width_ff_fixed']:+.1f}%) against {counts['width']:,} "
            f"({100 * growth['width']:+.1f}%) when ff_dim follows. The two defensible readings "
            f"of the exercise's own question are {counts['width'] - counts['width_ff_fixed']:,} "
            "parameters apart, and it names neither",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
