"""Exercise 3 — the bridge pays back at a twenty-million-parameter LLM.

    Compare parameter counts: Q-Former (12 layers, 768 hidden) vs a 2-layer MLP
    projector (1408 → 4096, two layers). At what LLM scale does the 188M
    Q-Former cost pay back in training efficiency?

Reading of the exercise: both counts are built from the lesson's own stated
shapes -- 256 patch tokens of 1408, a 4096-wide LLM, 32 queries, 12 layers at
768 -- rather than quoted, so the 188M can be checked rather than assumed. And
"pays back in training efficiency" is read as FLOPs per training image: the
bridge saves the LLM 224 tokens of forward-and-backward and charges its own
parameters over the 32 queries it runs, so the break-even is a size of LLM.

**ANSWER: 18.5M parameters.** The saving scales with the LLM and the cost does
not, so the break-even LLM is smaller than the bridge. At OPT-2.7B -- the
smallest LLM the lesson names -- the Q-Former is already **146x** past payback,
and at 11B, **594x**. There is no LLM scale at which the MLP is cheaper to
train; the choice has to be made on something else.

**FINDING: the lesson's own shapes do not reconstruct 188M.** Cross-attention
in every layer, the BERT-base embedding table, the 32 queries and the 768 ->
4096 projection give **152.2M** -- **19.0%** short. Cross-attention in every
other layer, the other reading of "12 layers", gives 132.2M and is further away.
Neither assumption in the lesson closes the gap.

**FINDING: the bridge is 6.75x the projector, and the gap is mostly not the
bridge.** The MLP projector is **22.55M**; of the Q-Former's 152.2M, **23.8M**
is the BERT embedding table and **125.2M** the blocks. The comparison the
exercise asks for is between a projector and a whole language encoder.

**FINDING: the break-even moves the wrong way with LLaVA's shape.** At 576
visual tokens rather than 256 the saving per image is 544 tokens and break-even
drops to **7.6M**. The bridge is easier to justify against the configuration
that abandoned it.

Structure: `qformer_params` is the layer-by-layer count under one
cross-attention policy, `mlp_params` is the two-layer projector, and
`breakeven` solves 6*L*(t - 32) = 6*dP*32 for L.
"""

from __future__ import annotations

from harness import practice

HIDDEN, ENCODER, LLM_DIM, LAYERS = 768, 1408, 4096, 12
VOCAB, MAX_POS, QUERIES = 30522, 512, 32
PAPER = 188_000_000
VISUAL_TOKENS = {"lesson ViT, 256": 256, "LLaVA 24x24, 576": 576}
LLM_SCALES = {"OPT-2.7B": 2.7e9, "OPT-6.7B": 6.7e9, "Flan-T5-XXL 11B": 11.0e9}
TRAIN_FLOPS = 6            # forward + backward, per parameter per token
LESSON_TOKENS, LLAVA_TOKENS = 256, 576


def qformer_params(every_layer=True, hidden=HIDDEN):
    """Parameter count from the lesson's stated shapes, by component."""
    self_attn = 4 * hidden * hidden + 4 * hidden
    cross = 2 * hidden * hidden + 2 * ENCODER * hidden + 4 * hidden
    ffn = 2 * hidden * 4 * hidden + 5 * hidden
    norms = 3 * 2 * hidden
    crosses = LAYERS if every_layer else LAYERS // 2
    blocks = (self_attn + ffn + norms) * LAYERS + cross * crosses
    embeddings = VOCAB * hidden + MAX_POS * hidden + 4 * hidden
    projection = hidden * LLM_DIM + LLM_DIM
    return {"blocks": blocks, "embeddings": embeddings,
            "queries": QUERIES * hidden, "projection": projection,
            "total": blocks + embeddings + QUERIES * hidden + projection}


def mlp_params():
    """LLaVA's 2-layer projector: 1408 -> 4096 -> 4096."""
    return ENCODER * LLM_DIM + LLM_DIM + LLM_DIM * LLM_DIM + LLM_DIM


def breakeven(extra, visual_tokens):
    """The LLM size at which the tokens saved pay for the bridge's own FLOPs."""
    return extra * QUERIES / (visual_tokens - QUERIES)


def payback(extra, llm, visual_tokens):
    """Ratio of LLM FLOPs saved per image to the bridge's own FLOPs per image."""
    saved = (visual_tokens - QUERIES) * TRAIN_FLOPS * llm
    return saved / (QUERIES * TRAIN_FLOPS * extra)


def solve():
    full, halved = qformer_params(), qformer_params(every_layer=False)
    projector = mlp_params()
    extra = full["total"] - projector
    return {
        "components": full, "total": full["total"], "halved": halved["total"],
        "shortfall": PAPER - full["total"],
        "shortfall_pct": round((PAPER - full["total"]) / PAPER * 100, 1),
        "projector": projector, "ratio": round(full["total"] / projector, 2),
        "extra": extra,
        "breakeven": {name: round(breakeven(extra, tokens) / 1e6, 1)
                      for name, tokens in VISUAL_TOKENS.items()},
        "payback": {name: round(payback(extra, size, VISUAL_TOKENS["lesson ViT, 256"]))
                    for name, size in LLM_SCALES.items()},
    }


def verify(result):
    components, breakevens = result["components"], result["breakeven"]
    return [
        practice.Check(
            "ANSWER: the bridge pays back at an 18.5M-parameter LLM",
            all([breakevens["lesson ViT, 256"] == 18.5,
                 result["payback"]["OPT-2.7B"] == 146,
                 result["payback"]["Flan-T5-XXL 11B"] == 594]),
            f"the LLM saves {LESSON_TOKENS - QUERIES} tokens of "
            f"forward-and-backward per image and the bridge charges its own "
            f"{result['extra'] / 1e6:.1f}M extra parameters over {QUERIES} queries, so "
            f"break-even is {breakevens['lesson ViT, 256']}M parameters. "
            f"{result['payback']} times past it at the scales the lesson names -- there is "
            "no LLM at which the MLP is cheaper to train",
        ),
        practice.Check(
            "FINDING: the lesson's own shapes do not reconstruct 188M",
            all([result["total"] == 152_229_376, result["shortfall_pct"] == 19.0,
                 result["halved"] == 132_156_928, result["halved"] < result["total"]]),
            f"cross-attention in every layer, the BERT-base embedding table, {QUERIES} "
            f"queries and the {HIDDEN} -> {LLM_DIM} projection give "
            f"{result['total'] / 1e6:.1f}M -- {result['shortfall_pct']}% short of the "
            f"paper's {PAPER / 1e6:.0f}M. Cross-attention in every other layer, the other "
            f"reading of '12 layers', gives {result['halved'] / 1e6:.1f}M and is further "
            "away; neither assumption in the lesson closes the gap",
        ),
        practice.Check(
            "FINDING: the bridge is 6.75x the projector, and the gap is mostly not the bridge",
            all([result["projector"] == 22_552_576, result["ratio"] == 6.75,
                 components["embeddings"] == 23_837_184,
                 components["blocks"] == 125_217_792]),
            f"the MLP projector is {result['projector'] / 1e6:.2f}M against the Q-Former's "
            f"{result['total'] / 1e6:.1f}M, {result['ratio']}x. Of that, "
            f"{components['embeddings'] / 1e6:.1f}M is the BERT embedding table and "
            f"{components['blocks'] / 1e6:.1f}M the blocks -- the comparison is between a "
            "projector and a whole language encoder",
        ),
        practice.Check(
            "FINDING: the break-even moves the wrong way with LLaVA's shape",
            breakevens["LLaVA 24x24, 576"] == 7.6,
            f"at {LLAVA_TOKENS} visual tokens the saving is "
            f"{LLAVA_TOKENS - QUERIES} tokens an image and "
            f"break-even drops to {breakevens['LLaVA 24x24, 576']}M. The bridge is "
            "easiest to justify against the configuration that abandoned it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
