"""Exercise 1 — the 22M is a different encoder.

    Compute the trainable-parameter count for the 2-layer MLP projector at
    `1024 → 4096 → 4096`. With GELU and bias, what fraction of LLaVA-13B does it
    represent?

Reading of the exercise: the count is derived from a formula and the formula is
checked against the lesson's own `MLPProjector.num_params()` at three small
shapes, because building the real one in pure Python would draw 21 million
Gaussians to answer an arithmetic question. "With GELU and bias" is taken
literally -- both are priced, and one of them is free.

**ANSWER: 20,979,712 parameters, 0.161% of LLaVA-13B.** GELU contributes **0**
-- it has no parameters -- and the two bias vectors contribute **8,192**, which
is 0.04% of the projector. The phrase in the exercise names two things and only
one of them is a number.

**FINDING: the lesson's own "22M" is a 1408-wide encoder, not this one.** At
`1024 → 4096 → 4096` the answer is 20.98M; at 1152 (SigLIP SO400m) it is 21.50M;
at **1408** it is **22,552,576**, which is the figure the takeaway prints and
the same number Lesson 12.03 uses for LLaVA's projector. The exercise names
CLIP ViT-L's 1024 and the lesson quotes the EVA/BLIP-2 width.

**FINDING: 80% of the projector does not change the dimension.** The second
layer is 4096 → 4096 -- **16,781,312** of the 20,979,712 -- while the first
layer already performs the only mapping the projector exists for. A single
1024 → 4096 linear is **5.0x** smaller.

**FINDING: the parameter fraction is the wrong measure of what this costs.**
0.161% of LLaVA-13B is 21M parameters; the same projector emits **576** tokens
per image, **28.1%** of a 2,048-token context. The cheap component is the one
that sets the context bill.

Structure: `params` is the two-layer formula, `PARITY` checks it against the
lesson's own `num_params()` at three shapes, and `WIDTHS` is the encoder-width
sweep that locates the 22M.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "05-llava-visual-instruction-tuning"
IN_DIM, HIDDEN, OUT_DIM = 1024, 4096, 4096
WIDTHS = {"CLIP ViT-L, 1024": 1024, "SigLIP SO400m, 1152": 1152, "EVA/BLIP-2, 1408": 1408}
LLAVA_13B, QUOTED = 13.0e9, 22_552_576
VISUAL_TOKENS, SMALL_CONTEXT = 576, 2048
PARITY = ((16, 32, 24), (8, 16, 12), (4, 4, 4))


def params(in_dim, hidden, out_dim):
    """Two linear layers with bias. GELU has none."""
    return hidden * in_dim + hidden + out_dim * hidden + out_dim


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    total = params(IN_DIM, HIDDEN, OUT_DIM)
    widths = {name: params(width, HIDDEN, OUT_DIM) for name, width in WIDTHS.items()}
    second = OUT_DIM * HIDDEN + OUT_DIM
    single = HIDDEN * IN_DIM + HIDDEN
    return {
        "total": total, "biases": HIDDEN + OUT_DIM, "gelu": 0,
        "bias_share": round((HIDDEN + OUT_DIM) / total * 100, 2),
        "share_13b": round(total / LLAVA_13B * 100, 3),
        "widths": widths, "quoted": QUOTED,
        "matches_quoted": [name for name, count in widths.items() if count == QUOTED],
        "second_layer": second,
        "second_share": round(second / total * 100, 1),
        "single_linear": single, "single_ratio": round(total / single, 1),
        "visual_tokens": VISUAL_TOKENS,
        "context_share": round(VISUAL_TOKENS / SMALL_CONTEXT * 100, 1),
        "parity": [(shape, ref.MLPProjector(*shape).num_params(), params(*shape))
                   for shape in PARITY],
    }


def verify(result):
    widths, checks = result["widths"], result["parity"]
    return [
        practice.Check(
            "ANSWER: 20,979,712 parameters, 0.161% of LLaVA-13B",
            all([result["total"] == 20_979_712, result["share_13b"] == 0.161,
                 result["gelu"] == 0, result["biases"] == 8192,
                 all(measured == derived for _, measured, derived in checks)]),
            f"{IN_DIM} -> {HIDDEN} -> {OUT_DIM} with bias is {result['total']:,} parameters, "
            f"{result['share_13b']}% of a 13B LLM. GELU contributes {result['gelu']} and the "
            f"biases {result['biases']:,}, {result['bias_share']}% of the projector. The "
            f"formula agrees with the lesson's own num_params() at all {len(checks)} shapes "
            f"checked: {[(s, m) for s, m, _ in checks]}",
        ),
        practice.Check(
            "FINDING: the lesson's own 22M is a 1408-wide encoder, not this one",
            all([result["matches_quoted"] == ["EVA/BLIP-2, 1408"],
                 widths["CLIP ViT-L, 1024"] == 20_979_712,
                 widths["SigLIP SO400m, 1152"] == 21_504_000]),
            f"across encoder widths the projector is {widths}; the takeaway's 22M is "
            f"{result['quoted']:,}, which only {result['matches_quoted']} produces. The "
            "exercise names CLIP ViT-L's 1024 and the lesson quotes the EVA/BLIP-2 width -- "
            "the same number Lesson 12.03 uses for LLaVA's projector",
        ),
        practice.Check(
            "FINDING: 80% of the projector does not change the dimension",
            all([result["second_layer"] == 16_781_312, result["second_share"] == 80.0,
                 result["single_ratio"] == 5.0]),
            f"the second layer is {OUT_DIM} -> {OUT_DIM} and holds "
            f"{result['second_layer']:,} of {result['total']:,} parameters, "
            f"{result['second_share']}%, while the first layer already performs the only "
            f"mapping the projector exists for. A single {IN_DIM} -> {HIDDEN} linear is "
            f"{result['single_linear']:,}, {result['single_ratio']}x smaller",
        ),
        practice.Check(
            "FINDING: the parameter fraction is the wrong measure of what this costs",
            all([result["context_share"] == 28.1, result["share_13b"] < 0.2]),
            f"{result['share_13b']}% of LLaVA-13B is {result['total'] / 1e6:.1f}M parameters, "
            f"and the same projector emits {result['visual_tokens']} tokens an image -- "
            f"{result['context_share']}% of a {SMALL_CONTEXT:,}-token context. The cheap "
            "component is the one that sets the context bill",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
