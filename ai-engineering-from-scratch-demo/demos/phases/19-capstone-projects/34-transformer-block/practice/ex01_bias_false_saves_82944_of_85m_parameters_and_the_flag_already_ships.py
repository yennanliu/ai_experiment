"""Exercise 1 — bias=False saves 82,944 of 85,054,464 parameters (0.098%), and the flag already ships.

    Add a `bias=False` flag to every linear in the block. Modern open weights LLMs ship without biases on the linear layers. Measure how many parameters you save in a 12 layer 768 dim model.

Reading of the exercise: "a 12 layer 768 dim model" is twelve of the
lesson's `TransformerBlock`s at d_model = 768 with GPT-2 small's 12 heads and
1,024 context, counted with and without linear biases. The lesson already has
the flag -- `BlockConfig.use_bias`, passed to all four `nn.Linear`s -- so the
solution uses it rather than adding a second one, and checks that it really
reaches every linear. Blocks are built on torch's `meta` device, so 85M
parameters are counted without allocating 340 MB.

**ANSWER: 82,944 parameters, 0.098% of the stack.** Each block drops 6,912
biases: fused QKV 2,304, output projection 768, MLP up 3,072 and MLP down 768.
Twelve blocks go from 85,054,464 to 84,971,520 parameters. Against the "124
million parameter GPT" the lesson says these blocks assemble into, the saving
is 0.067%. With `use_bias=False` all 48 linears have `bias is None`.

**FINDING: the flag the exercise asks for is already in the code.**
`BlockConfig` has `use_bias: bool = True`, and `MultiHeadAttention` and
`FeedForward` pass it as `bias=` to every linear. The lesson text never
mentions it, so a reader following "Add a flag" writes a second one.

**FINDING: bias=False does not make the block bias-free.** Each hand-rolled
`LayerNorm` keeps a learnable `shift` of 768, so 18,432 bias parameters
survive in 12 blocks: 22% as many as the flag removes. The LLaMA-style
RMSNorm the lesson points to has no shift at all.

Structure: `blocks()` builds the stack on the meta device; `solve()` counts
parameters by kind for both settings.
"""

from __future__ import annotations

import dataclasses
import re

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "34-transformer-block"
LAYERS, D_MODEL, HEADS = 12, 768, 12


def blocks(ref, use_bias):
    """Twelve GPT-2-small-sized blocks with no storage behind their tensors."""
    cfg = ref.BlockConfig(d_model=D_MODEL, num_heads=HEADS, use_bias=use_bias)
    with torch.device("meta"):
        return torch.nn.ModuleList(ref.TransformerBlock(cfg) for _ in range(LAYERS))


def counts(stack):
    linears = [m for m in stack.modules() if isinstance(m, torch.nn.Linear)]
    named = dict(stack.named_parameters())
    return {
        "total": sum(p.numel() for p in named.values()),
        "shift": sum(p.numel() for n, p in named.items() if n.endswith(".shift")),
        "linears": len(linears),
        "with_bias": sum(m.bias is not None for m in linears),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with_bias, without = counts(blocks(ref, True)), counts(blocks(ref, False))
    per_block = {n: m.bias.numel() for n, m in blocks(ref, True)[0].named_modules()
                 if isinstance(m, torch.nn.Linear)}
    fields = {f.name: f.default for f in dataclasses.fields(ref.BlockConfig)}
    doc = parity.doc_text(PHASE, LESSON)
    return {
        "with": with_bias, "without": without, "per_block": per_block,
        "saved": with_bias["total"] - without["total"],
        "flag_default": fields.get("use_bias"),
        "doc_mentions_flag": "use_bias" in doc,
        "gpt_millions": int(re.search(r"(\d+) million parameter GPT", doc).group(1)),
    }


def verify(result):
    r = result
    saved, total = r["saved"], r["with"]["total"]
    return [
        practice.Check(
            "ANSWER: 82,944 parameters, 0.098% of the stack",
            all([
                saved == 82_944,
                (total, r["without"]["total"]) == (85_054_464, 84_971_520),
                r["per_block"] == {"attn.qkv": 2304, "attn.out_proj": 768,
                                   "mlp.fc1": 3072, "mlp.fc2": 768},
                r["without"]["with_bias"] == 0,
                r["without"]["linears"] == 48,
                round(100 * saved / total, 3) == 0.098,
                round(100 * saved / (r["gpt_millions"] * 1e6), 3) == 0.067,
            ]),
            f"{total:,} -> {r['without']['total']:,}, saved {saved:,} "
            f"({saved / total:.3%}; {saved / (r['gpt_millions'] * 1e6):.3%} of "
            f"{r['gpt_millions']}M); per block {r['per_block']}; linears with bias after the "
            f"flag {r['without']['with_bias']}/{r['without']['linears']}",
        ),
        practice.Check(
            "FINDING: the flag the exercise asks for is already in the code",
            all([
                r["flag_default"] is True,
                not r["doc_mentions_flag"],
            ]),
            f"BlockConfig.use_bias default {r['flag_default']}; lesson text mentions it: "
            f"{r['doc_mentions_flag']}",
        ),
        practice.Check(
            "FINDING: bias=False does not make the block bias-free",
            all([
                r["without"]["shift"] == r["with"]["shift"] == 18_432,
                round(r["without"]["shift"] / saved, 2) == 0.22,
            ]),
            f"LayerNorm shift parameters left after bias=False: {r['without']['shift']:,} "
            f"({r['without']['shift'] / saved:.0%} of the {saved:,} removed)",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
