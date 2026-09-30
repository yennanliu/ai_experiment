"""Exercise 1 -- untying adds 38,597,376 parameters, the lesson's dedup is a no-op, and state_dict double-counts.

    Untie the LM head from the token embedding and recount parameters. Verify the delta is 50257 times 768 = 38 million.

Reading of the exercise: the lesson's `GPTConfig` already has the switch
(`weight_tying`), so untying means building the reference 124M model with
`weight_tying=False` and counting both models with the lesson's own
`count_parameters`. Both models are built on torch's `meta` device, so 287M
parameters are counted without allocating 1.1 GB. "Verify the delta" is read
as an exact integer comparison with 50257 x 768.

**ANSWER: the delta is exactly 38,597,376 = 50257 x 768.** Tied, the model
has 124,439,808 parameters; untied it has 163,037,184. The shared matrix is
31.0% of the tied model and 23.7% of the untied one. Rounded, 38,597,376 is
38.6M, so "38 million" is a truncation; it rounds to 39M. In the tied model
`lm_head.weight is tok_embed.weight`; in the untied one it is not.

**FINDING: the lesson's deduplicating counter does nothing that
`parameters()` does not already do.** `count_parameters` keys a dict by
`id(param)` "so weight tying is honored", but `nn.Module.parameters()`
already yields a shared tensor once: a plain `sum(p.numel())` also gives
124,439,808. Counting only goes wrong on paths the lesson never mentions:
`named_parameters(remove_duplicate=False)` and `state_dict()` both list
`lm_head.weight` beside `tok_embed.weight` and give 163,037,184. That is the
untied count, reported for a tied model. Anyone who sizes a checkpoint by
summing its state dict (lesson 37 loads one) is off by 38.6M.

**FINDING: the lesson's per-block arithmetic is low by one million.** It says
"Twelve blocks at roughly 7 million parameters each is 84 million". One block
has 7,087,872 parameters, and twelve have 85,054,464. With its own rounded
pieces (38M + 0.79M + 84M) the lesson's "sum the pieces" gives 122.8M, not
124M; the measured pieces give 124.4M.

Structure: `build()` makes a model on the meta device; `solve()` counts it
four ways.
"""

from __future__ import annotations

import re

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "35-gpt-model-assembly"


def build(ref, tied):
    """The lesson's 124M GPT with no storage behind its tensors."""
    with torch.device("meta"):
        return ref.GPTModel(ref.GPTConfig(weight_tying=tied))


def counts(ref, model):
    return {
        "lesson": ref.count_parameters(model),
        "plain": sum(p.numel() for p in model.parameters()),
        "no_dedup": sum(p.numel() for _, p in model.named_parameters(remove_duplicate=False)),
        "state_dict": sum(t.numel() for t in model.state_dict().values()),
        "shared": model.lm_head.weight is model.tok_embed.weight,
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cfg = ref.GPTConfig()
    tied_model = build(ref, True)
    doc = parity.doc_text(PHASE, LESSON)
    return {
        "tied": counts(ref, tied_model),
        "untied": counts(ref, build(ref, False)),
        "vocab_x_d": cfg.vocab_size * cfg.d_model,
        "block": sum(p.numel() for p in tied_model.blocks[0].parameters()),
        "layers": cfg.num_layers,
        "doc_blocks": re.search(r"roughly (\d+) million parameters each is (\d+) million", doc).groups(),
    }


def verify(result):
    r = result
    tied, untied, delta = r["tied"], r["untied"], r["untied"]["lesson"] - r["tied"]["lesson"]
    return [
        practice.Check(
            "ANSWER: the delta is exactly 38,597,376 = 50257 x 768",
            delta == r["vocab_x_d"] == 38_597_376
            and (tied["lesson"], untied["lesson"]) == (124_439_808, 163_037_184)
            and tied["shared"] and not untied["shared"]
            and round(delta / 1e6) == 39,
            f"tied {tied['lesson']:,}, untied {untied['lesson']:,}, delta {delta:,} "
            f"({delta / tied['lesson']:.1%} of tied, {delta / untied['lesson']:.1%} of untied; "
            f"{delta / 1e6:.1f}M rounds to {round(delta / 1e6)}M)",
        ),
        practice.Check(
            "FINDING: the lesson's deduplicating counter does nothing parameters() does not",
            tied["plain"] == tied["lesson"]
            and tied["no_dedup"] == tied["state_dict"] == untied["lesson"],
            f"tied model: lesson {tied['lesson']:,}, plain parameters() {tied['plain']:,}, "
            f"remove_duplicate=False {tied['no_dedup']:,}, state_dict {tied['state_dict']:,}",
        ),
        practice.Check(
            "FINDING: the lesson's per-block arithmetic is low by one million",
            r["doc_blocks"] == ("7", "84") and r["block"] == 7_087_872
            and r["block"] * r["layers"] == 85_054_464,
            f"doc: ~{r['doc_blocks'][0]}M each, {r['doc_blocks'][1]}M total; measured "
            f"{r['block']:,} per block, {r['block'] * r['layers']:,} for {r['layers']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
