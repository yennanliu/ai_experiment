"""Exercise 5 — the 32-query pool makes a decode step 2.9x cheaper, but a full pass at ex03's shape costs the same.

    Swap the cross-attention layer for a Q-Former-style attention block where a fixed 32-token query pool attends to image features once per layer.

Reading of the exercise: every `DecoderBlock` of a default seeded decoder has
its `cross_attn` replaced by `QFormerCross`. Each layer has its own 32
learned queries (hidden 256), which read the image through one lesson
`CrossAttention`, with a residual. The text then cross-attends to those 32
pooled tokens through a second lesson `CrossAttention` built with
`vision_dim=hidden`. Both are the reference class, used unchanged. Costs are
exact layer-0 counts from `FlopCounterMode`, at batch 1, for the demo's shape
(Nt=10, Nv=197) and for ex03's (Nt=64, Nv=576).

**ANSWER: the swap works, and the text reads 32 pooled tokens instead of
197.** The logits keep their shape, (2, 10, 1024), and still move (by 0.042)
when the image changes. The lesson's cache path works unchanged: cached and
uncached logits differ by 0.0, and each layer's K cache shrinks from
(2, 8, 197, 32) to (2, 8, 32, 32). The swap adds 1,085,440 parameters
(4 x 32 x 256 queries plus 4 more `CrossAttention` modules), taking the model
from 4,746,752 to 5,832,192.

**FINDING: the pool pays off only per decode step, not per forward pass.**

| layer-0 FLOPs | lesson | Q-Former |
|---|---:|---:|
| Nt=10, Nv=197: whole cross sub-layer | 56,281,088 | 77,824,000 (+38%) |
| Nt=10, Nv=197: text side, image work cached | 4,638,720 | 2,949,120 |
| Nt=64, Nv=576: whole cross sub-layer | 205,520,896 | 205,520,896 |
| Nt=64, Nv=576: text side, image work cached | 54,525,952 | 18,874,368 (2.9x less) |

The pool still projects every image token into K and V, which ex03 found is
most of the cost. Pooling pays for itself only once Nt > Q(2D + Nv)/(Nv - Q),
which is exactly Nt = 64 at Nv = 576. Since `VisionLanguageDecoder` rebuilds
its cache on every call (ex03), the lesson's own decoder never gets the
per-step saving. BLIP-2 also differs in where it puts this: it runs one
Q-Former, 32 queries of dimension 768 with cross-attention in every other
block, before the LM (https://arxiv.org/html/2301.12597, read 2026-09-29),
rather than one pool per decoder layer.

Structure: `QFormerCross` is the swapped layer; `build()` installs it;
`cross_flops()` counts one layer's cross sub-layer.
"""

from __future__ import annotations

import dataclasses

from harness import parity, practice

try:
    import torch
    from torch import nn
    from torch.utils.flop_counter import FlopCounterMode
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "61-cross-attention-fusion"
QUERIES = 32


class QFormerCross(nn.Module):
    """32 learned queries read the image; the text reads the 32 pooled tokens."""

    def __init__(self, ref, cfg):
        super().__init__()
        self.queries = nn.Parameter(0.02 * torch.randn(QUERIES, cfg.hidden))
        self.pool = ref.CrossAttention(cfg)
        self.read = ref.CrossAttention(dataclasses.replace(cfg, vision_dim=cfg.hidden))

    def pooled(self, memory):
        q = self.queries.expand(memory.shape[0], -1, -1)
        return q + self.pool(q, memory)

    def project_memory(self, memory):
        return self.read.project_memory(self.pooled(memory))

    def forward(self, x, memory, kv_cache=None):
        return self.read(x, self.pooled(memory), kv_cache=kv_cache)


def build(ref, nt, nv, qformer):
    cfg = ref.DecoderConfig(vision_tokens=nv, max_text_len=max(nt, 32))
    torch.manual_seed(0)
    dec = ref.VisionLanguageDecoder(cfg).eval()
    if qformer:
        for block in dec.blocks:
            block.cross_attn = QFormerCross(ref, cfg)
    return dec


def inputs(ref, batch, nt, nv, seed=0):
    return ref.synth_text(batch, nt, 1024, seed=seed), ref.synth_memory(batch, nv, 256, seed=1)


def cross_flops(ref, nt, nv, qformer):
    """Layer-0 FLOPs: whole cross sub-layer, and the part left once image-side work is cached."""
    dec, (ids, memory) = build(ref, nt, nv, qformer), inputs(ref, 1, nt, nv)
    with FlopCounterMode(display=False) as counter:  # no_grad trips its module tracker here
        dec(ids, memory)
    table = {k.split("blocks.0.")[-1]: sum(v.values()) for k, v in counter.get_flop_counts().items()
             if "blocks.0." in k}
    if qformer:
        return table["cross_attn"], table["cross_attn.read"] - table["cross_attn.read.kv_proj"]
    return table["cross_attn"], table["cross_attn"] - table["cross_attn.kv_proj"]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    base, swapped = build(ref, 10, 197, False), build(ref, 10, 197, True)
    ids, memory = inputs(ref, 2, 10, 197)
    with torch.no_grad():
        logits = swapped(ids, memory)
        image_move = (swapped(ids, ref.synth_memory(2, 197, 256, seed=9)) - logits).abs().max()
        cache_gap = (swapped(ids, memory, use_cache=True) - logits).abs().max().item()
        cache = [base.build_kv_cache(memory)[0][0].shape, swapped.build_kv_cache(memory)[0][0].shape]
    count = [sum(p.numel() for p in m.parameters()) for m in (base, swapped)]
    shapes = [(10, 197), (64, 576)]
    return {"shape": tuple(logits.shape), "image_move": image_move.item(), "params": count,
            "cache_gap": cache_gap, "cache": [tuple(c) for c in cache],
            "flops": {s: [cross_flops(ref, *s, q) for q in (False, True)] for s in shapes}}


def verify(result):
    r, f = result, result["flops"]
    small, big = f[(10, 197)], f[(64, 576)]
    even = QUERIES * (2 * 256 + 576) / (576 - QUERIES)
    return [
        practice.Check(
            "ANSWER: the swap works, and the text reads 32 pooled tokens instead of 197",
            r["shape"] == (2, 10, 1024) and r["image_move"] > 0.01 and r["cache_gap"] < 1e-5
            and r["cache"] == [(2, 8, 197, 32), (2, 8, 32, 32)]
            and r["params"][1] - r["params"][0] == 1_085_440,
            f"logits {r['shape']}, move {r['image_move']:.4f} with the image; cached vs uncached "
            f"{r['cache_gap']:.1e}; per-layer K cache {r['cache'][0]} -> {r['cache'][1]}; params "
            f"{r['params'][0]:,} -> {r['params'][1]:,}",
        ),
        practice.Check(
            "FINDING: the pool pays off only per decode step, not per forward pass",
            small == [(56_281_088, 4_638_720), (77_824_000, 2_949_120)]
            and big == [(205_520_896, 54_525_952), (205_520_896, 18_874_368)] and even == 64,
            f"layer-0 (cross, text-side) FLOPs lesson vs Q-Former at Nt=10,Nv=197 {small}; at "
            f"Nt=64,Nv=576 {big}; break-even Nt = Q(2D+Nv)/(Nv-Q) = {even:g}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
