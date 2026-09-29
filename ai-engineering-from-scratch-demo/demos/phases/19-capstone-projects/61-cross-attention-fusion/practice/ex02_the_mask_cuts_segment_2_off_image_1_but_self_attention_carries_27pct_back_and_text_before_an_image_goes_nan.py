"""Exercise 2 — the mask cuts segment 2 off image 1, but self-attention carries 27% back, and text before an image goes NaN.

    Implement interleaved attention where the same decoder consumes multiple images plus multiple text segments. Build the per-sample cross-attention mask that prevents text segment 2 from attending to image 1.

Reading of the exercise: each sample is `<image 1> text segment 1 <image 2>
text segment 2`. The two images are concatenated into one memory of 2 x 197
tokens. The rule is Flamingo's: a text token attends only to the image that
came just before it. The mask is `(B, Nt, Nv)` boolean and is built per
sample, so the two samples split their 12 text tokens at different places
(6+6 and 4+8). The lesson's `CrossAttention.forward` takes no mask, so
`masked_cross` reuses that module's own `q_proj`, `project_memory` and `out`
and adds a `masked_fill` before the softmax. It is installed on every block
of a default seeded `VisionLanguageDecoder` in float64. Leakage is measured by
replacing image 1 with fresh noise and checking how much segment 2 moves.

**ANSWER: the mask works where it applies.** With an all-True mask,
`masked_cross` matches the reference `CrossAttention` to 0.0. With the
interleaved mask, a single cross-attention layer's output for segment 2
changes by exactly 0.0 when image 1 changes. The unmasked reference layer
moves by 0.054.

**FINDING: the mask alone does not stop segment 2 from depending on image
1.** Across the full 4-layer decoder, segment 2's logits still move by 0.026
when image 1 changes, 27% of the 0.095 that segment 1 moves. Segment 1's text tokens read image 1, and from block 2
on, segment 2 reads segment 1 through causal self-attention. Flamingo only
promises that "the model only directly attends to a single image at a time"
(https://arxiv.org/html/2204.14198, read 2026-09-29). To cut the path completely, the
self-attention mask must also be block-diagonal by segment. Patching
`causal_mask` to do that brings the leak to 0.0, but only for batch 1,
because `CausalSelfAttention` rejects any mask that is not `(n, n)`: a per-sample
`(2, 12, 12)` mask raises `ValueError`.

**FINDING: text before the first image turns the whole sample into NaN.** A
token that has no preceding image gets a mask row that is all False, and its
softmax is NaN. By the last layer, self-attention has carried that NaN to
every later position: 12 of 12 positions come out NaN, not just the first.
Zeroing empty rows, which gives that token no image signal, fixes it. The
doc's own block snippet passes a `cross_mask`, but the shipped
`DecoderBlock.forward` has no such parameter.

Structure: `masked_cross()` is the masked forward; `decoder_run()` installs
it on every block; `with_self_mask()` adds the segment-diagonal self mask.
"""

from __future__ import annotations

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "61-cross-attention-fusion"
NV, NT = 197, 12
SEGS = [[0] * 6 + [1] * 6, [0] * 4 + [1] * 8]  # image before each text token, per sample


def masked_cross(ca, x, memory, mask, zero_empty=True):
    b, nt, d = x.shape
    q = ca.q_proj(x).reshape(b, nt, ca.cfg.heads, ca.cfg.head_dim).transpose(1, 2)
    k, v = ca.project_memory(memory)
    scores = ((q @ k.transpose(-2, -1)) * ca.scale).masked_fill(~mask[:, None], float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    if zero_empty:
        attn = attn.masked_fill(~mask.any(-1)[:, None, :, None], 0.0)
    return ca.drop(ca.out((attn @ v).transpose(1, 2).reshape(b, nt, d)))


@torch.no_grad()
def decoder_run(dec, ids, memory, mask, zero_empty=True):
    for block in dec.blocks:
        ca = block.cross_attn
        ca.forward = lambda x, mem, kv_cache=None, ca=ca: masked_cross(ca, x, mem, mask, zero_empty)
    try:
        return dec(ids, memory)
    finally:
        for block in dec.blocks:
            del block.cross_attn.forward


def moved(a, b, seg=1):
    """Largest change, over the text tokens of segment `seg`, between two runs."""
    return (a - b).abs()[torch.tensor(SEGS[: len(a)]) == seg].max().item()


@torch.no_grad()
def one_layer(ca, x, memory, other, mask):
    gap = (masked_cross(ca, x, memory, torch.ones_like(mask)) - ca(x, memory)).abs().max()
    masked = moved(masked_cross(ca, x, memory, mask), masked_cross(ca, x, other, mask))
    return gap.item(), masked, moved(ca(x, memory), ca(x, other))


def with_self_mask(ref, dec, ids, memories, mask):
    """causal_mask patched block-diagonal by segment: batch 1 works, a per-sample mask cannot."""
    seg = torch.tensor(SEGS)
    per_sample = torch.tril(torch.ones(NT, NT, dtype=torch.bool)) & (seg[:, :, None] == seg[:, None])
    saved, ref.causal_mask, error = ref.causal_mask, lambda n: per_sample[0], "accepted"
    try:
        one = [decoder_run(dec, ids[:1], m[:1], mask[:1]) for m in memories]
        dec.blocks[0].self_attn(dec.tok_emb(ids), mask=per_sample)
    except ValueError as exc:
        error = str(exc)
    finally:
        ref.causal_mask = saved
    return moved(*one), error


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(0)
        dec = ref.VisionLanguageDecoder(ref.DecoderConfig()).eval()
        gen = torch.Generator().manual_seed(1)
        memory = torch.randn(2, 2 * NV, 256, generator=gen)
        other = torch.cat([torch.randn(2, NV, 256, generator=gen), memory[:, NV:]], 1)
        ids = torch.randint(0, 1024, (2, NT), generator=gen)
        mask = torch.tensor(SEGS)[:, :, None] == torch.tensor([0] * NV + [1] * NV)  # (B, Nt, Nv)
        prefix = mask.clone()
        prefix[0, 0] = False  # sample 0's first token now comes before any image
        runs = [decoder_run(dec, ids, m, mask) for m in (memory, other)]
        nan = decoder_run(dec, ids, memory, prefix, zero_empty=False)[0].isnan().any(-1)
        x = dec.tok_emb(ids).detach()
        out = {"layer": one_layer(dec.blocks[0].cross_attn, x, memory, other, mask),
               "full": (moved(*runs), moved(*runs, seg=0)),
               "self": with_self_mask(ref, dec, ids, (memory, other), mask),
               "nan": (int(nan.sum()), decoder_run(dec, ids, memory, prefix).isnan().any().item())}
    finally:
        torch.set_default_dtype(torch.float32)
    out["cross_mask"] = ("cross_mask" in parity.doc_text(PHASE, LESSON),
                         "cross_mask" in ref.DecoderBlock.forward.__code__.co_varnames)
    return out


def verify(result):
    (gap, masked, unmasked), (seg2, seg1), (blocked, error) = result["layer"], result["full"], result["self"]
    return [
        practice.Check(
            "ANSWER: the mask works where it applies",
            all([gap == 0.0, masked == 0.0, 0.05 < unmasked < 0.06]),
            f"all-True mask vs reference {gap}; one layer, segment 2 moves {masked} masked vs "
            f"{unmasked:.4f} unmasked when image 1 changes",
        ),
        practice.Check(
            "FINDING: the mask alone does not stop segment 2 from depending on image 1",
            all([0.02 < seg2 < 0.03, round(seg2 / seg1, 2) == 0.27, blocked == 0.0,
                 "does not match" in error]),
            f"4-layer logits of segment 2 move {seg2:.4f} (segment 1: {seg1:.4f}); with a "
            f"segment-diagonal self mask (batch 1) {blocked}; per-sample (2,12,12) mask: {error}",
        ),
        practice.Check(
            "FINDING: text before the first image turns the whole sample into NaN",
            result["nan"] == (NT, False) and result["cross_mask"] == (True, False),
            f"(NaN positions, any NaN once empty rows are zeroed) {result['nan']}; (doc, code) name cross_mask {result['cross_mask']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
