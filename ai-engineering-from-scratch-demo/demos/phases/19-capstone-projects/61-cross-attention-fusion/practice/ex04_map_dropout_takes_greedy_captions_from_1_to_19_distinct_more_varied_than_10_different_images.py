"""Exercise 4 — map dropout takes greedy captions from 1 to 19 distinct, more varied than 10 different images.

    Add a query-side dropout on the cross-attention map and measure caption diversity on the demo (caption sample variance increases with dropout in the cross map).

Reading of the exercise: "the demo" is `main.py`'s own setup: the default
`DecoderConfig`, `torch.manual_seed(0)`, an untrained decoder in eval mode,
and `synth_memory(seed=1)` as the image, run in float64. The demo never
generates text, so a caption is 9 tokens decoded after token 0, and 20
captions are sampled per setting. "Query-side dropout on the cross-attention
map" means dropout on the softmax weights `(B, H, Nt, Nv)`: each text query
loses a random subset of the image tokens it reads, with inverted scaling. It
stays on at sampling time, and `drop_cross` wraps each block's reference
projections. Diversity is the number of distinct captions out of 20 and the
mean pairwise fraction of tokens that differ.

**ANSWER: under greedy decoding, diversity rises with dropout.**

| p | distinct / 20 | token disagreement |
|---:|---:|---:|
| 0.0 | 1 | 0.000 |
| 0.1 | 2 | 0.100 |
| 0.3 | 4 | 0.464 |
| 0.5 | 5 | 0.577 |
| 0.9 | 19 | 0.792 |

Under temperature-1 sampling the claim cannot be seen. Already at p = 0, 20 of
20 captions are distinct and 99.8% of tokens differ, so there is no room left
for dropout to add diversity. The untrained next-token distribution has
entropy 6.77 nats against a maximum of ln 1024 = 6.93, and its top token has
probability 0.0045.

**FINDING: at high p, dropout noise outweighs the image.** Without dropout,
ten different `synth_memory` images (seeds 1-10) give 5 distinct greedy
captions with 0.652 token disagreement. One image at p = 0.5 already varies
almost as much (0.577), and at p = 0.9 it varies more (0.792) than ten
different images do. The added "diversity" comes from the dropout mask, not
from the picture.

**FINDING: the shipped `dropout` is not map dropout, and the demo turns it
off.** `CrossAttention` applies `cfg.dropout` after the output projection,
not to the attention map. The default is 0.0, the demo calls `.eval()`, and
`main.py` has no function that generates text.

Structure: `drop_cross()` is the dropout-on-map forward; `captions()` runs
greedy or temperature decoding with it installed; `diversity()` scores 20
captions.
"""

from __future__ import annotations

import inspect
import itertools

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "61-cross-attention-fusion"
PS = (0.0, 0.1, 0.3, 0.5, 0.9)


def drop_cross(ca, p, gen):
    def forward(x, memory, kv_cache=None):
        b, nt, d = x.shape
        h, hd = ca.cfg.heads, ca.cfg.head_dim
        q = ca.q_proj(x).reshape(b, nt, h, hd).transpose(1, 2)
        k, v = ca.project_memory(memory)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * ca.scale, dim=-1)
        if p:
            attn = attn * (torch.rand(attn.shape, generator=gen) >= p) / (1 - p)
        return ca.drop(ca.out((attn @ v).transpose(1, 2).reshape(b, nt, d)))

    return forward


def captions(dec, memory, p=0.0, temp=0.0, length=10):
    gen = torch.Generator().manual_seed(0)
    for block in dec.blocks:
        block.cross_attn.forward = drop_cross(block.cross_attn, p, gen)
    ids = torch.zeros(memory.shape[0], 1, dtype=torch.long)
    try:
        with torch.no_grad():
            for _ in range(length - 1):
                logits = dec(ids, memory)[:, -1]
                if temp:
                    nxt = torch.multinomial(torch.softmax(logits / temp, -1), 1, generator=gen)
                else:
                    nxt = logits.argmax(-1, keepdim=True)
                ids = torch.cat([ids, nxt], 1)
    finally:
        for block in dec.blocks:
            del block.cross_attn.forward
    return [tuple(row) for row in ids[:, 1:].tolist()]


def diversity(caps):
    pairs = list(itertools.combinations(caps, 2))
    differ = sum(a != b for x, y in pairs for a, b in zip(x, y))
    return len(set(caps)), round(differ / (len(pairs) * len(caps[0])), 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(0)
        dec = ref.VisionLanguageDecoder(ref.DecoderConfig()).eval()
        image = ref.synth_memory(1, 197, 256, seed=1).expand(20, -1, -1)
        greedy = {p: diversity(captions(dec, image, p)) for p in PS}
        sampled = {p: diversity(captions(dec, image, p, temp=1.0)) for p in (0.0, 0.5)}
        many = torch.cat([ref.synth_memory(1, 197, 256, seed=s) for s in range(1, 11)])
        with torch.no_grad():
            probs = torch.softmax(dec(torch.zeros(1, 1, dtype=torch.long), image[:1])[0, -1], -1)
        entropy = -(probs * probs.log()).sum().item()
        per_image = diversity(captions(dec, many))
    finally:
        torch.set_default_dtype(torch.float32)
    forward = inspect.getsource(ref.CrossAttention.forward)
    return {"greedy": greedy, "sampled": sampled, "per_image": per_image,
            "entropy": round(entropy, 2), "top": round(probs.max().item(), 4),
            "drop_on_out": "self.drop(self.out(out))" in forward and "self.drop(attn" not in forward,
            "default": ref.DecoderConfig().dropout, "has_generate": "multinomial" in inspect.getsource(ref)
            or "argmax" in inspect.getsource(ref)}


def verify(result):
    g, s = result["greedy"], result["sampled"]
    return [
        practice.Check(
            "ANSWER: under greedy decoding, diversity rises with dropout",
            list(g.values()) == [(1, 0.0), (2, 0.1), (4, 0.464), (5, 0.577), (19, 0.792)]
            and s == {0.0: (20, 0.998), 0.5: (20, 1.0)}
            and (result["entropy"], result["top"]) == (6.77, 0.0045),
            f"greedy (distinct, disagreement) by p: {g}; temperature 1: {s}; next-token entropy "
            f"{result['entropy']} nats (max 6.93), top prob {result['top']}",
        ),
        practice.Check(
            "FINDING: at high p, dropout noise outweighs the image",
            result["per_image"] == (5, 0.652) and g[0.5][1] < 0.652 < g[0.9][1],
            f"10 images at p = 0 (distinct, disagreement) {result['per_image']}; one image at p = 0.5 "
            f"{g[0.5]}, at p = 0.9 {g[0.9]}",
        ),
        practice.Check(
            "FINDING: the shipped dropout is not map dropout, and the demo turns it off",
            (result["drop_on_out"], result["default"], result["has_generate"]) == (True, 0.0, False),
            f"dropout applied to the output projection only: {result['drop_on_out']}; default "
            f"{result['default']}; main.py samples or argmaxes tokens: {result['has_generate']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
