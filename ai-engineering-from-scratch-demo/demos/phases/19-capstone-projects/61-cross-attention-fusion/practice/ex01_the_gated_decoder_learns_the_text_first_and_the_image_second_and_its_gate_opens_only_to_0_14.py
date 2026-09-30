"""Exercise 1 — the gated decoder learns the text first and the image second, and its gate opens only to 0.14.

    Add a learned tanh gate to the cross-attention residual (the Flamingo trick) and verify training converges from a near-zero initial gate. The gate starts at 0; the model recovers text-only behavior before mixing the image stream in.

Reading of the exercise: the gate is a per-layer scalar `alpha`, initialised
to exactly 0, and it multiplies each lesson `CrossAttention` output by
`tanh(alpha)` before the lesson `DecoderBlock` adds it to the residual. It is
wrapped around the reference module, not a fork of it. `main.py` has no
training loop, so the check uses a small synthetic task in float64: a 2-layer
decoder (hidden 64) reads BOS plus six counting tokens (s, s+1, ...), which
text alone predicts, and must then emit the class of the image (4 noisy
prototypes, 8 tokens each), which only the image predicts. Adam at 3e-3,
batch 64, 200 steps, seeded. The same run without the gate is the control.

**ANSWER: it converges, and it learns text-only first.** At step 0 the gated
decoder's logits are identical for two different images (max difference 0.0),
and every cross-attention weight gets a gradient of exactly 0.0. Only the two
alphas move. Both runs reach a text loss of 0.002 and an image loss of 0.001
by step 200. The order differs:

| run | text loss < 0.1 | image loss < 0.1 | image loss when text converges |
|---|---:|---:|---:|
| gated | step 19 | step 34 | 1.256 (ln 4 = 1.386 is chance) |
| ungated | step 21 | step 15 | 0.040 |

**FINDING: the gate opens only to |tanh(alpha)| = 0.14 and stays there.** It
is already 0.139 by step 50, and the image loss still falls to 0.001, so the image
path works through a gate that is 86% closed. A gate value near 1 is not what
"mixing the image stream in" looks like here.

**FINDING: the doc gets the reason for the gate backwards.** It says
Flamingo added the gate "at training-time stability cost". Flamingo
(arXiv:2204.14198, https://arxiv.org/html/2204.14198, read 2026-09-29) says
that with alpha initialised to 0 "the model output matches that of the
pretrained LM, improving training stability and final performance".

Structure: `Gated` wraps a reference `CrossAttention`; `batch()` makes the
synthetic task; `train()` runs one seeded run and records both losses.
"""

from __future__ import annotations

from harness import parity, practice

try:
    import torch
    from torch import nn
    from torch.nn.functional import cross_entropy
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "61-cross-attention-fusion"
VOCAB, CLASSES, BOS = 16, 4, 15


class Gated(nn.Module):
    """tanh(alpha) * cross_attn(x, memory), alpha a learnable scalar starting at 0."""

    def __init__(self, inner):
        super().__init__()
        self.inner, self.alpha = inner, nn.Parameter(torch.zeros(()))

    def forward(self, x, memory, kv_cache=None):
        return torch.tanh(self.alpha) * self.inner(x, memory, kv_cache=kv_cache)


def build(ref, gated):
    cfg = ref.DecoderConfig(hidden=64, heads=4, depth=2, mlp_ratio=2.0, text_vocab=VOCAB,
                            max_text_len=8, vision_dim=32, vision_tokens=8)
    torch.manual_seed(0)
    dec = ref.VisionLanguageDecoder(cfg)
    for block in dec.blocks if gated else []:
        block.cross_attn = Gated(block.cross_attn)
    return dec


def batch(protos, n, gen):
    cls = torch.randint(0, CLASSES, (n,), generator=gen)
    memory = protos[cls] + 0.5 * torch.randn(n, 8, 32, generator=gen)
    count = (torch.randint(0, 12, (n, 1), generator=gen) + torch.arange(7)) % 12
    ids = torch.cat([torch.full((n, 1), BOS), count[:, :6]], 1)  # BOS, s, ..., s+5
    return ids, memory, torch.cat([count[:, :6], (12 + cls)[:, None]], 1)  # s, ..., s+5, class


def losses(dec, ids, memory, target):
    logits = dec(ids, memory)
    text = cross_entropy(logits[:, 1:6].reshape(-1, VOCAB), target[:, 1:6].reshape(-1))
    return text, cross_entropy(logits[:, 6], target[:, 6])


def train(ref, protos, gated, steps=200):
    dec, gen = build(ref, gated), torch.Generator().manual_seed(2)
    opt, hist = torch.optim.Adam(dec.parameters(), lr=3e-3), []
    for _ in range(steps + 1):
        text, image = losses(dec, *batch(protos, 64, gen))
        opt.zero_grad()
        (text + image).backward()
        opt.step()
        hist.append((text.item(), image.item()))
    t_ok, i_ok = (next(k for k, h in enumerate(hist) if h[j] < 0.1) for j in (0, 1))
    gates = [torch.tanh(b.cross_attn.alpha).item() for b in dec.blocks if gated]
    return {"at": (t_ok, i_ok), "image_then": round(hist[t_ok][1], 3), "final": max(hist[-1]),
            "gates": [round(g, 3) for g in gates]}


def at_init(ref, protos):
    """Logit gap across two images, and max |grad| on cross-attention vs the alphas, at step 0."""
    dec = build(ref, gated=True)
    ids, memory, target = batch(protos, 64, torch.Generator().manual_seed(2))
    with torch.no_grad():
        gap = (dec(ids, memory) - dec(ids, memory.flip(0))).abs().max().item()
    sum(losses(dec, ids, memory, target)).backward()
    inner = [p.grad.abs().max().item() for b in dec.blocks for p in b.cross_attn.inner.parameters()]
    return gap, max(inner), [round(b.cross_attn.alpha.grad.item(), 4) for b in dec.blocks]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_default_dtype(torch.float64)
    try:
        protos = torch.randn(CLASSES, 8, 32, generator=torch.Generator().manual_seed(1))
        out = {"init": at_init(ref, protos), "gated": train(ref, protos, True),
               "plain": train(ref, protos, False)}
    finally:
        torch.set_default_dtype(torch.float32)
    out["doc"] = "at training-time stability cost" in parity.doc_text(PHASE, LESSON)
    return out


def verify(result):
    g, p, (gap, xattn, alpha) = result["gated"], result["plain"], result["init"]
    return [
        practice.Check(
            "ANSWER: it converges, and it learns text-only first",
            all([gap == 0.0, xattn == 0.0, 0 not in alpha, g["final"] < 0.01, p["final"] < 0.01,
                 (g["at"], p["at"]) == ((19, 34), (21, 15)), g["image_then"] > 1.0,
                 p["image_then"] < 0.1]),
            f"init logit gap across images {gap}, x-attn grad {xattn}, alpha grads {alpha}; "
            f"(text, image) < 0.1 at steps gated {g['at']} (image loss then {g['image_then']}), "
            f"ungated {p['at']} ({p['image_then']}); final max loss {g['final']:.3f} / {p['final']:.3f}",
        ),
        practice.Check(
            "FINDING: the gate opens only to |tanh(alpha)| = 0.14 and stays there",
            [abs(v) for v in g["gates"]] == [0.143, 0.145],
            f"tanh(alpha) per layer after 200 steps {g['gates']}",
        ),
        practice.Check(
            "FINDING: the doc gets the reason for the gate backwards",
            result["doc"],
            f"doc says the gate comes 'at training-time stability cost': {result['doc']}; "
            "Flamingo says it improves training stability",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
