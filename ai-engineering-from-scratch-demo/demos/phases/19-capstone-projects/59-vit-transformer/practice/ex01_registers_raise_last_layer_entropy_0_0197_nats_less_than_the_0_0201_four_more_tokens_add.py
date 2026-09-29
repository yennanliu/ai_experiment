"""Exercise 1 — registers raise last-layer entropy 0.0197 nats, less than the 0.0201 four more tokens add.

    Add register tokens (4 learned vectors prepended after CLS) and rerun. Compare attention map smoothness via the entropy of the softmax distribution on the last layer.

Reading of the exercise: "rerun" is the lesson's demo run -- the seed-0
ViT-Base `VisionEncoder` (untrained, as the demo is) on the seed-0 fixture
image. Four register vectors, initialised like the CLS token
(trunc-normal, std 0.02, seed 1), are inserted at positions 1-4 after the
front end has added its position table, as DINOv2 does, so they carry no
position. Both runs go through the same `ViT` stack with `store_attn=True`;
smoothness is the Shannon entropy (nats) of each softmax row of the last
block, averaged over the 12 heads and all query rows, and also divided by
ln(sequence length), its maximum.

**ANSWER: at the demo's untrained weights the registers change nothing
visible.** The last-layer entropy goes from 5.2795 to 5.2992 nats. The
maximum rises by ln(201/197) = 0.0201 because the sequence is 4 tokens
longer, so all of the increase is the longer row: normalised entropy is
0.99930 without registers and 0.99923 with them. Both maps are already
flat; the four registers take 2.04% of the attention mass, against 1.99%
for any 4 uniform tokens. The smoother maps reported for registers are a
property of trained ViTs, whose high-norm artifact tokens the registers
absorb (Darcet et al., "Vision Transformers Need Registers", 2023,
https://arxiv.org/abs/2309.16588, read 2026-09-29). An untrained stack has
no artifacts to absorb. The lesson credits registers to the DINOv2 paper,
but they come from this later paper by the same group.

**FINDING: the "stabilised" CLS norm is sqrt(768) for any input.** The demo
trace rises steadily, 0.53 after the front end to 35.80 after block 12,
without levelling off, and then the final LayerNorm (weight 1, bias 0)
sets it to 27.713 = sqrt(768). A zero image gives the same 27.713.

Structure: `entropy()` scores attention rows; `solve()` runs the demo
encoder with and without registers and traces the CLS norm.
"""

from __future__ import annotations

import math

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "59-vit-transformer"
N_REG = 4


def entropy(attn):
    """Mean Shannon entropy (nats) of the softmax rows, over heads and queries."""
    return -(attn * attn.clamp_min(1e-30).log()).sum(-1).mean().item()


def last_attn(enc, tokens):
    enc.vit(tokens, store_attn=True)
    return enc.vit.blocks[-1].attn.last_attn


def cls_trace(enc, img):
    x, norms = enc.front(img), []
    norms.append(x[0, 0].norm().item())
    for block in enc.vit.blocks:
        x = block(x)
        norms.append(x[0, 0].norm().item())
    return norms, enc.vit.norm(x)[0, 0].norm().item()


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.manual_seed(0)
    enc = ref.VisionEncoder(ref.ViTConfig()).eval()
    img = ref.synthesize_image(seed=0)
    torch.manual_seed(1)
    registers = torch.nn.init.trunc_normal_(torch.zeros(1, N_REG, enc.cfg.hidden), std=0.02)
    with torch.no_grad():
        tokens = enc.front(img)
        plain = last_attn(enc, tokens)
        with_reg = last_attn(enc, torch.cat([tokens[:, :1], registers, tokens[:, 1:]], dim=1))
        trace, final = cls_trace(enc, img)
        zero_cls = enc(torch.zeros_like(img))[1].norm().item()
    n0, n1 = plain.shape[-1], with_reg.shape[-1]
    return {"n": (n0, n1), "h0": entropy(plain), "h1": entropy(with_reg),
            "norm0": entropy(plain) / math.log(n0), "norm1": entropy(with_reg) / math.log(n1),
            "reg_mass": with_reg[..., 1:1 + N_REG].sum(-1).mean().item(),
            "trace": trace, "rising": all(b > a for a, b in zip(trace, trace[1:])),
            "final": final, "zero_cls": zero_cls, "hidden": enc.cfg.hidden}


def verify(result):
    r = result
    shift = math.log(r["n"][1] / r["n"][0])
    t, rising = r["trace"], r["rising"]
    root = math.sqrt(r["hidden"])
    return [
        practice.Check(
            "ANSWER: at the demo's untrained weights the registers change nothing visible",
            all([r["n"] == (197, 201), abs(r["h0"] - 5.2795) < 1e-3, abs(r["h1"] - 5.2992) < 1e-3,
                 abs((r["h1"] - r["h0"]) - shift) < 1e-3, min(r["norm0"], r["norm1"]) > 0.999,
                 abs(r["norm0"] - r["norm1"]) < 1e-4, abs(r["reg_mass"] - N_REG / 201) < 1e-3]),
            f"entropy {r['h0']:.4f} -> {r['h1']:.4f} nats (ln(201/197) = {shift:.4f}); normalised "
            f"{r['norm0']:.5f} -> {r['norm1']:.5f}; register mass {r['reg_mass']:.4f} vs uniform "
            f"{N_REG / 201:.4f}",
        ),
        practice.Check(
            "FINDING: the 'stabilised' CLS norm is sqrt(768) for any input",
            all([rising, abs(t[0] - 0.534) < 0.01, abs(t[-1] - 35.80) < 0.05,
                 abs(r["final"] - root) < 1e-3, abs(r["zero_cls"] - root) < 1e-3]),
            f"CLS norm {t[0]:.2f} -> {t[-1]:.2f} over 12 blocks, rising every block: {rising}; "
            f"final LN {r['final']:.3f}, zero image {r['zero_cls']:.3f}, sqrt(768) = {root:.3f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
