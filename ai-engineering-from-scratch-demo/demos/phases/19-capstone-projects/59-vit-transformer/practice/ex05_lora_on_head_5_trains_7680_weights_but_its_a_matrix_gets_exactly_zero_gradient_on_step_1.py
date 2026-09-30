"""Exercise 5 — LoRA on head 5 trains 7,680 weights, but its A matrix gets exactly zero gradient on step 1.

    Replace one attention head's q-k-v projections with a low-rank LoRA adapter, freeze the rest, and verify the gradient only flows where you expect.

Reading of the exercise: the model is the lesson's full ViT-Base
`VisionEncoder` (seed 0). Every parameter is frozen, then block 0's fused
`qkv` linear is wrapped so that head 5's 192 output rows (64 of q, of k and
of v) get a rank-8 update `x @ A.T @ B.T`: A (8 x 768) with the usual
Kaiming init, B (192 x 8) at zero, as LoRA initialises it. The rows come from
the lesson's own layout, `reshape(b, n, 3, heads, head_dim)`: row
`p * 768 + head * 64 + i` is projection p (q, k, v) of that head. Two Adam
steps (lr 1e-3) run on the CLS output against a fixed random target.

**ANSWER: gradient reaches only the adapter, and only head 5 changes.**
After backward, only A and B have a `.grad`; all 149 frozen tensors have
`None`. The adapter trains 7,680 weights of 85,647,360, 0.009%. After the
two steps, the update to block 0's `qkv` output is non-zero in exactly 3 of
the 36 (projection, head) slices, q, k and v of head 5, and exactly 0 in the
other 33.

**FINDING: on the first step A gets exactly zero gradient.** With B = 0 the
gradient of A is `B.T @ (...)`, so A's gradient norm is 0.0 on step 1 while
B's is not. A starts learning on step 2, after B has moved. "Gradient flows
to the adapter" is true for B only at step 1.

**FINDING: slicing "head 5's rows" as one contiguous block misses the
head.** Rows 5 * 192 to 6 * 192 look like head 5's q-k-v in a
`(heads, 3, head_dim)` layout. In the lesson's `(3, heads, head_dim)`
layout they are k of heads 3, 4 and 5. An adapter there changes 3 slices,
none of them head 5's q or v.

Structure: `head_rows()` maps a head to its qkv rows; `lora()` wraps the
linear; `changed()` lists the slices an update touches; `solve()` trains.
"""

from __future__ import annotations

import math

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "59-vit-transformer"
HEAD, RANK = 5, 8


def head_rows(head, d=768, hd=64):
    return torch.cat([torch.arange(p * d + head * hd, p * d + (head + 1) * hd) for p in range(3)])


def lora(base, rows, rank):
    """Wrap a frozen linear: out + (x @ A.T @ B.T) scattered into `rows`."""
    wrapper = torch.nn.Module()
    wrapper.base, wrapper.rows = base, rows
    wrapper.A = torch.nn.Parameter(torch.empty(rank, base.in_features))
    torch.nn.init.kaiming_uniform_(wrapper.A, a=math.sqrt(5))
    wrapper.B = torch.nn.Parameter(torch.zeros(len(rows), rank))

    def forward(x):
        out = base(x)
        delta = torch.zeros_like(out)
        delta[..., rows] = x @ wrapper.A.T @ wrapper.B.T
        return out + delta

    wrapper.forward = forward
    return wrapper


def changed(wrapper, x, heads=12, hd=64):
    """(projection, head) slices where the wrapped output differs from the base."""
    with torch.no_grad():
        diff = (wrapper(x) - wrapper.base(x)).abs().amax(dim=(0, 1)).reshape(3, heads, hd).amax(-1)
    return [("qkv"[p], h) for p in range(3) for h in range(heads) if diff[p, h] > 0]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.manual_seed(0)
    enc = ref.VisionEncoder(ref.ViTConfig())
    enc.requires_grad_(False)
    attn = enc.vit.blocks[0].attn
    attn.qkv = adapter = lora(attn.qkv, head_rows(HEAD), RANK)
    g = torch.Generator().manual_seed(1)
    img, target = torch.rand(2, 3, 224, 224, generator=g), torch.randn(2, 768, generator=g)
    opt = torch.optim.Adam([adapter.A, adapter.B], lr=1e-3)
    grads = []
    for _ in range(2):
        opt.zero_grad()
        (enc(img)[1] - target).pow(2).mean().backward()
        grads.append((adapter.A.grad.norm().item(), adapter.B.grad.norm().item()))
        opt.step()
    frozen = [p for p in enc.parameters() if p is not adapter.A and p is not adapter.B]
    x = enc.vit.blocks[0].ln1(enc.front(img))
    naive = lora(adapter.base, torch.arange(HEAD * 192, (HEAD + 1) * 192), RANK)
    torch.nn.init.ones_(naive.B)
    return {"with_grad": sorted(n for n, p in enc.named_parameters() if p.grad is not None),
            "frozen": len(frozen), "frozen_grads": sum(p.grad is not None for p in frozen),
            "trainable": adapter.A.numel() + adapter.B.numel(),
            "total": sum(p.numel() for p in enc.parameters()) - adapter.A.numel() - adapter.B.numel(),
            "grads": grads, "changed": changed(adapter, x), "naive": changed(naive, x)}


def verify(result):
    r = result
    (a1, b1), (a2, b2) = r["grads"]
    return [
        practice.Check(
            "ANSWER: gradient reaches only the adapter, and only head 5 changes",
            r["with_grad"] == ["vit.blocks.0.attn.qkv.A", "vit.blocks.0.attn.qkv.B"]
            and r["frozen_grads"] == 0 and r["frozen"] == 149
            and (r["trainable"], r["total"]) == (7_680, 85_647_360)
            and r["changed"] == [("q", HEAD), ("k", HEAD), ("v", HEAD)],
            f"tensors with .grad: {r['with_grad']}; frozen with .grad {r['frozen_grads']}/{r['frozen']}; "
            f"trainable {r['trainable']:,} of {r['total']:,}; changed slices {r['changed']}",
        ),
        practice.Check(
            "FINDING: on the first step A gets exactly zero gradient",
            a1 == 0.0 and b1 > 0 and a2 > 0 and b2 > 0,
            f"|grad A|, |grad B| step 1: {a1}, {b1:.2e}; step 2: {a2:.2e}, {b2:.2e}",
        ),
        practice.Check(
            "FINDING: slicing 'head 5's rows' as one contiguous block misses the head",
            r["naive"] == [("k", 3), ("k", 4), ("k", 5)],
            f"rows {HEAD * 192}-{(HEAD + 1) * 192 - 1} change slices {r['naive']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
