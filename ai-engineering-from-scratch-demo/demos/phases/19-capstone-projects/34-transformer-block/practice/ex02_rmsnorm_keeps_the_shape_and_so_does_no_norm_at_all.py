"""Exercise 2 — RMSNorm keeps the shape, and so does no norm at all.

    Replace `nn.LayerNorm` with a hand rolled RMSNorm and verify the output shape is unchanged.

Reading of the exercise: the lesson's code has no `nn.LayerNorm` to replace.
It uses its own hand-rolled `LayerNorm` class, so that class is the one
swapped. The module-level name `LayerNorm` is patched to a hand-rolled RMSNorm
while the demo's 6-block pre-LN stack (d_model 192, 6 heads, tokens (2, 32))
is built, and the lesson's `LayerNorm` is put back afterwards. Both stacks are
built from seed 0. The norms draw no random numbers, so every linear and
embedding weight is identical in the two stacks.

**ANSWER: the shape is unchanged, (2, 32, 192) both ways.** RMSNorm divides
each token by `sqrt(mean(x^2) + eps)` and has one learnable scale and no
shift. It drops 13 x 192 = 2,496 parameters from the stack (2,694,144 ->
2,691,648). It matches `torch.nn.RMSNorm(192, eps=1e-5)` to within 1e-6. The
outputs differ from the LayerNorm stack's, as they should: each output token
has RMS 1.000 but a mean that is not zero.

**FINDING: the exercise names a module the code never uses.**
`nn.LayerNorm` appears once in `main.py`, inside a docstring ("Equivalent to
nn.LayerNorm(d_model)"). That equivalence does hold: on a (2, 16, 64) input
the hand-rolled class matches `nn.LayerNorm` to within 1e-6.

**FINDING: a shape check cannot tell a norm from no norm.** Swap
`LayerNorm` for `nn.Identity` and the stack still returns (2, 32, 192). What
separates them is the statistic the norm sets. The mean per-token RMS of the
final output is 1.000 with RMSNorm and with LayerNorm, and 1.25 with no norm.
Separately, the per-token mean is 0 under LayerNorm and averages 0.057 in
absolute value under RMSNorm.

Structure: `rms_norm_class()` returns the replacement module; `stack()`
builds the demo stack with the lesson's `LayerNorm` name bound to a given
factory; `solve()` runs the three variants on the same tokens.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "34-transformer-block"
D_MODEL, EPS = 192, 1e-5


def rms_norm_class():
    class RMSNorm(torch.nn.Module):
        """x / sqrt(mean(x^2) + eps) * scale, over the embedding axis. No shift."""

        def __init__(self, d_model, eps=EPS):
            super().__init__()
            self.eps, self.scale = eps, torch.nn.Parameter(torch.ones(d_model))

        def forward(self, x):
            return self.scale * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    return RMSNorm


def stack(ref, norm):
    """The demo's 6-block pre-LN stack at seed 0, with `LayerNorm` bound to `norm`."""
    saved, ref.LayerNorm = ref.LayerNorm, norm
    try:
        torch.manual_seed(0)
        cfg = ref.BlockConfig(d_model=D_MODEL, num_heads=6, context_length=64,
                              attn_dropout=0.0, residual_dropout=0.0, pre_ln=True)
        return ref.BlockStack(cfg, depth=6).eval()
    finally:
        ref.LayerNorm = saved


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rms = rms_norm_class()
    tokens = torch.randint(0, 128, (2, 32), generator=torch.Generator().manual_seed(0))
    out = {}
    for name, norm in (("layer", ref.LayerNorm), ("rms", rms),
                       ("none", lambda d: torch.nn.Identity())):
        model = stack(ref, norm)
        with torch.no_grad():
            y = model(tokens)
        out[name] = {"shape": tuple(y.shape), "params": sum(p.numel() for p in model.parameters()),
                     "rms": y.pow(2).mean(-1).sqrt().mean().item(),
                     "abs_mean": y.mean(-1).abs().mean().item(), "y": y}
    x = torch.randn(2, 16, D_MODEL, generator=torch.Generator().manual_seed(1)) * 5 + 3
    with torch.no_grad():
        vs_torch = (rms(D_MODEL)(x) - torch.nn.RMSNorm(D_MODEL, eps=EPS)(x)).abs().max().item()
        ln_vs_nn = (ref.LayerNorm(64)(x[..., :64]) - torch.nn.LayerNorm(64)(x[..., :64]))
    src = inspect.getsource(ref)
    return {"out": out, "vs_torch": vs_torch, "ln_vs_nn": ln_vs_nn.abs().max().item(),
            "nn_ln_uses": src.count("nn.LayerNorm"),
            "nn_ln_in_doc": "Equivalent to nn.LayerNorm(d_model)" in src,
            "differs": not torch.allclose(out["rms"]["y"], out["layer"]["y"], atol=1e-3)}


def verify(result):
    r, o = result, result["out"]
    shapes = {k: v["shape"] for k, v in o.items()}
    return [
        practice.Check(
            "ANSWER: the shape is unchanged, (2, 32, 192) both ways",
            all([
                shapes["rms"] == shapes["layer"] == (2, 32, 192),
                o["layer"]["params"] - o["rms"]["params"] == 13 * D_MODEL,
                o["rms"]["params"] == 2_691_648,
                r["vs_torch"] < 1e-6,
                r["differs"],
                abs(o["rms"]["rms"] - 1) < 1e-3,
                o["rms"]["abs_mean"] > 0.01,
            ]),
            f"shapes {shapes['layer']} -> {shapes['rms']}; params {o['layer']['params']:,} -> "
            f"{o['rms']['params']:,}; vs torch.nn.RMSNorm {r['vs_torch']:.1e}; output RMS "
            f"{o['rms']['rms']:.3f}, |mean| {o['rms']['abs_mean']:.3f}",
        ),
        practice.Check(
            "FINDING: the exercise names a module the code never uses",
            all([
                r["nn_ln_uses"] == 1,
                r["nn_ln_in_doc"],
                r["ln_vs_nn"] < 1e-6,
            ]),
            f"'nn.LayerNorm' occurrences in main.py: {r['nn_ln_uses']} (a docstring); "
            f"hand-rolled LayerNorm vs nn.LayerNorm {r['ln_vs_nn']:.1e}",
        ),
        practice.Check(
            "FINDING: a shape check cannot tell a norm from no norm",
            all([
                shapes["none"] == (2, 32, 192),
                abs(o["layer"]["rms"] - 1) < 1e-3,
                abs(o["none"]["rms"] - 1.25) < 0.01,
            ]),
            f"nn.Identity shape {shapes['none']}; output RMS LayerNorm {o['layer']['rms']:.3f}, "
            f"RMSNorm {o['rms']['rms']:.3f}, no norm {o['none']['rms']:.2f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
