"""Exercise 2 -- nn.Unfold + nn.Linear matches the Conv2d to 1.2e-6, and a channels-last weight flatten is off by 0.63.

    Swap the `Conv2d` for an explicit `nn.Unfold` plus `nn.Linear` and assert the outputs match to within float tolerance. Same math, two ways to spell it.

Reading of the exercise: the lesson's `PatchEmbed` at its default ViT-B/16
config (224 px, patch 16, hidden 768, seed 0) is compared with
`nn.Unfold(16, stride=16)` followed by an `nn.Linear(768, 768)` that holds the
same weights, on the lesson's `synthesize_image(0)` fixture and on a random
batch of 4. The lesson's own `unfold_then_linear` (a `Tensor.unfold` spelling)
is the third way. The weight copy is `conv.weight.reshape(hidden, -1)`.

**ANSWER: they match to 1.19e-6 in float32 and 1.8e-15 in float64.** On the
fixture the Conv2d and the Unfold+Linear outputs differ by at most 1.19e-6,
which is 84x inside the 1e-4 bar `main.py` uses and 8x inside the test's
1e-5, on outputs as large as 2.02. They are not bitwise equal, because the two
kernels sum in different orders. On a batch of 4 uniform random images the gap is
9.5e-7. In float64 it is 1.8e-15, so the difference is rounding, not math.
Both modules hold 590,592 parameters, the doc's ViT-B/16 count.

**FINDING: nn.Unfold and the lesson's `unfold_then_linear` are bitwise equal.**
The two unfold spellings agree exactly (difference 0.0) and share the same
1.19e-6 gap to the Conv2d. `nn.Unfold` lays out each column channel-major,
`(c, ph, pw)`, which is exactly `Conv2d.weight.reshape(hidden, -1)`.

**FINDING: the one way to get this wrong is the weight layout.** Flattening
the kernel as `(ph, pw, c)`, the channels-last order an image library hands
you, still runs and returns the right shape, but differs from the Conv2d by
0.63 in float64. A shape check cannot see this; only the numeric comparison
the exercise asks for does.

**FINDING: both spellings drop edge pixels in silence.** On a 230x230 input
the Conv2d and `nn.Unfold` both return 196 patches and discard the last 6
rows and columns: 2,724 of 52,900 pixels per channel (5.1%). Only the lesson's
`PatchEmbed.forward` refuses the size, and it refuses every size except 224.

Structure: `pair()` builds the Conv2d and the matching Unfold+Linear;
`solve()` measures the gaps.
"""

from __future__ import annotations

from harness import parity, practice

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
torch.set_num_threads(1)

PHASE, LESSON = "19-capstone-projects", "58-vision-encoder-patches"


def pair(ref):
    torch.manual_seed(0)
    cfg = ref.FrontEndConfig()
    patch = ref.PatchEmbed(cfg).eval()
    unfold = nn.Unfold(cfg.patch_size, stride=cfg.patch_size)
    linear = nn.Linear(cfg.in_channels * cfg.patch_size**2, cfg.hidden)
    with torch.no_grad():
        linear.weight.copy_(patch.proj.weight.reshape(cfg.hidden, -1))
        linear.bias.copy_(patch.proj.bias)
    return patch, unfold, linear


def gap(a, b):
    return (a - b).abs().max().item()


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    patch, unfold, linear = pair(ref)
    img = ref.synthesize_image(0)
    batch = torch.rand(4, 3, 224, 224, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        conv, spelled = patch(img), linear(unfold(img).transpose(1, 2))
        lesson = ref.unfold_then_linear(img, patch.proj.weight, patch.proj.bias, 16)
        out = {
            "fp32": gap(conv, spelled), "bitwise": torch.equal(conv, spelled),
            "batch": gap(patch(batch), linear(unfold(batch).transpose(1, 2))),
            "lesson_vs_unfold": gap(lesson, spelled), "peak": conv.abs().max().item(),
            "params": [sum(p.numel() for p in m.parameters()) for m in (patch, linear)],
        }
        patch, linear, img = patch.double(), linear.double(), img.double()
        conv = patch(img)
        out["fp64"] = gap(conv, linear(unfold(img).transpose(1, 2)))
        hwc = patch.proj.weight.permute(0, 2, 3, 1).reshape(768, -1)
        out["hwc"] = gap(conv, unfold(img).transpose(1, 2) @ hwc.T + patch.proj.bias)
        odd = torch.rand(1, 3, 230, 230, dtype=torch.float64)
        out["odd"] = [patch.proj(odd).flatten(2).shape[-1], unfold(odd).shape[-1]]
    try:
        patch(odd)
    except ValueError as exc:
        out["refuses"] = str(exc)
    return out


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: they match to 1.19e-6 in float32 and 1.8e-15 in float64",
            r["fp32"] < 2e-6 and r["batch"] < 3e-6 and r["fp64"] < 1e-14
            and not r["bitwise"] and r["params"] == [590_592, 590_592],
            f"float32 {r['fp32']:.3g} (outputs up to {r['peak']:.2f}), batch of 4 "
            f"{r['batch']:.3g}, float64 {r['fp64']:.2g}, bitwise equal {r['bitwise']}, "
            f"params {r['params']}",
        ),
        practice.Check(
            "FINDING: nn.Unfold and the lesson's unfold_then_linear are bitwise equal",
            r["lesson_vs_unfold"] == 0.0,
            f"max gap between the two unfold spellings {r['lesson_vs_unfold']}",
        ),
        practice.Check(
            "FINDING: the one way to get this wrong is the weight layout",
            r["hwc"] > 0.1,
            f"(ph, pw, c) flatten vs Conv2d in float64: {r['hwc']:.3f}",
        ),
        practice.Check(
            "FINDING: both spellings drop edge pixels in silence",
            r["odd"] == [196, 196] and "spatial mismatch" in r.get("refuses", ""),
            f"230 px input -> Conv2d / nn.Unfold patches {r['odd']}; "
            f"{230 * 230 - 224 * 224:,} of {230 * 230:,} pixels dropped; "
            f"PatchEmbed: {r.get('refuses')!r}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
