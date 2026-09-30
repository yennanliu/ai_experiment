"""Exercise 4 -- the patch step is 0.66% of ViT-B/16's FLOPs, but costs 1.94x a layer's attention-score matmuls.

    Profile the patch step at batch sizes 1, 8, 64. The patch projection is rarely the bottleneck; the attention layers downstream dominate.

Reading of the exercise: "the patch step" is the lesson's `PatchEmbed` at its
default ViT-B/16 config (224 px, patch 16, hidden 768). "The attention layers
downstream" are the 12 encoder layers the next lesson stacks on it. Each is
modelled as `nn.TransformerEncoderLayer(768, 12 heads, 3072, pre-LN)` on the
197-token output, and 12 layers are timed as 12 x one (the layers are
identical). Two profiles are taken. FLOPs are counted per image, analytically,
with torch's `FlopCounterMode` as a cross-check on the patch step. Wall clock
is the best of 3 on one CPU thread at each batch size. Timing checks use wide
margins, because the numbers depend on the machine.

**ANSWER: the patch step is not the bottleneck at any batch size.** It is
231.2 MFLOPs per image against 2.908 GFLOPs per encoder layer, so 0.66% of
patch + 12 layers. The FLOP count is independent of batch. In wall clock its
share of patch + 12 layers was about 0.6%, 0.8% and 7% at batch 1, 8 and 64 on
the authoring machine; the check only requires under 25% at each. The rise at
batch 64 is the Conv2d slowdown in the last finding, not extra work.

**FINDING: inside a layer the MLP dominates, not attention.** Per image per
layer: QKV 697.2M, output projection 232.4M, MLP 1,859.1M (64% of the
layer), and the two 197 x 197 attention matmuls (QK^T and AV) only 119.2M
(4.1%). The patch projection alone is 1.94x one layer's attention-score
matmuls. What makes the stack expensive is 12 x the linears, and the
quadratic term that motivated patching in the first place is small at 197
tokens.

**FINDING: on this CPU the Conv2d spelling falls off a cliff past batch 8.**
The lesson says production code ships the Conv2d because "it is faster on
GPU". On CPU (one thread, torch 2.14, Apple arm64) its per-image time was
0.4 ms at batch 1 and 8, then about 4-5 ms at batch 64. At batch 64 it is
about 11x slower than the lesson's own `unfold_then_linear`, which computes
the same output as one matmul (the run prints the ratio). The check requires a factor of 2. The exact ratio is
machine-dependent; the direction is what the run shows.

Structure: `layer_flops()` counts one encoder layer analytically; `best()`
times one call; `solve()` profiles the three batch sizes.
"""

from __future__ import annotations

import time

from harness import parity, practice

try:
    import torch
    from torch import nn
    from torch.utils.flop_counter import FlopCounterMode
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
torch.set_num_threads(1)

PHASE, LESSON = "19-capstone-projects", "58-vision-encoder-patches"
T, D, FF, LAYERS, BATCHES = 197, 768, 3072, 12, (1, 8, 64)


def layer_flops():
    """Multiply-add x 2 for one pre-LN encoder layer on one 197-token image."""
    return {"qkv": 2 * T * D * 3 * D, "out": 2 * T * D * D, "mlp": 2 * 2 * T * D * FF,
            "attn": 2 * 2 * T * T * D}


def best(fn, reps=3):
    fn()
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.manual_seed(0)
    front = ref.VisionFrontEnd(ref.FrontEndConfig()).eval()
    layer = nn.TransformerEncoderLayer(D, 12, FF, dropout=0.0, batch_first=True, norm_first=True)
    w, b = front.patch.proj.weight, front.patch.proj.bias
    out = {"wall": {}, "unfold": {}, "flops": layer_flops()}
    with torch.no_grad():
        img = ref.synthesize_image(0)
        with FlopCounterMode(display=False) as counter:
            front.patch(img)
        out["patch_flops"] = counter.get_total_flops()
        for n in BATCHES:
            x = img.repeat(n, 1, 1, 1)
            tokens = front(x)
            out["wall"][n] = (best(lambda: front.patch(x)), best(lambda: layer(tokens)))
            out["unfold"][n] = best(lambda: ref.unfold_then_linear(x, w, b, 16))
    return out


def verify(result):
    r, f = result, result["flops"]
    per_layer = sum(f.values())
    share = r["patch_flops"] / (r["patch_flops"] + LAYERS * per_layer)
    wall = {n: round(p / (p + LAYERS * lay), 4) for n, (p, lay) in r["wall"].items()}
    ms = {n: round(p * 1e3, 1) for n, (p, _) in r["wall"].items()}
    cliff = r["wall"][64][0] / r["unfold"][64]
    return [
        practice.Check(
            "ANSWER: the patch step is not the bottleneck at any batch size",
            all([
                r["patch_flops"] == 2 * 196 * D * D == 231_211_008,
                per_layer == 2_907_909_120,
                round(share, 4) == 0.0066,
                max(wall.values()) < 0.25,
            ]),
            f"patch {r['patch_flops']:,} FLOPs/image vs {per_layer:,} per layer "
            f"({share:.2%} of patch + 12 layers); wall-clock patch share by batch {wall}",
        ),
        practice.Check(
            "FINDING: inside a layer the MLP dominates, not attention",
            all([
                round(f["mlp"] / per_layer, 2) == 0.64,
                round(f["attn"] / per_layer, 3) == 0.041,
                round(r["patch_flops"] / f["attn"], 2) == 1.94,
            ]),
            f"per layer {f}; MLP {f['mlp'] / per_layer:.0%}, attention matmuls "
            f"{f['attn'] / per_layer:.1%}; patch / attention matmuls {r['patch_flops'] / f['attn']:.2f}x",
        ),
        practice.Check(
            "FINDING: on this CPU the Conv2d spelling falls off a cliff past batch 8",
            cliff > 2,
            f"Conv2d patch ms by batch {ms}; unfold_then_linear at 64 "
            f"{r['unfold'][64] * 1e3:.1f} ms; Conv2d / unfold at 64 = {cliff:.1f}x",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
