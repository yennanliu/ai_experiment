"""Exercise 3 — mask head 56 vs 28 ceiling.

    **(Hard)** Replace Mask R-CNN's mask head with one that predicts at 56x56
    instead of 28x28. Measure mAP@IoU=0.75 before and after. Explain why the gain
    (or lack of one) matches the expected boundary-precision / memory trade-off.

Reading of the exercise: the swap itself is the easy half and is done here for
real -- one more 2x `ConvTranspose2d` in front of the stock `MaskRCNNPredictor`,
so RoIAlign's 14x14 patch runs 14 -> 28 -> 56 -- and the patch fed through it
comes from the lesson's own `roi_align_single`, at the output_size=14 the mask
branch actually uses.

`widen` takes the `MaskRCNNPredictor` class from the model itself rather than
importing it, so the 56x56 head is built exactly like the 28x28 one it replaces.

The measurement half is a false premise. "mAP@0.75 before and after" is a number
about a trained detector on a real dataset; the lesson's constructors hard-code
`weights=...DEFAULT`, so producing it means a 170 MB checkpoint fetch plus a
training run, neither of which belongs in a reproducible test. More importantly,
that number could not answer the question even if it were free, because at a
budget small enough to run here the arms differ by training noise, not by
resolution. So the explanation the exercise asks for is measured directly and
deterministically instead: push a ground-truth mask through an RxR grid and paste
it back at the object's pixel size, which is exactly the encode/decode a mask
head of resolution R is bounded by, and read off the IoU that survives. That is
the ceiling on what any RxR head could ever score.

The ceiling settles it. At 28x28 the round trip keeps 1.0000 IoU on a 24px
object, 0.9804 at 64px, 0.9634 at 128px and 0.9607 at 256px -- every one of them
far above 0.75, so at the threshold the exercise picked not a single instance is
lost to mask resolution and mAP@0.75 has no room to move. The two resolutions do
not separate until roughly IoU 0.97, and only on large objects: at 24px both are
exactly 1.0000, because a 28x28 grid already over-samples a 24-pixel box and the
extra deconv is interpolating its own upsampling. What the swap does buy, at
every object size, is 2.0x the mask-predictor parameters and 4x the logit bytes;
taking the other route to 56x56 -- RoIAlign at 28x28 with the stock single deconv
-- costs 4x the mask-head activations instead. Both are measured below, and the
real command and its GPU cost for the trained comparison are printed with them.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "08-instance-segmentation-mask-rcnn"

BOX_SIZES, PROPOSALS, CLASSES, ROI, HIDDEN = (24, 64, 128, 256), 100, 3, 14, 256
REAL_RUN = ("the trained comparison needs the checkpoint: `python finetune.py --weights coco --mask-res 56 "
            "--images 50 --epochs 20 --min-size 800` is ~5 GPU-min an arm on an A10G, about 4 cents")

table = lambda pairs: ", ".join(f"{k}px {v:.4f}" for k, v in pairs)                                    # noqa: E731
disk = lambda xx, yy, c, r: ((xx - c) ** 2 + (yy - c) ** 2) < r ** 2                                   # noqa: E731
diamond = lambda xx, yy, c, r: (xx - c).abs() + (yy - c).abs() < r                                     # noqa: E731
star = lambda xx, yy, c, r: (((xx - c) ** 2 + (yy - c) ** 2).sqrt() <                                  # noqa: E731
                             r * (0.55 + 0.45 * ((yy - c).atan2(xx - c) * 5).cos().abs()))
bytes_of = lambda logits: PROPOSALS * logits[0].numel() * 4                                            # noqa: E731


def instance(torch, side, kind):
    """One ground-truth instance filling a side x side box: disk, diamond or 5-lobed star."""
    axis = torch.arange(side, dtype=torch.float32)
    yy, xx = axis[:, None].expand(side, side), axis.expand(side, side)
    return (disk, diamond, star)[kind](xx, yy, (side - 1) / 2, side * 0.45)


def ceiling(torch, resize, side, resolution):
    """Encode each instance at RxR and paste it back: the best IoU an RxR head could score."""
    kept = []
    for kind in range(3):
        truth = instance(torch, side, kind)
        small = resize(truth.float()[None, None], (resolution, resolution), mode="bilinear", align_corners=False)
        back = resize(small, (side, side), mode="bilinear", align_corners=False)[0, 0] > 0.5
        kept.append(float((truth & back).sum() / (truth | back).sum()))
    return sum(kept) / len(kept)


def widen(nn, model):
    """14 -> 28 -> 56: one more 2x deconv in front of the predictor the lesson swaps in."""
    stock = model.roi_heads.mask_predictor
    channels = stock.conv5_mask.in_channels
    return nn.Sequential(nn.ConvTranspose2d(channels, channels, 2, 2), nn.ReLU(inplace=True),
                         type(stock)(channels, HIDDEN, CLASSES))


def solve():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=CLASSES)
    small, big = model.roi_heads.mask_predictor, widen(nn, model)
    feature = torch.randn(1, HIDDEN, 50, 50)        # one FPN level, stride 16
    patch = ref.roi_align_single(feature[0], [96.0, 112.0, 288.0, 304.0], ROI, 1 / 16).unsqueeze(0)
    with torch.no_grad():
        shared = model.roi_heads.mask_head(patch)
        thin, wide = small(shared), big(shared)
    head_bytes = [PROPOSALS * HIDDEN * side * side * 4 * 4 for side in (ROI, 2 * ROI)]
    return {"roi": tuple(model.roi_heads.mask_roi_pool.output_size), "patch": tuple(patch.shape),
            "sides": (int(thin.shape[-1]), int(wide.shape[-1])),
            "logits": (tuple(thin.shape), tuple(wide.shape)),
            "params": (sum(p.numel() for p in small.parameters()), sum(p.numel() for p in big.parameters())),
            "bytes": (bytes_of(thin), bytes_of(wide)), "head_bytes": head_bytes,
            "ceiling": {r: [(s, ceiling(torch, functional.interpolate, s, r)) for s in BOX_SIZES] for r in (28, 56)}}


def verify(result):
    thin, wide = result["sides"]
    low, high = dict(result["ceiling"][28]), dict(result["ceiling"][56])
    params, byte_count, biggest, tiny = result["params"], result["bytes"], max(BOX_SIZES), min(BOX_SIZES)
    crossing = round((low[biggest] + high[biggest]) / 2, 2)
    return [
        practice.Check(
            "ANSWER: the swapped head emits 56x56, for 2.0x the parameters and 4x the logit bytes",
            (thin, wide) == (28, 56) and result["roi"] == (ROI, ROI),
            f"the lesson's own `roi_align_single` supplies the {result['patch']} patch, the size mask_roi_pool "
            f"uses ({result['roi']}); one extra 2x ConvTranspose2d turns {result['logits'][0]} into "
            f"{result['logits'][1]}, the predictor grows {params[0]:,} -> {params[1]:,} ({params[1] / params[0]:.1f}x) "
            f"and {PROPOSALS} proposals of logits grow {byte_count[0] / 1e6:.2f} -> "
            f"{byte_count[1] / 1e6:.2f} MB ({byte_count[1] // byte_count[0]}x)"),
        practice.Check(
            "ANSWER: the grid is not what holds mAP@0.75 back -- its ceiling is 0.96 at worst",
            min(low.values()) > 0.95,
            f"pushing a ground-truth mask through an RxR grid and pasting it back is the encode/decode any RxR "
            f"head is bounded by. Mean IoU surviving at 28x28: {table(result['ceiling'][28])}. The worst of "
            f"those clears IoU 0.75 by {min(low.values()) - 0.75:.2f}, so mask resolution never puts an "
            f"instance under the threshold the exercise picked. This bounds a *perfect* mask, so it does not "
            f"say mAP cannot move -- it says any movement comes from what the head predicts, not from the "
            f"grid it predicts on, which is the claim the exercise's 'why' is really about. {REAL_RUN}"),
        practice.Check(
            "MECHANISM: the two resolutions separate only on objects bigger than the grid",
            min(low[tiny], high[tiny]) > 0.999 and (1 - low[biggest]) > 1.8 * (1 - high[biggest]),
            f"the 56x56 ceiling runs {table(result['ceiling'][56])}. At {tiny}px both are {low[tiny]:.4f}: a "
            f"28x28 grid already over-samples a {tiny}px box, so the extra deconv only interpolates its own "
            f"upsampling. At {biggest}px the residual error falls "
            f"{(1 - low[biggest]) / (1 - high[biggest]):.1f}x, {1 - low[biggest]:.4f} -> {1 - high[biggest]:.4f}"),
        practice.Check(
            "FINDING: the threshold that could see this swap sits near IoU %.2f" % crossing,
            low[biggest] < crossing < high[biggest],
            f"at {biggest}px the ceilings are {low[biggest]:.4f} and {high[biggest]:.4f}, so only a threshold "
            f"between them -- IoU {crossing:.2f}, say -- separates the two heads on the representation alone. "
            f"COCO's mAP averages 0.50:0.95 and stops below that, which is why 28x28 is still the default"),
        practice.Check(
            "CONTROL: the other route to 56x56 spends the 4x on activations instead of parameters",
            result["head_bytes"][1] == 4 * result["head_bytes"][0],
            f"pooling RoIAlign at 28x28 and keeping the stock single deconv reaches the same 56x56 grid, but the "
            f"four 3x3 convs of `mask_head` then run at 28x28, so {PROPOSALS} proposals cost "
            f"{result['head_bytes'][0] / 1e6:.1f} -> {result['head_bytes'][1] / 1e6:.1f} MB of activations "
            f"against {params[1] - params[0]:,} parameters for the deconv route -- same grid, different bill"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
