"""Exercise 1 — roialign parity and roipool drift.

    **(Easy)** Verify your RoIAlign against `torchvision.ops.roi_align` on 100
    random boxes. Report the max absolute difference. Also run RoIPool (pre-2017
    behaviour) and show it diverges by ~1-2 feature-map pixels on boxes near the
    border.

Reading of the exercise: "verify" is taken literally -- 100 random boxes rather
than the three hand-picked ones in `compare_with_torchvision_roi_align` -- and
the verification fails. The lesson's own text promises agreement "to within
1e-5"; across 100 boxes the max absolute difference is 0.25, four orders of
magnitude worse, and the failures are exactly the boxes whose sampling grid
reaches past the last pixel centre of the feature map. `F.grid_sample` zero-pads
there, torchvision's kernel replicates the border pixel, so the from-scratch
output is the reference scaled by the in-bounds bilinear weight -- checked below
in closed form, to four decimals. The RoIPool half is a false premise twice over.
RoIPool max-pools and RoIAlign bilinearly samples, so their raw difference is not
a displacement at all. The displacement is recovered by feeding both a coordinate
ramp -- channel 0 holding x, channel 1 holding y, so a returned value *is* a
feature-map coordinate -- and sliding one 80px box across 41 sub-pixel offsets:
the peak-to-peak of pool-minus-align over that sweep is the rounding error, and
its constant part is the max-versus-centre convention, which is not a
misalignment. And the drift is not worse "near the border": it is the same 0.950
feature-map pixels against the edge as in the middle, because it comes from
rounding the box, and rounding a box does not know where the box is.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "08-instance-segmentation-mask-rcnn"

SIDE, CHANNELS, SCALE, GRID, N_BOXES = 50, 16, 1 / 4, 7, 100
SWEEP, STEP = 41, 0.2          # 41 sub-pixel offsets, 0.2 image px = 0.05 feature px
FLAGS = ((True, 2), (True, -1), (False, 1), (False, 2), (False, -1))

listing = lambda pairs: ", ".join(f"aligned={a}/ratio={s} {d:.2f}" for (a, s), d in pairs)  # noqa: E731
overhang = lambda axis: (axis - (SIDE - 1)).clamp(min=0.0) + (-axis).clamp(min=0.0)         # noqa: E731


def random_boxes(torch):
    """100 boxes over the 200x200 image the 50x50 stride-4 feature map came from."""
    gen = torch.Generator().manual_seed(1)
    corner = torch.rand(N_BOXES, 2, generator=gen) * 190
    far = (corner + 8 + torch.rand(N_BOXES, 2, generator=gen) * 100).clamp(max=199.0)
    return torch.cat([torch.zeros(N_BOXES, 1), corner, far], dim=1)


def sample_grid(torch, box):
    """Where the lesson's own formula reads, so the failures are predicted not guessed."""
    x1, y1, x2, y2 = (c * SCALE - 0.5 for c in box)
    step_x, step_y = (x2 - x1) / GRID, (y2 - y1) / GRID
    return torch.linspace(x1 + step_x / 2, x2 - step_x / 2, GRID), torch.linspace(y1 + step_y / 2, y2 - step_y / 2, GRID)


def drift(torch, ops, start):
    """Slide an 80px box over the coordinate ramp; returns (peak-to-peak, constant offset)."""
    axis = torch.arange(SIDE, dtype=torch.float32)
    ramp = torch.stack([axis.expand(SIDE, SIDE), axis.unsqueeze(1).expand(SIDE, SIDE)]).unsqueeze(0)
    shifts = [torch.tensor([[0.0, s, s, s + 80.0, s + 80.0]])
              for s in (start + k * STEP for k in range(SWEEP))]
    stack = torch.stack([ops.roi_pool(ramp, b, (GRID, GRID), SCALE)[0]
                         - ops.roi_align(ramp, b, (GRID, GRID), SCALE, 1, True)[0] for b in shifts])
    return float((stack.amax(0) - stack.amin(0)).max()), float(stack[:, 0].mean())


def solve():
    try:
        import torch
        from torchvision import ops
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    feature, boxes = torch.randn(1, CHANNELS, SIDE, SIDE), random_boxes(torch)
    mine = torch.stack([ref.roi_align_single(feature[0], b[1:].tolist(), GRID, SCALE) for b in boxes])
    theirs = ops.roi_align(feature, boxes, (GRID, GRID), SCALE, 1, True)
    gap = (mine - theirs).abs().amax(dim=(1, 2, 3))
    grids = [sample_grid(torch, b[1:].tolist()) for b in boxes]
    worst = int(gap.argmax())
    gx, gy = grids[worst]
    chan, rest = divmod(int((mine[worst] - theirs[worst]).abs().flatten().argmax()), GRID * GRID)
    row, col = divmod(rest, GRID)
    inward = boxes[worst] - torch.tensor([0.0, 12.0, 12.0, 12.0, 12.0])
    return {"gap": gap, "worst": (worst, boxes[worst, 1:].tolist()), "cell": (row, col),
            "spill": torch.tensor([float(overhang(a).max() + overhang(b).max()) for a, b in grids]),
            "fracs": (float(overhang(gx)[col]), float(overhang(gy)[row])),
            "columns": (overhang(gx) > 0).nonzero().flatten().tolist(),
            "values": (float(mine[worst][chan, row, col]), float(theirs[worst][chan, row, col])),
            "inward": float((ref.roi_align_single(feature[0], inward[1:].tolist(), GRID, SCALE)
                             - ops.roi_align(feature, inward.unsqueeze(0), (GRID, GRID), SCALE, 1, True)[0]).abs().max()),
            "middle": drift(torch, ops, 40.0), "edge": drift(torch, ops, 108.0),
            "flags": [((a, s), float((mine - ops.roi_align(feature, boxes, (GRID, GRID), SCALE, s, a)).abs()
                                     .max())) for a, s in FLAGS]}


def verify(result):
    gap, outside, (index, box) = result["gap"], result["spill"] > 0, result["worst"]
    (ours, ref_val), (fx, fy), (row, col) = result["values"], result["fracs"], result["cell"]
    predicted, loud, mid, edge = ref_val * (1 - fx) * (1 - fy), gap > 1e-4, result["middle"], result["edge"]
    return [
        practice.Check(
            "ANSWER: max|diff| over 100 boxes is 0.25, not the 1e-5 the lesson promises",
            float(gap.max()) > 1e-3 and float(gap[~loud].max()) < 1e-4,
            f"{N_BOXES} random boxes, {SIDE}x{SIDE} feature map, spatial_scale={SCALE}: max|ours - torchvision| "
            f"= {float(gap.max()):.4f}, mean {float(gap.mean()):.5f}. {int(loud.sum())} boxes exceed 1e-4; the "
            f"other {int((~loud).sum())} agree to {float(gap[~loud].max()):.2e} -- the lesson picked lucky boxes"),
        practice.Check(
            "MECHANISM: the failures are exactly the boxes that sample past the last pixel centre",
            bool((loud == outside).all()) and int(outside.sum()) == int(loud.sum()) > 0,
            f"a grid is in range while its samples stay inside [0, {SIDE - 1}]; {int(outside.sum())} of {N_BOXES} "
            f"boxes break that and they are the same {int(loud.sum())} that disagree. Worst is box {index} = "
            f"{[round(v, 1) for v in box]}, whose only bad columns are {result['columns']}"),
        practice.Check(
            "MECHANISM: grid_sample zero-pads where torchvision replicates, to four decimals",
            abs(ours - predicted) < 1e-3,
            f"its worst cell (row {row}, col {col}) samples {fx:.4f} px past x={SIDE - 1} and {fy:.4f} past "
            f"y={SIDE - 1}. padding_mode='zeros' weights those fractions onto nothing, leaving "
            f"ref*(1-{fx:.4f})*(1-{fy:.4f}) = {predicted:.4f}; measured {ours:.4f} against torchvision's "
            f"{ref_val:.4f}, which clamps to the border pixel instead"),
        practice.Check(
            "CONTROL: slide the same box 12px inside and the disagreement is gone",
            result["inward"] < 1e-4,
            f"the same box moved 12 image px (3 feature px) up and left, same shape, size and aspect, drops "
            f"to {result['inward']:.2e}: distance to the map edge is the whole variable, and the "
            f"from-scratch kernel is otherwise exact"),
        practice.Check(
            "CONTROL: the agreement holds for one of torchvision's six configurations",
            all(value > 1.0 for _, value in result["flags"]),
            f"the same 100 boxes under the other flag settings, max|diff|: {listing(result['flags'])} -- against "
            f"{float(gap.max()):.2f} at aligned=True/ratio=1, the one configuration this code implements"),
        practice.Check(
            "FINDING: RoIPool drifts 0.95 feature px, and not one bit more at the border",
            0.85 < mid[0] < 1.15 and abs(mid[0] - edge[0]) < 0.02,
            f"over {SWEEP} sub-pixel offsets of an 80px box on the coordinate ramp, pool-minus-align swings "
            f"{mid[0]:.3f} px mid-map and {edge[0]:.3f} px against the right/bottom edge; the constant part "
            f"({mid[1]:.3f} vs {edge[1]:.3f}) is the max-versus-centre convention, not a misalignment"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
