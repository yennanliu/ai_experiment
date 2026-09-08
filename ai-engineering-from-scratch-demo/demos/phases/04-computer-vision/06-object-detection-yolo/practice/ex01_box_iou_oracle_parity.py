"""Exercise 1 — box iou oracle parity.

    **(Easy)** Implement `box_iou` and run it against `torchvision.ops.box_iou`
    on 1,000 random box pairs. Verify max absolute difference is below `1e-6`.

Reading of the exercise: D5 forbids re-implementing the lesson's `box_iou`, so
the thing under test is the lesson's own function. `torchvision.ops.box_iou` is
compared against whenever the `vision` extra provides it (the check states which
case it ran in), but it cannot carry the exercise on its own: it evaluates the
*same* algebra -- edge lengths multiplied into areas -- so agreeing with it to
1e-6 mostly proves that two spellings of one formula round the same way. The
weight is carried instead by three oracles that share none of that structure: a
scalar loop over Python floats, a rasteriser that counts unit cells inside each
box, and a Monte-Carlo point sampler. The 1e-6 threshold is then examined rather
than trusted, and it turns out to be one that no float question can fail and only
a *convention* question can, so the last three checks measure what it actually
discriminates.

Structure, since the helpers carry no docstrings of their own:

    random_boxes    valid xyxy boxes: two coordinates per axis, sorted
    scalar_iou      oracle 1 -- one pair at a time, Python floats, `min`/`max`,
                    no broadcasting and no numpy
    raster_iou      oracle 2 -- no area formula at all: paint both boxes into
                    boolean masks and count the cells in both over either
    monte_carlo_iou oracle 3 -- sample points in the enclosing box and count
                    where they land, so the estimate carries sampling noise
    voc_iou         the historical PASCAL VOC convention, width = x2 - x1 + 1,
                    used as the control for what the 1e-6 tolerance discriminates
    integer_boxes   whole-pixel boxes at least one cell wide, so the rasteriser
                    has something to count
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "06-object-detection-yolo"

PAIRS, EXTENT, TOLERANCE, SAMPLES, MC_PAIRS = 1000, 100.0, 1e-6, 200_000, 50
RASTER_PAIRS, RASTER_SIDE, BIG = 200, 64, 40.0   # integer boxes in a 64x64 grid; 4,000-px coords for float32
DEGENERATE = (([5.0] * 4, [5.0] * 4), ([5.0] * 4, [0.0, 0.0, 10.0, 10.0]), ([10.0, 10.0, 0.0, 0.0], [0.0, 0.0, 10.0, 10.0]))

# comprehensions kept at module level so the helpers below stay inside D14's complexity 8
diagonal = lambda fn, a, b: [float(fn(a[[i]], b[[i]])[0, 0]) for i in range(len(a))]   # noqa: E731
apply2 = lambda fn, a, b, n: [fn(a[i], b[i]) for i in range(n)]                        # noqa: E731
area_voc = lambda z: (z[:, 2] - z[:, 0] + 1) * (z[:, 3] - z[:, 1] + 1)                 # noqa: E731


def random_boxes(np, rng, count, extent=EXTENT):
    x, y = (np.sort(rng.uniform(0, extent, size=(count, 2)), axis=1) for _ in range(2))
    return np.stack([x[:, 0], y[:, 0], x[:, 1], y[:, 1]], axis=1)


def scalar_iou(a, b) -> float:
    inter_w, inter_h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    inter = inter_w * inter_h if inter_w > 0 and inter_h > 0 else 0.0
    return inter / union if (union := (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter) > 0 else 0.0


def raster_iou(np, a, b) -> float:
    mask_a, mask_b = (np.zeros((RASTER_SIDE,) * 2, bool) for _ in range(2))
    mask_a[int(a[1]):int(a[3]), int(a[0]):int(a[2])], mask_b[int(b[1]):int(b[3]), int(b[0]):int(b[2])] = True, True
    return float((mask_a & mask_b).sum()) / max(int((mask_a | mask_b).sum()), 1)


def monte_carlo_iou(np, rng, a, b) -> float:
    points = rng.uniform(np.minimum(a[:2], b[:2]), np.maximum(a[2:], b[2:]), size=(SAMPLES, 2))
    in_a, in_b = (np.all((points >= z[:2]) & (points < z[2:]), axis=1) for z in (a, b))
    return float((in_a & in_b).sum()) / max(int((in_a | in_b).sum()), 1)   # 0 union -> 0, never 0/0


def voc_iou(np, a, b):
    inter = np.clip(np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]) + 1, 0, None) * np.clip(np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]) + 1, 0, None)
    return inter / (area_voc(a) + area_voc(b) - inter)


def integer_boxes(np, rng, count):
    boxes = random_boxes(np, rng, count, extent=RASTER_SIDE * 0.6).round()
    return np.concatenate([boxes[:, :2], np.maximum(boxes[:, 2:], boxes[:, :2] + 1)], axis=1)


def solve():
    try:
        import numpy as np
    except ImportError as exc:                      # pragma: no cover - T1 needs numpy
        raise practice.Skip(f"needs numpy: uv sync --extra vision ({exc})") from None
    ref, rng = parity.load_reference(PHASE, LESSON, "main"), np.random.default_rng(0)
    a, b, ra, rb = *(random_boxes(np, rng, PAIRS) for _ in range(2)), *(integer_boxes(np, rng, RASTER_PAIRS) for _ in range(2))
    got, wide = np.array(diagonal(ref.box_iou, a, b)), ((a * BIG).astype(np.float32), (b * BIG).astype(np.float32))
    try:                                            # the exercise's named oracle, when the vision extra ships it
        import torch
        from torchvision.ops import box_iou as tv_box_iou
        named = float(np.abs(np.diag(tv_box_iou(torch.tensor(a), torch.tensor(b)).numpy()) - got).max())
    except ImportError:                             # pragma: no cover - torchvision is optional here
        named = None
    return {"iou": got, "overlap": got > 0, "voc": np.abs(voc_iou(np, a, b) - got), "named": named,
            "degenerate": [float(ref.box_iou(np.array([p]), np.array([q]))[0, 0]) for p, q in DEGENERATE],
            "scalar": np.abs(got - np.array(apply2(scalar_iou, a, b, PAIRS))).max(),
            "raster": np.abs(np.array(diagonal(ref.box_iou, ra, rb))
                             - np.array(apply2(lambda p, q: raster_iou(np, p, q), ra, rb, RASTER_PAIRS))).max(),
            "monte": np.abs(got[:MC_PAIRS] - np.array(apply2(lambda p, q: monte_carlo_iou(np, rng, p, q), a, b, MC_PAIRS))).max(),
            "float32": np.abs(np.array(diagonal(ref.box_iou, *wide)) - np.array(apply2(scalar_iou, a * BIG, b * BIG, PAIRS))).max(),
            "matrix": float(np.abs(np.diag(ref.box_iou(a, b)) - got).max())}


def verify(result):
    voc, overlap, scalar, f32, named = (result[k] for k in ("voc", "overlap", "scalar", "float32", "named"))
    quoted = "torchvision.ops.box_iou is not installed" if named is None else f"torchvision.ops.box_iou to {named:.3e}"
    return [
        practice.Check(
            f"ANSWER: {PAIRS} random pairs agree with the named oracle and with an independent scalar one to 0.0",
            scalar < 1e-12 and (named is None or named < 1e-12),
            f"max |lesson - scalar Python-float loop| over {PAIRS} pairs is {scalar:.3e}, and the lesson matches {quoted}, both against the "
            f"{TOLERANCE:.0e} asked for. IoU spans {result['iou'].min():.3f} to {result['iou'].max():.3f}; {int(overlap.sum())} pairs overlap"),
        practice.Check(
            "ANSWER: a rasteriser that never multiplies edge lengths agrees exactly too",
            result["raster"] < 1e-12,
            f"painting {RASTER_PAIRS} integer box pairs into a {RASTER_SIDE}x{RASTER_SIDE} grid and dividing the count of cells in both by "
            f"the count in either reproduces the lesson to {result['raster']:.3e}: the half-open x2 - x1 convention counts pixels"),
        practice.Check(
            "FINDING: the third oracle agrees only to its sampling error, not to 1e-6",
            1e-5 < result["monte"] < 5e-3,
            f"{SAMPLES:,} uniform points per pair over {MC_PAIRS} pairs land within {result['monte']:.2e} of the closed form -- 1/sqrt(N) noise, "
            "three orders above the tolerance: exactness in the other oracles comes from computing an area in closed form"),
        practice.Check(
            "CONTROL: 1e-6 sits between the two dtypes -- it is a float32 tolerance, not a test",
            1e-9 < f32 < TOLERANCE and scalar < 1e-12 < f32,
            f"the same boxes at {EXTENT * BIG:.0f}-px coordinates in float32, scored against the float64 scalar oracle, give {f32:.2e} -- inside "
            f"{TOLERANCE:.0e} by only {TOLERANCE / f32:.0f}x, where float64 clears it by every digit it has ({scalar:.1e}). IoU is a ratio in "
            "[0, 1], so coordinate magnitude cancels and only the storage dtype's ~1e-7 relative epsilon survives"),
        practice.Check(
            "CONTROL: what 1e-6 does catch is the convention, by four orders of magnitude",
            voc.max() > 1e-2 and voc[overlap].mean() > 100 * TOLERANCE,
            f"scoring the same {PAIRS} pairs under the PASCAL VOC +1 width convention moves IoU by up to {voc.max():.3f} (mean "
            f"{voc[overlap].mean():.4f} over the {int(overlap.sum())} overlapping pairs), {voc.max() / TOLERANCE:.0e}x the tolerance: it "
            "discriminates box conventions, not arithmetic"),
        practice.Check(
            "FINDING: clip(union, 1e-8) makes the function total, and broadcasting is consistent",
            result["degenerate"] == [0.0, 0.0, 0.0] and result["matrix"] < 1e-12,
            f"a zero-area box against itself, against a real box, and an inverted box (x2 < x1) return {result['degenerate']} rather than 0/0; "
            f"the {PAIRS}x{PAIRS} call's diagonal matches the {PAIRS} single-pair calls to {result['matrix']:.3e}, so the [:, None] broadcast "
            "is not silently transposing a or b"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
