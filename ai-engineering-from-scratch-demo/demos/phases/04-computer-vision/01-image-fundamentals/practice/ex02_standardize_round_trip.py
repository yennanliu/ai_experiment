"""Exercise 2 — standardize and its inverse, on one image and on a batch.

    **(Medium)** Write `standardize(img, mean, std)` and its inverse that together
    pass a `roundtrip_max_diff <= 1` test on any uint8 image. Your functions must
    work on a single image in HWC and on a batch in NCHW with the same call.

Reading of the exercise: "any uint8 image" is taken literally in check 1 — a ramp carrying all
256 values, where a correct pair scores 0 rather than the 1 the exercise allows. Checks 2-3
measure the lesson's own pair against the same bar and locate the one line that costs it. "The
same call" for HWC and NCHW is a broadcasting problem, not a signature problem, and check 4 is
the shape where getting it wrong is silent.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "01-image-fundamentals"


def channel_axis(shape, channels) -> int:
    """HWC/NHWC put channels last; CHW puts them first and NCHW second. Pick by shape."""
    if len(shape) == 4:
        return 1 if shape[1] == channels else -1
    return 0 if shape[0] == channels and shape[-1] != channels else -1


def standardize(numpy, img, mean, std):
    """One call for HWC and NCHW: the stats are reshaped onto the channel axis."""
    x = numpy.asarray(img, dtype=numpy.float32) / 255.0
    axis = channel_axis(x.shape, len(mean))
    form = [1] * x.ndim
    form[axis] = len(mean)
    return (x - numpy.reshape(mean, form)) / numpy.reshape(std, form)


def destandardize(numpy, x, mean, std):
    """The inverse, rounding rather than truncating on the way back to uint8."""
    axis = channel_axis(x.shape, len(mean))
    form = [1] * x.ndim
    form[axis] = len(mean)
    y = (x * numpy.reshape(std, form) + numpy.reshape(mean, form)) * 255.0
    return numpy.clip(numpy.rint(y), 0, 255).astype(numpy.uint8)


def ramp(numpy):
    """Every uint8 value once per channel — 'any uint8 image', as small as that gets."""
    return numpy.arange(256, dtype=numpy.uint8).reshape(1, 256, 1).repeat(3, axis=2)


def spread(numpy, a, b) -> tuple:
    gap = numpy.abs(a.astype(int) - b.astype(int))
    return int(gap.max()), int((gap > 0).sum()), int(gap.size)


def broadcast_probe(numpy, mean, std, shape) -> str:
    """What the naive `(x - mean) / std` does at a given layout."""
    try:
        out = (numpy.zeros(shape, dtype=numpy.float32) - mean) / std
    except ValueError as exc:
        return f"ValueError: {' '.join(str(exc).split())[:52]}"
    return f"returns {out.shape}"


def solve():
    numpy = parity.try_numpy()
    ref = parity.load_reference(PHASE, LESSON, "main")
    mean, std = ref.IMAGENET_MEAN, ref.IMAGENET_STD
    bars = ramp(numpy)
    mine = destandardize(numpy, standardize(numpy, bars, mean, std), mean, std)
    theirs = ref.deprocess_imagenet(ref.preprocess_imagenet(bars))
    hwc = standardize(numpy, bars, mean, std)
    batch = numpy.stack([ref.hwc_to_chw(bars)] * 2)
    nchw = standardize(numpy, batch, mean, std)
    return {"mine": spread(numpy, mine, bars), "theirs": spread(numpy, theirs, bars),
            "rounded": spread(numpy, destandardize(
                numpy, ref.preprocess_imagenet(bars), mean, std).transpose(1, 2, 0), bars),
            "shapes": (hwc.shape, nchw.shape),
            "per_channel": tuple(", ".join(f"{v:.4f}" for v in row) for row in
                                 (hwc.reshape(-1, 3).mean(axis=0), nchw.mean(axis=(0, 2, 3)))),
            "naive": {"HWC (256, 3)": broadcast_probe(numpy, mean, std, (1, 256, 3)),
                      "NCHW (2, 3, 8, 8)": broadcast_probe(numpy, mean, std, (2, 3, 8, 8)),
                      "NCHW (2, 3, 8, 3)": broadcast_probe(numpy, mean, std, (2, 3, 8, 3))}}


def verify(result):
    mine, theirs, rounded = result["mine"], result["theirs"], result["rounded"]
    naive, hwc_shape, nchw_shape = result["naive"], *result["shapes"]
    return [
        practice.Check("ANSWER: a correct pair round-trips exactly, not merely within 1",
                       mine[0] == 0,
                       f"over a ramp carrying all 256 uint8 values ({mine[2]} samples), "
                       f"`destandardize(standardize(x))` differs from x by at most {mine[0]} — "
                       f"the exercise allows 1 and the answer is 0"),
        practice.Check("FINDING: the lesson's own pair scores the 1 the exercise allows, on 18% "
                       "of values",
                       theirs[0] == 1 and theirs[1] > 100,
                       f"`deprocess_imagenet(preprocess_imagenet(x))` on the same ramp is off by "
                       f"{theirs[0]} on {theirs[1]} of {theirs[2]} samples "
                       f"({100 * theirs[1] / theirs[2]:.0f}%). It passes the exercise's bar, and "
                       f"the bar is set exactly where its defect is"),
        practice.Check("MECHANISM: the defect is one truncating cast, not float32",
                       rounded[0] == 0,
                       f"`deprocess_imagenet` ends in `clip(x * 255.0, 0, 255).astype(uint8)`, "
                       f"which truncates: 0.9999 becomes 0. Rounding the same float32 pipeline "
                       f"instead — the lesson's own CHW output through the inverse above — takes "
                       f"the error to {rounded[0]} on every one of the {rounded[2]} samples, so "
                       f"the precision was never the problem"),
        practice.Check("MECHANISM: 'the same call' is a broadcasting problem, and NCHW is where "
                       "it bites",
                       "ValueError" in naive["NCHW (2, 3, 8, 8)"]
                       and "returns" in naive["NCHW (2, 3, 8, 3)"],
                       "the naive `(x - mean) / std` with mean of shape (3,) — "
                       + "; ".join(f"{k}: {v}" for k, v in naive.items())
                       + ". NumPy aligns from the right, so a real NCHW batch raises — and one "
                       "whose width happens to be 3 does not, returning a silently wrong answer"),
        practice.Check("CONTROL: picking the axis by shape makes the two layouts agree",
                       result["per_channel"][0] == result["per_channel"][1],
                       f"the same call returns {hwc_shape} for HWC and {nchw_shape} for NCHW, and "
                       f"the per-channel means agree exactly: {result['per_channel'][0]} against "
                       f"{result['per_channel'][1]}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
