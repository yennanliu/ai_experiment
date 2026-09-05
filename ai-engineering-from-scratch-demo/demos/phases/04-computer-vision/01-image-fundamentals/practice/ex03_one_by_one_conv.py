"""Exercise 3 — a frozen 1x1 conv as grayscale, and which transforms that generalises to.

    **(Hard)** Take a 3-channel ImageNet-standardized tensor and run it through a
    1x1 conv that learns a weighted mixture of RGB into a single grayscale
    channel. Initialize the weights to `[0.299, 0.587, 0.114]`, freeze them, and
    verify the output matches your manual `rgb_to_grayscale` to within
    floating-point error. What other classical color-space transforms can be
    written as 1x1 convolutions?

Reading of the exercise: "matches to within floating-point error" is two claims, and checks
1-2 separate them — the conv matches the exact weighted sum, and does not match the lesson's
`rgb_to_grayscale`, which truncates to uint8. Check 3 takes "ImageNet-standardized" literally,
where the named weights stop being grayscale at all. Checks 4-5 answer the closing question by
exhibiting one transform that is a 1x1 conv and one that provably is not.
"""

from __future__ import annotations

from harness import parity, practice

try:
    import torch
except ImportError as exc:                       # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None
torch.set_num_threads(1)

PHASE, LESSON = "04-computer-vision", "01-image-fundamentals"
LUMA = (0.299, 0.587, 0.114)                     # the weights the exercise names
YCBCR = ((0.299, 0.587, 0.114), (-0.168736, -0.331264, 0.5), (0.5, -0.418688, -0.081312))


def frozen_conv(rows) -> torch.nn.Conv2d:
    """A 1x1 conv holding `rows` as its mixture, with the weights frozen as asked."""
    conv = torch.nn.Conv2d(3, len(rows), kernel_size=1, bias=False)
    with torch.no_grad():
        conv.weight.copy_(torch.tensor(rows, dtype=torch.float32).view(len(rows), 3, 1, 1))
    conv.weight.requires_grad_(False)
    return conv


def apply_conv(conv, chw):
    with torch.no_grad():
        return conv(torch.tensor(chw, dtype=torch.float32).unsqueeze(0))[0].numpy()


def affine_residual(numpy, mixed, gray) -> float:
    """How far conv(standardized) is from *any* affine function of true grayscale."""
    slope, offset = numpy.polyfit(gray.ravel(), mixed.ravel(), 1)
    return float(numpy.abs(slope * gray.ravel() + offset - mixed.ravel()).max())


def hsv_additivity(ref, numpy) -> float:
    """A linear map obeys f(a + b) = f(a) + f(b). HSV is where that is checked."""
    a = numpy.array([[[10, 200, 30]]], dtype=numpy.uint8)
    b = numpy.array([[[100, 50, 220]]], dtype=numpy.uint8)
    both = numpy.clip(a.astype(int) + b.astype(int), 0, 255).astype(numpy.uint8)
    return float(numpy.abs(ref.rgb_to_hsv(both) - (ref.rgb_to_hsv(a) + ref.rgb_to_hsv(b))).max())


def solve():
    numpy = parity.try_numpy()
    ref = parity.load_reference(PHASE, LESSON, "main")
    img = ref.synthetic_image(32, 48, seed=2)
    gray_conv, ycbcr_conv = frozen_conv([LUMA]), frozen_conv(YCBCR)
    raw = apply_conv(gray_conv, ref.hwc_to_chw(img))[0]
    exact = img.astype(numpy.float32) @ numpy.array(LUMA, dtype=numpy.float32)
    theirs = ref.rgb_to_grayscale(img)
    mixed = apply_conv(gray_conv, ref.preprocess_imagenet(img))[0]
    manual = numpy.tensordot(img.astype(numpy.float32),
                             numpy.array(YCBCR, dtype=numpy.float32).T, axes=([2], [0]))
    return {"frozen": sum(p.requires_grad for p in gray_conv.parameters()),
            "vs_exact": float(numpy.abs(raw - exact).max()),
            "vs_theirs": float(numpy.abs(raw - theirs.astype(numpy.float32)).max()),
            "low": int((theirs.astype(int) < numpy.rint(exact).astype(int)).sum()),
            "pixels": int(theirs.size),
            "ratio": (numpy.array(LUMA) / ref.IMAGENET_STD).round(4).tolist(),
            "residual": affine_residual(numpy, mixed, exact / 255.0),
            "ycbcr": float(numpy.abs(apply_conv(ycbcr_conv, ref.hwc_to_chw(img))
                                     - manual.transpose(2, 0, 1)).max()),
            "hsv": hsv_additivity(ref, numpy)}


def verify(result):
    return [
        practice.Check("ANSWER: the frozen 1x1 conv is the weighted mixture, to float32 epsilon",
                       result["frozen"] == 0 and result["vs_exact"] < 1e-4,
                       f"weights set to {LUMA} and `requires_grad_(False)` leaves "
                       f"{result['frozen']} parameters trainable; the output matches the exact "
                       f"sum `img @ w` to {result['vs_exact']:.2e} over "
                       f"{result['pixels']} pixels of values up to 255 — float32 has about 7 "
                       f"significant digits, so that is the arithmetic agreeing"),
        practice.Check("FINDING: it does not match the lesson's `rgb_to_grayscale`, which "
                       "truncates",
                       0.9 < result["vs_theirs"] < 1.0 and result["low"] > result["pixels"] / 3,
                       f"against the lesson's own function the gap reaches "
                       f"{result['vs_theirs']:.5f}, because it ends in `.astype(np.uint8)` — a "
                       f"truncation, not a round — so {result['low']} of {result['pixels']} "
                       f"pixels come out one level low. 'Within floating-point error' holds "
                       f"against the float sum and fails against the function the exercise names"),
        practice.Check("FINDING: on a *standardized* tensor these weights are not grayscale",
                       result["residual"] > 1e-3,
                       f"standardizing divides each channel by a different number, so the mixture "
                       f"the conv then applies is effectively w/std = {result['ratio']} — no "
                       f"longer proportional to {LUMA}. Fitting the best affine function of true "
                       f"grayscale to conv(standardized) still leaves {result['residual']:.4f} of "
                       f"residual, so the two are not even the same up to scale and offset. "
                       f"Standardize-then-mix and mix-then-standardize commute only when std is "
                       f"the same in every channel"),
        practice.Check("ANSWER: every *linear* colour transform is a 1x1 conv — RGB to YCbCr "
                       "matches exactly",
                       result["ycbcr"] == 0.0,
                       f"the 3x3 YCbCr matrix loaded as three 1x1 filters reproduces the manual "
                       f"`tensordot` to {result['ycbcr']:.1f} — bit for bit. A 1x1 conv *is* a "
                       f"per-pixel matrix multiply, so RGB to YIQ, RGB to XYZ, channel swaps and "
                       f"grayscale are all the same operation with different constants"),
        practice.Check("CONTROL: and HSV provably is not one",
                       result["hsv"] > 1.0,
                       f"a linear map obeys f(a + b) = f(a) + f(b); the lesson's own "
                       f"`rgb_to_hsv` misses that by {result['hsv']:.1f} on one pair of colours. "
                       f"It takes a max, a min and a divide by their difference, so no matrix — "
                       f"and therefore no 1x1 conv — can express it"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
