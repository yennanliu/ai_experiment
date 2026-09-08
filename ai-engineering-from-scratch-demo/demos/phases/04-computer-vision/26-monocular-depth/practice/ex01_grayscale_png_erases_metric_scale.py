"""Exercise 1 — grayscale png erases metric scale.

    **(Easy)** Run Depth Anything V2 on any 10 images of your desk. Save depth as
    grayscale PNGs and inspect. Identify one object whose predicted depth looks
    wrong and explain why the monocular cues failed.

Reading of the exercise: there is no Depth Anything V2 here -- `depth_anything_v2`,
`transformers` and `timm` all raise ModuleNotFoundError, and no weights are
downloaded -- so the only "prediction" the lesson ships is `main()`'s
`pred = gt + 0.4 * randn`, ground truth plus Gaussian noise. That map contains no
monocular cue, so nothing in it can fail for a monocular reason. The sharper
question is the one the exercise's own method hides: the prescribed medium, an
8-bit grayscale PNG, is normalised per image, and per-image min-max is *exactly*
the affine family `align_scale_shift` fits. So the encoding is invariant by
construction to the one ambiguity monocular depth actually has. Ten depth maps
are saved here -- five metric rescalings of one scene, five with the object at a
different distance -- and the five rescalings, spanning 1.00-4.96 m to
8.00-55.50 m, come out as a single byte-identical PNG. "Inspect and identify an
object whose depth looks wrong" cannot be answered from that file.

Structure: `status` records one absent package as a measurement, an
`importlib.import_module` that returns `ModuleNotFoundError`; the other two
solutions import it from here rather than repeating it. `encode` does the real
PNG round trip through PIL and returns the file bytes and the decode; `blobs`
collects those bytes so files can be compared directly rather than through a
hash, and `files` gathers every PNG measurement. `grid` rebuilds the object mask
and the background ramp from `synthetic_depth`'s own predicate, so its 2.0 m
patch can be scored; `scene` moves that patch to a chosen distance; `extent` and
`probe` report a map's metric range and the depth a mid-gray pixel means in it.
"""

from __future__ import annotations

import importlib
import io

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "26-monocular-depth"
SIZE, NOISE, MID, STACK = 96, 0.4, 128, ("depth_anything_v2", "transformers", "timm")
GAINS = ((1.0, 0.0), (3.0, 0.7), (0.25, 10.0), (12.0, -4.0), (2.0, 0.0))
DEPTHS = (2.0, 1.5, 2.5, 3.4, 1.0)

blobs = lambda maps: [encode(m)[0] for m in maps]                                         # noqa: E731
extent = lambda dmap: f"{dmap.min():.2f}-{dmap.max():.2f} m"                              # noqa: E731
probe = lambda dmap: float(dmap.min()) + (MID / 255.0) * float(dmap.max() - dmap.min())   # noqa: E731
scene = lambda np, base, depth: np.where(grid(np)[0], depth, base).astype(np.float32)     # noqa: E731
files = lambda maps, moved: {"extents": [extent(m) for m in maps], "bytes": len(blobs(maps)[0]),
                             "gray": [probe(m) for m in maps], "moved": len(set(blobs(moved))),
                             "rescaled": len(set(blobs(maps)))}                           # noqa: E731


def status(name) -> str:
    try:
        importlib.import_module(name)
    except ImportError as exc:                  # pragma: no cover - none of these is installed
        return type(exc).__name__
    return "present"


def grid(np):
    rows, cols = np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing="ij")
    mask = (np.abs(cols - SIZE / 2) < SIZE / 6) & (np.abs(rows - SIZE * 0.6) < SIZE / 6)
    return mask, (1.0 + (rows / SIZE) * 4.0).astype(np.float32)


def encode(depth):
    import numpy as np
    from PIL import Image
    low, span = float(depth.min()), float(depth.max() - depth.min())
    gray = np.round((depth - low) / span * 255.0).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(gray, mode="L").save(buffer, format="PNG")
    return buffer.getvalue(), gray.astype(np.float32) / 255.0 * span + low


def solve():
    try:
        import numpy as np
        import torch
        from PIL import Image                   # noqa: F401 - encode() needs it too
    except ImportError as exc:                  # pragma: no cover - T1 needs torch and pillow
        raise practice.Skip(f"needs torch+pillow: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    base = ref.synthetic_depth(SIZE)
    ground = torch.from_numpy(base)
    noisy = ground + NOISE * torch.randn_like(ground)
    decoded = torch.from_numpy(encode(noisy.numpy())[1])
    mask, behind = [torch.from_numpy(part) for part in grid(np)]
    span = float(noisy.max() - noisy.min())
    rescaled = [(gain * base + bias).astype(np.float32) for gain, bias in GAINS]
    return dict(files(rescaled, [scene(np, base, depth) for depth in DEPTHS]),
                missing={name: status(name) for name in STACK}, span=span, step=span / 255.0, pixels=int(mask.sum()),
                raw=[ref.abs_rel_error(pred, ground) for pred in (noisy, decoded)],
                fit=[ref.abs_rel_error(ref.align_scale_shift(pred, ground), ground)
                     for pred in (noisy, decoded)],
                mean=float(noisy[mask].mean()), behind=float(behind[mask].mean()),
                flipped=float((noisy[mask] > behind[mask]).float().mean()))


def verify(result):
    missing, gray, fit, raw = result["missing"], result["gray"], result["fit"], result["raw"]
    return [
        practice.Check(
            "ANSWER: there is no model to run, so no monocular cue can fail",
            all(v == "ModuleNotFoundError" for v in missing.values()),
            "importing the stack gives " + ", ".join(f"{k} {v}" for k, v in missing.items())
            + f", and no weights are fetched. main()'s only 'prediction' is `gt + {NOISE} * randn`, which "
            f"reads no image: its errors are Gaussian, and its mean over the object is "
            f"{result['mean']:.4f} m against a true 2.0 -- unbiased, as no monocular predictor is"),
        practice.Check(
            "ANSWER: no object can be called wrong from the PNG -- 5 metric scenes give 1 file",
            result["rescaled"] == 1,
            f"five depth maps spanning {result['extents'][0]}, {result['extents'][2]} and "
            f"{result['extents'][3]} encode to {result['rescaled']} distinct {result['bytes']}-byte PNG: "
            "a 55-metre and a 5-metre scene are one file, so inspecting it cannot see absolute depth"),
        practice.Check(
            "MECHANISM: min-max normalisation is exactly the affine family the lesson fits out",
            result["moved"] == len(DEPTHS),
            "(d - min) / (max - min) cancels any a*d + b with a > 0 -- the two parameters "
            f"`align_scale_shift` solves for. Non-affine edits survive: the object at {DEPTHS} m gives "
            f"{result['moved']} distinct PNGs of {len(DEPTHS)}. Scale and shift go, everything else stays"),
        practice.Check(
            "FINDING: the same gray level means a different distance in every saved file",
            max(gray) / min(gray) > 10,
            f"pixel value {MID} decodes to " + ", ".join(f"{v:.2f}" for v in gray)
            + f" m across the five images -- a {max(gray) / min(gray):.1f}x spread. Ten PNGs of ten desk "
            "photos are ten incomparable scales, so 'this looks too far' is about one file's normalisation"),
        practice.Check(
            "CONTROL: the lesson's own metrics are just as blind, so they cannot flag the loss",
            abs(fit[0] - fit[1]) < 1e-3 and result["step"] < 0.03,
            f"the PNG round trip moves aligned absRel {fit[0]:.6f} -> {fit[1]:.6f} and raw absRel "
            f"{raw[0]:.6f} -> {raw[1]:.6f}. The only measurable cost is quantisation, {result['step']:.5f}"
            f" m per level over {result['span']:.3f} m -- the expensive loss is where no metric looks"),
        practice.Check(
            "FINDING: what looks wrong here is noise, and it is a half-percent effect",
            0.0 < result["flipped"] < 0.02,
            f"the object covers {result['pixels']} pixels at a true 2.0 m against a ramp averaging "
            f"{result['behind']:.3f} m there -- {NOISE} sigma of noise on a {result['behind'] - 2.0:.3f} "
            f"m contrast, and {result['flipped']:.2%} of its pixels still read farther than the ramp "
            "behind them. That is the only wrong depth here, and no monocular cue explains it"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
