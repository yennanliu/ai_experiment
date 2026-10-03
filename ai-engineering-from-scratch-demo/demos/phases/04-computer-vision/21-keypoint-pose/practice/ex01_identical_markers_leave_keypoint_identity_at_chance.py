"""Exercise 1 — identical markers leave keypoint identity at chance.

    **(Easy)** Train the tiny keypoint model on the synthetic 4-point dataset.
    Report mean L2 error between predicted and true keypoints after 200 steps.

Reading of the exercise: "the tiny keypoint model" and "the synthetic 4-point
dataset" are the lesson's own `TinyKeypointNet` and `make_synthetic_sample`, and
the lesson's own `main()` already trains them for exactly 200 steps and prints
the error, so the ANSWER is that number, read from `main()` itself. The rest
asks why the number is what it is. Every keypoint is drawn as the same 4x4 black
square, in random order, so nothing in the image says which square is keypoint 0;
the same net is therefore retrained on more evaluation samples to see what the
heatmaps hold, and once more with each keypoint drawn in its own colour.

**ANSWER: 12.458 px** after 200 steps (`main()`, seed 0, 8 evaluation images).
On a 64x64 canvas that is a fifth of the image, not a localisation error.

**FINDING: the error is identity, not localisation.** Retrained the same way and
scored on 64 images, the ordered error is **14.59 px**, near the **17.03 px** a
uniform pick among the four squares would give -- yet a predicted keypoint lies
on average **0.51 px** from *some* square. Each heatmap peaks at **0.26**, the
MSE-optimal answer to "one of these four is mine": a quarter of the target's
height on every square. The net finds the squares; the data never says whose
they are.

**FINDING: the `F.interpolate` in `main()` does nothing.** `TinyKeypointNet`
already returns 64x64 heatmaps (two stride-2 convs, two stride-2 transposed
convs), so the "upsample pred to full resolution" step changes them by **0.0**.

**CONTROL: give the keypoints an identity and the same net solves it.** Drawing
keypoint k in its own colour, same net, same 200 steps, same seed: **0.84 px**,
with heatmap peaks at **0.74**.

Structure: `coloured` redraws the lesson's sample with per-keypoint colours;
`train` runs the lesson's training loop and scores 64 held-out images; `solve`
also runs `main()` itself and reads its printed error.
"""

from __future__ import annotations

import contextlib
import io
import re

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "21-keypoint-pose"
STEPS, BATCH, EVAL = 200, 16, 64
COLOURS = ((0, 1, 1), (1, 0, 1), (1, 1, 0), (0, 0, 0))


def coloured(np, ref, rng):
    """The lesson's sample, with keypoint k drawn in COLOURS[k] instead of black."""
    img, hms, kps = ref.make_synthetic_sample(rng=rng)
    img = np.ones_like(img)
    for k, (cx, cy) in enumerate(kps.astype(int)):
        img[:, cy - 2 : cy + 2, cx - 2 : cx + 2] = np.array(COLOURS[k], "float32")[:, None, None]
    return img, hms, kps


def train(torch, np, ref, sample):
    """The lesson's loop (seed 0, Adam 3e-3, MSE), scored on EVAL held-out images."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = ref.TinyKeypointNet(num_keypoints=4)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    stack = lambda b, i: torch.from_numpy(np.stack([s[i] for s in b]))
    for _ in range(STEPS):
        batch = [sample(rng) for _ in range(BATCH)]
        loss = torch.nn.functional.mse_loss(model(stack(batch, 0)), stack(batch, 1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    batch = [sample(rng) for _ in range(EVAL)]
    with torch.no_grad():
        pred = model.eval()(stack(batch, 0))
    gt, coords = stack(batch, 2), ref.heatmap_to_coords(pred)
    up = torch.nn.functional.interpolate(pred, size=(64, 64), mode="bilinear", align_corners=False)
    return {
        "l2": float((coords - gt).norm(dim=-1).mean()),
        "nearest": float((coords[:, :, None] - gt[:, None]).norm(dim=-1).min(-1).values.mean()),
        "chance": float((gt[:, :, None] - gt[:, None]).norm(dim=-1).mean()),
        "peak": float(pred.amax((-1, -2)).mean()),
        "upsample": float((up - pred).abs().max()),
    }


def solve():
    try:
        import numpy as np
        import torch
    except ImportError as exc:  # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    torch.set_num_threads(2)
    ref = parity.load_reference(PHASE, LESSON, "main")
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        ref.main()
    found = re.findall(r"\(argmax\):\s+([\d.]+)", printed.getvalue())
    return {
        "main": float(found[0]),
        "plain": train(torch, np, ref, lambda rng: ref.make_synthetic_sample(rng=rng)),
        "colour": train(torch, np, ref, lambda rng: coloured(np, ref, rng)),
    }


def verify(result):
    plain, colour = result["plain"], result["colour"]
    return [
        practice.Check(
            "ANSWER: main() reports a mean L2 error of about 12 px after 200 steps",
            5.0 < result["main"] < 25.0,
            f"the lesson's own main() prints {result['main']:.3f} px (seed 0, 8 images) -- a "
            "fifth of the 64-pixel canvas, which is not a localisation error",
        ),
        practice.Check(
            "FINDING: the error is identity, not localisation",
            plain["l2"] > 8 and plain["nearest"] < 1.5 and abs(plain["peak"] - 0.25) < 0.08,
            f"over {EVAL} images the ordered error is {plain['l2']:.2f} px against "
            f"{plain['chance']:.2f} px for a uniform pick among the four squares, yet every "
            f"prediction lies {plain['nearest']:.2f} px from some square on average. Heatmaps "
            f"peak at {plain['peak']:.2f}, the MSE-optimal quarter height for 'one of these four'",
        ),
        practice.Check(
            "FINDING: main()'s F.interpolate to full resolution is a no-op",
            plain["upsample"] == 0.0,
            "TinyKeypointNet already returns 64x64 heatmaps, so the bilinear resize changes "
            f"them by {plain['upsample']}",
        ),
        practice.Check(
            "CONTROL: colour-coded keypoints are learned by the same net in the same 200 steps",
            colour["l2"] < 2.0 and colour["peak"] > 0.5,
            f"drawing keypoint k in its own colour gives {colour['l2']:.2f} px and peaks of "
            f"{colour['peak']:.2f}; the model was never the bottleneck, the labels were",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
