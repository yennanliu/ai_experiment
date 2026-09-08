"""Exercise 1 — circularity by epoch.

    **(Easy)** Train the DCGAN above on the synthetic circle dataset and save a
    grid of 16 samples at the end of each epoch. By which epoch do the generated
    circles become clearly circular?

Reading of the exercise: "clearly circular" has to be given a number before the
question means anything, so each epoch's samples are scored two ways -- the
fraction of the frame that is foreground (the real circles fill 0.198 of it) and
the IoU between that foreground and the equal-area disc at its centroid (the real
circles score 0.993). Read that way the question turns out to rest on a false
premise: there is no such epoch. The lesson's own five epochs end with the
generator painting the entire 32x32 frame, and by epoch 25 the blob is the right
*size* while its outline is still nowhere near a circle. The IoU alone would lie
about this -- a solid frame scores 0.91 against its own equal-area disc -- which
is why both numbers are reported together. Training is the lesson's own main()
configuration (feat=32, 400 images, batch 32, Adam 2e-4/(0.5, 0.999), spectral-norm
D) run for 26 epochs instead of 5 so the transition is inside the window; that is
~18 s on a CI core. Seeds 1 and 2 were run offline and cross the half-frame mark
at epochs 10 and 11 against this run's 13, ending at disc-IoU 0.690 and 0.723
against 0.727 -- the answer is a window, not an epoch.
"""

from __future__ import annotations

import math
import pathlib
import tempfile

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "09-image-generation-gans"

Z_DIM, FEAT, NUM, BATCH, EPOCHS, SIDE, GRID, EVAL_N, EVAL_SEED = 64, 32, 400, 32, 26, 32, 4, 32, 1234
HALF, THRESHOLD = 0.5, 0.175    # "no longer fills the frame"; bg is -1, every colour >= -0.3

honest_ious = lambda hist: [h[0] for h in hist if h[1] < HALF]                   # noqa: E731
crossing = lambda areas: next((i for i, a in enumerate(areas) if a < HALF), EPOCHS)   # noqa: E731
trace = lambda values: " ".join(f"{v:.2f}" for v in values)                      # noqa: E731


def shape_stats(torch, imgs):
    """(n, 3, 32, 32) in [0, 1] -> mean disc-IoU and mean foreground area fraction."""
    yy, xx = torch.meshgrid(torch.arange(SIDE).float(), torch.arange(SIDE).float(), indexing="ij")
    mask = (imgs.max(1).values > THRESHOLD).float()
    area = mask.sum((1, 2)).clamp(min=1.0)
    cx, cy = (mask * xx).sum((1, 2)) / area, (mask * yy).sum((1, 2)) / area
    disc = (((xx - cx[:, None, None]) ** 2 + (yy - cy[:, None, None]) ** 2)
            < (area / math.pi)[:, None, None]).float()
    iou = (mask * disc).sum((1, 2)) / (mask + disc - mask * disc).sum((1, 2)).clamp(min=1)
    return float(iou.mean()), float(mask.mean())


def save_grid(Image, np, imgs, path):
    """The exercise's deliverable: 16 samples tiled 4x4 into one PNG."""
    frames = (imgs[:GRID * GRID].permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    tiled = frames.reshape(GRID, GRID, SIDE, SIDE, 3).transpose(0, 2, 1, 3, 4)
    Image.fromarray(tiled.reshape(GRID * SIDE, GRID * SIDE, 3)).save(path)
    return path


def train(torch, np, Image, ref, loader_cls, dataset_cls, real, out_dir):
    """The lesson's main() loop, plus a grid and a score after every epoch."""
    torch.manual_seed(0)
    G = ref.Generator(z_dim=Z_DIM, img_channels=3, feat=FEAT)
    D = ref.Discriminator(img_channels=3, feat=FEAT, use_sn=True)
    opt_g, opt_d = (torch.optim.Adam(m.parameters(), lr=2e-4, betas=(0.5, 0.999)) for m in (G, D))
    loader = loader_cls(dataset_cls(real), batch_size=BATCH, shuffle=True)
    history, grids = [], []
    for epoch in range(EPOCHS):
        for (batch,) in loader:
            ref.train_step(G, D, batch, torch.randn(batch.size(0), Z_DIM), opt_g, opt_d, "cpu")
        with torch.random.fork_rng():               # the lesson's sampler, same noise every epoch
            torch.manual_seed(EVAL_SEED)
            samples = ref.sample(G, n=EVAL_N, z_dim=Z_DIM)
        grids.append(save_grid(Image, np, samples, out_dir / f"epoch{epoch:02d}.png"))
        history.append(shape_stats(torch, samples) + (float(samples.std(0).mean()),))
    return history, grids


def solve():
    try:
        import numpy as np
        import torch
        from PIL import Image
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    real = ref.synthetic_circles(num=NUM)
    out_dir = pathlib.Path(tempfile.mkdtemp(prefix="dcgan_grids_"))
    history, grids = train(torch, np, Image, ref, DataLoader, TensorDataset, real, out_dir)
    return {"history": history, "dir": str(out_dir), "grids": [str(p) for p in grids],
            "sizes": [Image.open(p).size for p in (grids[0], grids[-1])],
            "real": shape_stats(torch, (real + 1) / 2) + (float(((real + 1) / 2).std(0).mean()),)}


def verify(result):
    hist, (real_iou, real_area, real_std) = result["history"], result["real"]
    areas, honest = [h[1] for h in hist], honest_ious(hist)
    lesson_end, last, flat_iou, crossed = hist[4], hist[-1], hist[0][0], crossing(areas)
    return [
        practice.Check(
            "ANSWER: there is no such epoch -- the outline never gets close to a circle",
            max(honest) < 0.85 * real_iou,
            f"once the blob stops filling the frame its disc-IoU peaks at {max(honest):.3f} over "
            f"{len(honest)} epochs and ends at {last[0]:.3f}, against {real_iou:.3f} real: {trace(honest)}"),
        practice.Check(
            "FINDING: at the lesson's own 5 epochs the generator is still painting the whole frame",
            lesson_end[1] > 0.85 and areas[0] > 0.9,
            f"after epoch 4 -- the last one main() runs -- {lesson_end[1]:.1%} of every 32x32 frame is over "
            f"the background against {real_area:.1%} for real data (epoch 0: {areas[0]:.1%})"),
        practice.Check(
            "ANSWER: the milestone that is reachable is size, and it lands mid-run",
            5 <= crossed <= 22 and abs(last[1] - real_area) < 0.3 * abs(areas[0] - real_area),
            f"the painted area first drops below half the frame at epoch {crossed} and ends at "
            f"{last[1]:.3f}, {abs(last[1] - real_area):.3f} from the real {real_area:.3f} against "
            f"{abs(areas[0] - real_area):.3f} at epoch 0. Area by epoch: {trace(areas)}"),
        practice.Check(
            "MECHANISM: IoU alone would have answered 'epoch 0', which is why area comes with it",
            flat_iou > 0.85 and real_iou - flat_iou < 0.15,
            f"a wholly-foreground frame scores {flat_iou:.3f} against its own equal-area disc, within "
            f"{real_iou - flat_iou:.3f} of the real {real_iou:.3f}: a disc of 1024 pixels covers most of a "
            "32x32 square, so a shape score means nothing until the size is right"),
        practice.Check(
            "CONTROL: the samples are less varied than the data they are imitating",
            last[2] < 0.85 * real_std,
            f"per-pixel standard deviation over {EVAL_N} samples from fixed noise is {last[2]:.3f} at the "
            f"last epoch against {real_std:.3f} over {NUM} real images, {real_std / last[2]:.1f}x wider"),
        practice.Check(
            "CONTROL: one grid per epoch really was written, and reads back at 128x128",
            len(result["grids"]) == EPOCHS and set(result["sizes"]) == {(GRID * SIDE, GRID * SIDE)},
            f"{len(result['grids'])} PNGs of {GRID}x{GRID} samples in {result['dir']}; the first and last "
            f"re-open at {result['sizes']}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
