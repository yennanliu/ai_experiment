"""Exercise 2 — transposed conv checkerboard.

    **(Medium)** Replace the `nn.Upsample + conv` up-block with a
    `nn.ConvTranspose2d` up-block. Train both on the synthetic dataset and
    compare mIoU. Observe where checkerboard artifacts appear in the
    transposed-conv version.

Reading of the exercise: "a `nn.ConvTranspose2d` up-block" does not name a kernel
size, and that omission is the whole exercise. Odena et al.'s rule says the
checkerboard comes from uneven overlap when the kernel is not divisible by the
stride, so three kernels are built -- 2, 3 and 4 at stride 2 -- and the overlap
is measured exactly with uniform weights (1, 4 and 1). The rule turns out to
predict the wrong winner once the weights are *learned*: every transposed
variant, kernel 2 included, ends up with a strong fixed grid in its logits, while
the lesson's bilinear block has essentially none. "Where the artifacts appear" is
answered by splitting the Nyquist response into the part that is a constant
offset (a grid) and the part that tracks image content, and by asking whether it
concentrates on class boundaries or spreads over flat background. mIoU, the
comparison the exercise actually asks for, is shown to be blind to all of it.
Runs are scaled to a base-8 U-Net, 48 images, 8 epochs and two seeds for the
lesson's runtime budget; everything else here is exact.

How the pieces below fit together. `up_block_class` returns the replacement
block, built inside `solve` because `nn` only exists after the guarded import: a
`ConvTranspose2d(c, c, k, stride=2)` with `padding=(k-1)//2` and
`output_padding=k%2`, which doubles the side for every k, followed by the
lesson's own `DoubleConv` on the concatenation, so the only thing that differs
from the lesson's `Up` is how the upsampling is done. The lesson's `Up` also
carries an `F.interpolate` guard for sizes that do not halve evenly; at 64x64
with four poolings every stage divides exactly, so that branch never runs here
and is left out rather than shipped untested. `train` runs one variant at one
seed on the lesson's own U-Net, `combined_loss` and Adam, then measures the
period-2 (Nyquist) component of the held-out logits: `locked` is the share of it
that is a constant offset across every 2x2 block -- a fixed grid rather than
image content -- and `edge_ratio` compares that amplitude on blocks straddling a
class boundary against blocks that do not, where `spans` marks a block as
straddling exactly when its minimum and maximum labels differ. `overlap` drives
one transposed layer with weights of one and an input of ones and reports the
interior max over min, which is Odena's uneven-overlap number.

The file stays 20 lines over D14's 120-line target, and the exercise names four
separate deliverables that each cost code. The replacement up-block is 10 lines;
`train` is 25, of which 13 are the training run the exercise asks for and 9 are
the artifact measurement it also asks for; `overlap`, the uniform-weight
arithmetic that is supposed to predict the artifact and is shown not to, is 7.
Those three run over four variants and two seeds. The five checks are 42 lines,
19 of which are the measured numbers they quote.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "07-semantic-segmentation-unet"

SIZE, SAMPLES, SPLIT, CLASSES = 64, 60, 48, 3
BASE, EPOCHS, BATCH, LR, SEEDS = 8, 8, 8, 1e-3, (0, 1)
VARIANTS = ("upsample", 2, 3, 4)        # "upsample" is the lesson's own Up block
LABELS = {"upsample": "bilinear", 2: "k=2", 3: "k=3", 4: "k=4"}

stages = lambda: (("u1", BASE * 24, BASE * 8), ("u2", BASE * 12, BASE * 4),
                  ("u3", BASE * 6, BASE * 2), ("u4", BASE * 3, BASE))          # noqa: E731
column = lambda runs, kind, key: [r[key] for r in runs[kind]]                  # noqa: E731
means = lambda runs, key: {k: sum(column(runs, k, key)) / len(SEEDS) for k in VARIANTS}  # noqa: E731
every = lambda runs, kinds, key: [r[key] for k in kinds for r in runs[k]]      # noqa: E731
ranges = lambda runs, key: [max(column(runs, k, key)) - min(column(runs, k, key)) for k in VARIANTS]  # noqa: E731
fmt = lambda values: " ".join(f"{v:.3f}" for v in values)                      # noqa: E731
named = lambda table: ", ".join(f"{LABELS[k]} {table[k]:.3f}" for k in VARIANTS)   # noqa: E731
laps_of = lambda laps: ", ".join(f"k={k} {laps[k]:.1f}" for k in VARIANTS[1:])     # noqa: E731
swapped = lambda runs, params: (params[2] > params["upsample"]                    # noqa: E731
                               and len(set(every(runs, VARIANTS, "shape"))) == 1)


def up_block_class(torch, nn, ref):
    class TransposedUp(nn.Module):
        def __init__(self, in_c, out_c, kernel):   # the skip contributes out_c of the in_c
            super().__init__()
            self.up = nn.ConvTranspose2d(in_c - out_c, in_c - out_c, kernel, stride=2,
                                         padding=(kernel - 1) // 2, output_padding=kernel % 2)
            self.conv = ref.DoubleConv(in_c, out_c)
        def forward(self, x, skip):
            return self.conv(torch.cat([skip, self.up(x)], dim=1))
    return TransposedUp


def overlap(torch, nn, kernel):
    layer = nn.ConvTranspose2d(1, 1, kernel, stride=2, padding=(kernel - 1) // 2,
                               output_padding=kernel % 2, bias=False)
    with torch.no_grad():
        layer.weight.fill_(1.0)
        out = layer(torch.ones(1, 1, 16, 16))[0, 0, 4:28, 4:28]
    return float(out.max() / out.min())


def train(torch, ref, block, kind, seed, data, spans):
    xtr, ytr, xva, yva = data
    torch.manual_seed(seed)
    net = ref.UNet(3, CLASSES, base=BASE)
    if kind != "upsample":
        for name, in_c, out_c in stages():
            setattr(net, name, block(in_c, out_c, kind))
    opt, gen = torch.optim.Adam(net.parameters(), lr=LR), torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        net.train()
        for start in range(0, SPLIT, BATCH):
            batch = torch.randperm(SPLIT, generator=gen)[start:start + BATCH]
            opt.zero_grad()
            ref.combined_loss(net(xtr[batch]), ytr[batch], CLASSES)[0].backward()
            opt.step()
    with torch.no_grad():
        logits = net.eval()(xva)
    field = (logits[..., 0::2, 0::2] - logits[..., 0::2, 1::2]
             - logits[..., 1::2, 0::2] + logits[..., 1::2, 1::2]) / 4
    size, edge = field.abs(), spans.expand_as(field)
    return {"miou": float(ref.iou_per_class(logits, yva, CLASSES).nan_to_num(0).mean()),
            "params": sum(p.numel() for p in net.parameters()), "shape": tuple(logits.shape),
            "amplitude": float(size.mean() / logits.std()),
            "edge_ratio": float(size[edge].mean() / size[~edge].mean()),
            "locked": float(field.mean(dim=(0, 2, 3)).abs().mean() / size.mean())}


def solve():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    images, masks = ref.synthetic_segmentation(num_samples=SAMPLES, size=SIZE, seed=0)
    x, y = torch.from_numpy(images).permute(0, 3, 1, 2), torch.from_numpy(masks)
    data, grid = (x[:SPLIT], y[:SPLIT], x[SPLIT:], y[SPLIT:]), y[SPLIT:].unsqueeze(1).float()
    spans = (functional.max_pool2d(grid, 2) + functional.max_pool2d(-grid, 2)).abs() > 0
    block = up_block_class(torch, nn, ref)
    return {"runs": {kind: [train(torch, ref, block, kind, s, data, spans) for s in SEEDS]
                     for kind in VARIANTS},
            "overlap": {k: overlap(torch, nn, k) for k in VARIANTS[1:]}}


def verify(result):
    runs, laps = result["runs"], result["overlap"]
    miou, locked = means(runs, "miou"), means(runs, "locked")
    amplitude, spread = means(runs, "amplitude"), ranges(runs, "miou")
    params = {k: runs[k][0]["params"] for k in VARIANTS}
    edges, shape = {k: column(runs, k, "edge_ratio") for k in VARIANTS}, runs["upsample"][0]["shape"]
    return [
        practice.Check(
            "ANSWER: the swap works, and mIoU cannot rank the two up-blocks",
            swapped(runs, params) and max(spread) > max(miou.values()) - min(miou.values()),
            f"padding (k-1)//2 with output_padding k%2 doubles the side for every k, so all four variants "
            f"still map to {shape}, at {params['upsample']:,} parameters for bilinear and {params[2]:,} "
            f"(+{params[2] / params['upsample'] - 1:.1%}), {params[3]:,}, {params[4]:,} for k=2, 3, 4. Mean "
            f"mIoU over {len(SEEDS)} seeds {named(miou)} -- but the seeds differ by up to {max(spread):.3f} "
            f"within one variant against {max(miou.values()) - min(miou.values()):.3f} between the means"),
        practice.Check(
            "FINDING: every transposed variant grows a fixed grid in its logits; bilinear does not",
            min(every(runs, VARIANTS[1:], "locked")) > 0.15 > max(column(runs, "upsample", "locked")),
            "share of the Nyquist (period-2) response that is a constant offset rather than image content, "
            f"per run: bilinear {fmt(column(runs, 'upsample', 'locked'))}; k=2 {fmt(column(runs, 2, 'locked'))}; "
            f"k=3 {fmt(column(runs, 3, 'locked'))}; k=4 {fmt(column(runs, 4, 'locked'))} -- amplitude over "
            f"the logits' own standard deviation {amplitude['upsample']:.4f} -> {amplitude[2]:.4f} at k=2"),
        practice.Check(
            "ANSWER: the artifacts sit on flat background, which is what makes them visible",
            max(edges[2]) < 1.5 < 2.5 < min(edges["upsample"]),
            "ratio of Nyquist amplitude on 2x2 blocks straddling a class boundary to blocks that do not: "
            f"bilinear {fmt(edges['upsample'])} -- its high frequencies are the object outlines -- against "
            f"k=2 {fmt(edges[2])}, flat across the image, background included"),
        practice.Check(
            "MECHANISM: uniform-weight overlap is 4:1 at k=3 and exactly even at k=2 and k=4",
            laps[3] == 4.0 and laps[2] == laps[4] == 1.0,
            f"with weights of one and an input of ones, the interior output max/min runs {laps_of(laps)}: "
            "stride 2 divides kernels 2 and 4 evenly, leaving k=3 a 1/2/4 tiling"),
        practice.Check(
            "CONTROL: 'make the kernel divisible by the stride' does not remove the artifact",
            locked[2] > 0.6 and locked[2] > locked[3] and laps[2] == 1.0,
            f"k=2 has perfectly even overlap ({laps[2]:.1f}) and still ends with the largest fixed grid of "
            f"the four, {locked[2]:.3f} against {locked[3]:.3f} at the 4:1-overlap k=3 and "
            f"{locked['upsample']:.3f} for bilinear. At stride 2 a 2x2 kernel gives each output phase its "
            "own weight and shares none, so nothing ties the phases together once the weights are learned "
            "-- even overlap only ever described the untrained layer"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
