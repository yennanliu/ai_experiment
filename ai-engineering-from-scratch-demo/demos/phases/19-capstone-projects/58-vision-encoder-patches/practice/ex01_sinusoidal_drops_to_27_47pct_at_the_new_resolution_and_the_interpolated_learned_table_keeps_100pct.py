"""Exercise 1 -- sinusoidal drops to 27-47% at the new resolution, and the interpolated learned table keeps 100%.

    Replace the sinusoidal position with a learned `nn.Parameter` and compare the first-epoch loss on a tiny synthetic classification task. Learned positions win at fixed resolution; sinusoidal wins when you change resolution after training.

Reading of the exercise: the task has to need position, so it is: a 32x32
image with one bright 8x8 square in one of the 16 patch cells, labelled by its
quadrant (4 classes). Every bright patch has the same content, so only
position tells the classes apart. The model is the lesson's `VisionFrontEnd`
(patch 8, hidden 32) with one `nn.TransformerEncoderLayer` and a linear head
on CLS. For the learned variant the `pos_embed` buffer is swapped for an
`nn.Parameter`; no lesson code is copied. The ViT/timm convention initialises
it at std 0.02, the same init the lesson gives its CLS token, and unit std
(the scale of the sinusoidal entries) is run too. One epoch is 4,096 images in
batches of 32, Adam at 1e-3, seeds 0-4. "Change resolution after training"
means the same scenes rendered at 64x64 (an 8x8 grid) after 3 epochs of
training. The lesson's `PatchEmbed.forward` rejects any size but the
configured one, so the model is rebuilt at 64 px and the trained weights are
loaded into it. The learned table is bicubic-resized 4x4 -> 8x8 the way timm
does it, and the sinusoidal table is rebuilt for the new grid.

**ANSWER: learned wins epoch 1 only at unit init, and at the new resolution
learned wins.** Mean epoch-1 loss over 5 seeds: learned std 1.0 0.605,
sinusoidal 1.394, learned std 0.02 1.399. The last two are both still at
chance (ln 4 = 1.386), so at the conventional init the comparison is a tie.
After 3 epochs every model gets 100% at 32x32. At 64x64 the sinusoidal models
fall to 27.0-47.3% (chance is 25%), while every interpolated learned model
keeps 100%.

**FINDING: the lesson's resolution story is backwards on a task that needs
position.** `sinusoidal_2d` encodes the integer patch index. Row 2 meant the
bottom half on the 4x4 grid, and on the 8x8 grid it is in the top half. The
table is deterministic, but that does not make it resolution-invariant. The
doc says it "interpolates cleanly to grids the model never saw"; in fact it
extrapolates. Bicubic-resizing the sinusoidal table the same way as the
learned one gives 100%. What saves the learned table is the interpolation.

Structure: `scenes()` renders the task; `train()` fits one model; `moved()`
rebuilds the trained models at 64 px with the position table resized or
rebuilt.
"""

from __future__ import annotations

import math

from harness import parity, practice

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

PHASE, LESSON = "19-capstone-projects", "58-vision-encoder-patches"
P, H, N, SEEDS = 8, 32, 4096, range(5)


def scenes(n, seed, scale):
    gen = torch.Generator().manual_seed(seed)
    cell = torch.randint(0, 16, (n,), generator=gen)
    img, s = torch.rand(n, 3, 32 * scale, 32 * scale, generator=gen) * 0.2, P * scale
    for i, (r, c) in enumerate(zip(cell // 4, cell % 4)):
        img[i, :, r * s : (r + 1) * s, c * s : (c + 1) * s] += 0.8
    return img, (cell // 4 >= 2) * 2 + (cell % 4 >= 2)


def build(ref, image, seed, init=None):
    torch.manual_seed(seed)
    front = ref.VisionFrontEnd(ref.FrontEndConfig(image_size=image, patch_size=P, hidden=H))
    if init is not None:
        shape = front.pos_embed.shape
        del front.pos_embed
        front.pos_embed = nn.Parameter(torch.randn(shape) * init)
    enc = nn.TransformerEncoderLayer(H, 4, 64, dropout=0.0, batch_first=True)
    return nn.ModuleDict({"fe": front, "enc": enc, "head": nn.Linear(H, 4)})


def logits(m, x):
    return m["head"](m["enc"](m["fe"](x))[:, 0])


def train(ref, seed, init, epochs):
    m, (x, y) = build(ref, 32, seed, init), scenes(N, seed + 100, 1)
    opt, losses = torch.optim.Adam(m.parameters(), lr=1e-3), []
    for step in range(epochs * N // 32):
        i = step % (N // 32) * 32
        loss = F.cross_entropy(logits(m, x[i : i + 32]), y[i : i + 32])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return m, sum(losses[: N // 32]) / (N // 32)


def acc(m, data):
    with torch.no_grad():
        return round((logits(m, data[0]).argmax(1) == data[1]).float().mean().item(), 3)


def moved(ref, runs, interpolate, data):
    """64 px accuracy of each trained model, its table rebuilt or bicubic-resized 4x4 -> 8x8."""
    accs = []
    for (m, _), seed in zip(runs, SEEDS):
        big = build(ref, 64, seed)
        big.load_state_dict({k: v for k, v in m.state_dict().items() if "pos_embed" not in k}, False)
        if interpolate:
            pos = m["fe"].pos_embed.detach()
            grid = pos[:, 1:].reshape(1, 4, 4, H).permute(0, 3, 1, 2)
            up = F.interpolate(grid, size=(8, 8), mode="bicubic", align_corners=False)
            del big["fe"].pos_embed
            big["fe"].pos_embed = nn.Parameter(torch.cat([pos[:, :1], up.permute(0, 2, 3, 1).reshape(1, 64, H)], 1))
        accs.append(acc(big, data))
    return accs


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    small, large = scenes(512, 999, 1), scenes(512, 999, 2)
    sin, learned = ([train(ref, s, init, 3) for s in SEEDS] for init in (None, 1.0))
    tiny = [train(ref, s, 0.02, 1) for s in SEEDS]
    return {
        "loss": {k: round(sum(v[1] for v in runs) / 5, 3)
                 for k, runs in (("sin", sin), ("learned_002", tiny), ("learned_1", learned))},
        "small": [acc(m, small) for m, _ in sin + learned],
        "large": {"sin": moved(ref, sin, False, large), "learned_1": moved(ref, learned, True, large)},
        "sin_resized": moved(ref, sin, True, large),
        "doc": "interpolates cleanly" in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    r, loss, large = result, result["loss"], result["large"]
    return [
        practice.Check(
            "ANSWER: learned wins epoch 1 only at unit init, and at the new resolution learned wins",
            all([loss["learned_1"] < 0.8, min(r["small"]) == 1.0, max(large["sin"]) < 0.5,
                 abs(loss["sin"] - math.log(4)) < 0.03, abs(loss["learned_002"] - math.log(4)) < 0.03,
                 min(large["learned_1"]) == 1.0]),
            f"epoch-1 loss {loss}; 32 px accuracy of all 10 models {r['small']}; 64 px accuracy {large}",
        ),
        practice.Check(
            "FINDING: the lesson's resolution story is backwards on a task that needs position",
            all([r["doc"], min(r["sin_resized"]) == 1.0, max(large["sin"]) < 0.5]),
            f"rebuilt sinusoidal {large['sin']} vs bicubic-resized sinusoidal {r['sin_resized']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
