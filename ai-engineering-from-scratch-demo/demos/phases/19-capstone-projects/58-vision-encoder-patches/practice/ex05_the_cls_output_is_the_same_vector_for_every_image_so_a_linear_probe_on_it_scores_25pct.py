"""Exercise 5 -- the CLS output is the same vector for every image, so a linear probe on it scores 25%.

    Train the front end as a frozen feature extractor on a 4-class synthetic shape dataset (circles, squares, triangles, stars). The CLS token output should linearly separate.

Reading of the exercise: "train ... as a frozen feature extractor" is read
as the standard linear probe. The lesson's `VisionFrontEnd` (default ViT-B/16
config, seed 0) is frozen, and a logistic regression (sklearn, lbfgs) is
trained on its outputs. The dataset is 400 training and 200 test 224x224
images drawn with PIL: one white filled circle, square, triangle or 5-point
star on black, with random centre, radius 30-70 px and rotation, and balanced
classes. The probe is fitted on the CLS token, as the exercise says. For
comparison it is also fitted on the mean of the 196 patch tokens and on the
raw mean patch (the 768 pixel values the projection sees).

**ANSWER: no -- the CLS output does not separate, because it does not depend
on the image.** Across all 600 images the CLS row of the output differs by
0.0: it is exactly the learned `cls_token`, since its position row is zero
and nothing in the front end mixes patches into it. The probe scores 25.0%
train and 25.0% test, which is chance. The CLS token only becomes a summary
once attention (the next lesson's transformer) lets it read the patches.

**FINDING: patch tokens carry shape, but no more than the raw pixels do.** Mean-pooled patch tokens give
42.5% test accuracy (44.5% train), well above chance and far from separable.
The raw mean patch gives 42.0% test (48.0% train). The mean of the patch
tokens is one linear map of the mean patch, so both probes see the same
information, and the half-point gap comes from the probe's L2 penalty acting
on different scales.

**FINDING: training the front end through CLS cannot fix it.** One
cross-entropy backward pass from a linear head on CLS leaves the patch Conv2d's
weight gradient exactly zero (max |grad| 0.0) while the CLS token's gradient
is not. Only `cls_token` would learn, and one vector cannot tell 4 classes
apart. The lesson's "CLS token broadcasts across batch dim without leakage"
test passes for the same reason: the CLS row is a constant.

Structure: `render()` draws one shape; `probe()` fits and scores one feature
set; `solve()` runs the probes and the gradient check.
"""

from __future__ import annotations

import math

from harness import parity, practice

try:
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from sklearn.linear_model import LogisticRegression
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch, pillow, sklearn: uv sync --extra vision ({exc})") from None
torch.set_num_threads(1)

PHASE, LESSON = "19-capstone-projects", "58-vision-encoder-patches"
SHAPES, SIZE = ("circle", "square", "triangle", "star"), 224


def outline(kind, cx, cy, r, rot):
    if kind == "star":
        pts = [(r if k % 2 == 0 else 0.4 * r, rot + k * math.pi / 5) for k in range(10)]
    else:
        n = {"square": 4, "triangle": 3}[kind]
        pts = [(r, rot + k * 2 * math.pi / n) for k in range(n)]
    return [(cx + a * math.cos(t), cy + a * math.sin(t)) for a, t in pts]


def render(kind, rng):
    img = Image.new("RGB", (SIZE, SIZE))
    draw = ImageDraw.Draw(img)
    r = rng.uniform(30, 70)
    cx, cy = rng.uniform(r, SIZE - r, 2)
    rot = rng.uniform(0, 2 * math.pi)
    if kind == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255))
    else:
        draw.polygon(outline(kind, cx, cy, r, rot), fill=(255, 255, 255))
    return np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255


def dataset(n, seed):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % len(SHAPES)
    return torch.from_numpy(np.stack([render(SHAPES[k], rng) for k in y])), y


def probe(train, test, ytr, yte):
    clf = LogisticRegression(max_iter=2000).fit(train, ytr)
    return round(clf.score(train, ytr), 3), round(clf.score(test, yte), 3)


def cls_gradients(front, x, y):
    head = torch.nn.Linear(front.cfg.hidden, len(SHAPES))
    loss = torch.nn.functional.cross_entropy(head(front(x)[:, 0]), torch.from_numpy(y))
    loss.backward()
    return front.patch.proj.weight.grad.abs().max().item(), front.cls_token.grad.abs().max().item()


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.manual_seed(0)
    front = ref.VisionFrontEnd(ref.FrontEndConfig()).eval()
    (xtr, ytr), (xte, yte) = dataset(400, 0), dataset(200, 1)
    with torch.no_grad():
        ftr, fte = front(xtr), front(xte)
    both = torch.cat([ftr[:, 0], fte[:, 0]])
    pix = [x.unfold(2, 16, 16).unfold(3, 16, 16).mean((2, 3)).flatten(1) for x in (xtr, xte)]
    return {
        "cls_spread": (both - both[:1]).abs().max().item(),
        "cls_is_param": torch.equal(ftr[0, 0], front.cls_token.detach()[0, 0]),
        "cls": probe(ftr[:, 0].numpy(), fte[:, 0].numpy(), ytr, yte),
        "mean": probe(ftr[:, 1:].mean(1).numpy(), fte[:, 1:].mean(1).numpy(), ytr, yte),
        "pixels": probe(pix[0].numpy(), pix[1].numpy(), ytr, yte),
        "grads": cls_gradients(front, xtr[:32], ytr[:32]),
        "balanced": np.bincount(ytr).tolist(),
    }


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: no -- the CLS output does not separate, because it does not depend on the image",
            r["cls_spread"] == 0.0 and r["cls_is_param"] and r["cls"] == (0.25, 0.25)
            and r["balanced"] == [100] * 4,
            f"CLS spread over 600 images {r['cls_spread']}, equals cls_token: "
            f"{r['cls_is_param']}; probe train/test {r['cls']}",
        ),
        practice.Check(
            "FINDING: patch tokens carry shape, but no more than the raw pixels do",
            r["mean"] == (0.445, 0.425) and r["pixels"] == (0.48, 0.42),
            f"mean-pooled tokens train/test {r['mean']}; raw mean patch {r['pixels']}",
        ),
        practice.Check(
            "FINDING: training the front end through CLS cannot fix it",
            r["grads"][0] == 0.0 and r["grads"][1] > 0,
            f"max |grad| patch Conv2d {r['grads'][0]}, cls_token {r['grads'][1]:.3g}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
