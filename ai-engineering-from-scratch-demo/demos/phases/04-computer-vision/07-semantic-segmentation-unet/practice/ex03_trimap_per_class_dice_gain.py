"""Exercise 3 — trimap per class dice gain.

    **(Hard)** Take a real segmentation dataset (Oxford-IIIT Pets, Cityscapes mini
    split, or a medical subset) and train the U-Net to within 2 IoU points of the
    `smp.Unet` reference. Report per-class IoU and identify which classes benefit
    most from adding Dice to the loss.

Reading of the exercise: two of the three things it names are out of reach here
and saying so is part of the answer. There is no network in this sandbox, so
Oxford-IIIT Pets cannot be downloaded, and `segmentation_models_pytorch` is not
in the `vision` dependency group, so there is no `smp.Unet` to be within 2 points
of. The real command and its cost are printed below. What is built instead keeps
the part of Pets that matters for the question: its three-class trimap --
background, object interior, and a thin border ring -- laid over *real*
photographic pixels, cropped from the two photographs scikit-learn ships, with
the lesson's own shapes supplying exact ground truth. The last clause is the one
that can be answered properly, and the honest way to answer it is not a training
delta: at this budget the seed-to-seed spread is nine times the 2-point tolerance
the exercise quotes and swamps every per-class difference. So "which classes
benefit most from adding Dice" is answered by decomposing where each term puts
its gradient, which is exact and needs no training at all. Runs are scaled to a
base-8 U-Net, 48 images, 10 epochs and three seeds for the runtime budget.

How the pieces below fit together. `trimap` builds the dataset: the lesson's own
`synthetic_segmentation` supplies the shapes, a RING-wide dilate-minus-erode band
around each shape becomes the border class, and the background is replaced with a
random 64x64 crop of one of scikit-learn's two photographs, so the pixels are
real while the labels stay exact. It returns the lesson's original flat-green
images alongside, purely so the two backgrounds' colour spread can be compared.
`train` is one run of the lesson's own U-Net and `combined_loss` at a given `lam`
-- 0 for cross-entropy alone, 1 for cross-entropy plus Dice -- reporting held-out
per-class IoU. `gradient_share` is the part that actually answers the exercise's
last clause: it backpropagates one loss term into the logits and reports what
fraction of the total gradient magnitude lands on pixels of each true class, so
the two terms' class weightings can be compared directly rather than inferred
from a training delta that the seed noise swamps.

The file stays 18 lines over D14's 120-line target, and the exercise names three
separate deliverables that each cost code. Building a dataset that is genuinely
Pets-shaped on real photographic pixels is 13 lines; the per-class IoU report
needs the 15-line training run executed at two loss settings and three seeds; the
gradient attribution that answers "which classes benefit most" is 6 more, and it
is the only one of the three that answers that clause at all. The five checks are
44 lines, 21 of which are the measured numbers they quote.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "07-semantic-segmentation-unet"

SIZE, SAMPLES, SPLIT, CLASSES = 64, 60, 48, 3
BASE, EPOCHS, BATCH, LR, SEEDS = 8, 10, 8, 1e-3, (0, 1, 2)
RING, INTERIOR, BORDER = 5, 1, 2         # a RING-wide dilate-minus-erode band, as in a Pets trimap
LAMBDAS = (0.0, 1.0)                     # the `lam` of the lesson's own combined_loss
NAMES = ("background", "interior", "border")
TOLERANCE = 0.02                         # the exercise's "2 IoU points"
REAL_RUN = ("full scale: `python train.py --data oxford-iiit-pet --arch unet --size 256 "
            "--epochs 60` is ~1.5 h on one A10G (~$0.40/h spot, ~$0.60 a run), and the "
            "smp.Unet(resnet34, imagenet) baseline it is meant to match is ~1 h more")

fmt = lambda values: " ".join(f"{v:.3f}" for v in values)                          # noqa: E731
per_class = lambda rows: [sum(r[c] for r in rows) / len(rows) for c in range(CLASSES)]  # noqa: E731
mious = lambda rows: [sum(r) / CLASSES for r in rows]                             # noqa: E731
ratio = lambda a, b: [x / y for x, y in zip(a, b)]                                # noqa: E731
gaps = lambda a, b: [abs(x - y) for x, y in zip(a, b)]                            # noqa: E731
listing = lambda table: ", ".join(f"{n} {v:.3f}" for n, v in zip(NAMES, table))    # noqa: E731


def trimap(np, torch, functional, ref, photos):
    images, masks = ref.synthetic_segmentation(num_samples=SAMPLES, size=SIZE, seed=0)
    shape = torch.from_numpy((masks > 0).astype("float32")).unsqueeze(1)
    grown = functional.max_pool2d(shape, RING, 1, RING // 2)
    shrunk = -functional.max_pool2d(-shape, RING, 1, RING // 2)
    label = torch.zeros_like(shape)
    label[grown > 0], label[shrunk > 0] = BORDER, INTERIOR
    rng, out = np.random.default_rng(0), np.empty_like(images)
    for i in range(SAMPLES):
        photo = photos[i % len(photos)]
        top, left = (int(rng.integers(0, photo.shape[d] - SIZE)) for d in (0, 1))
        out[i] = np.where((masks[i] > 0)[..., None], images[i], photo[top:top + SIZE, left:left + SIZE])
    return torch.from_numpy(np.clip(out, 0, 1)).permute(0, 3, 1, 2), label[:, 0].long(), images


def train(torch, ref, data, lam, seed):
    xtr, ytr, xva, yva = data
    torch.manual_seed(seed)
    net = ref.UNet(3, CLASSES, base=BASE)
    opt, gen = torch.optim.Adam(net.parameters(), lr=LR), torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        net.train()
        for start in range(0, SPLIT, BATCH):
            batch = torch.randperm(SPLIT, generator=gen)[start:start + BATCH]
            opt.zero_grad()
            ref.combined_loss(net(xtr[batch]), ytr[batch], CLASSES, lam=lam)[0].backward()
            opt.step()
    with torch.no_grad():
        logits = net.eval()(xva)
    return ref.iou_per_class(logits, yva, CLASSES).nan_to_num(0).tolist()


def gradient_share(torch, logits, targets, loss_fn):
    field = logits.clone().requires_grad_(True)
    loss_fn(field, targets).backward()
    weight = field.grad.abs().sum(1)
    per = torch.stack([weight[targets == c].sum() for c in range(CLASSES)])
    return (per / per.sum()).tolist()


def solve():
    try:
        import numpy as np
        import torch
        import torch.nn.functional as functional
        from sklearn.datasets import load_sample_images
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    photos = [p.astype("float32") / 255.0 for p in load_sample_images().images]
    x, y, flat = trimap(np, torch, functional, ref, photos)
    data, held = (x[:SPLIT], y[:SPLIT], x[SPLIT:], y[SPLIT:]), y[SPLIT:]
    runs = {lam: [train(torch, ref, data, lam, s) for s in SEEDS] for lam in LAMBDAS}
    uniform = torch.zeros(len(held), CLASSES, SIZE, SIZE)
    return {"runs": runs, "photos": len(photos),
            "freq": [float((held == c).float().mean()) for c in range(CLASSES)],
            "colour": [float(x.permute(0, 2, 3, 1)[y == 0].std(0).mean()),
                       float(flat.reshape(-1, 3)[(y.reshape(-1) == 0).numpy()].std(0).mean())],
            "ce": gradient_share(torch, uniform, held, lambda f, t: functional.cross_entropy(f, t)),
            "dice": gradient_share(torch, uniform, held, lambda f, t: ref.dice_loss(f, t, CLASSES))}


def verify(result):
    freq, ce, dice = result["freq"], result["ce"], result["dice"]
    plain, both = per_class(result["runs"][0.0]), per_class(result["runs"][1.0])
    scores = mious(result["runs"][0.0]) + mious(result["runs"][1.0])
    spread, delta = max(scores) - min(scores), gaps(both, plain)
    lift, photo_std, flat_std = ratio(dice, ce), *result["colour"]
    return [
        practice.Check(
            "ANSWER: a Pets-shaped trimap on real photographic pixels, since Pets is unreachable",
            freq[0] > 0.8 > max(freq[1:]) and freq[2] > freq[1] and photo_std > 10 * flat_std,
            f"three classes at held-out frequencies {listing(freq)} -- a {RING}-wide dilate-minus-erode "
            f"ring, so the border outnumbers the interior it surrounds. Background pixels come from the "
            f"{result['photos']} photographs scikit-learn ships and carry a colour standard deviation of "
            f"{photo_std:.3f} against {flat_std:.3f} for the lesson's flat green, "
            f"{photo_std / flat_std:.0f}x. {REAL_RUN}"),
        practice.Check(
            "ANSWER: per-class IoU, with and without the Dice term",
            min(both) > 0.5 and both[2] == min(both) and plain[2] == min(plain),
            f"the lesson's own `combined_loss` at lam=0 and lam=1, {len(SEEDS)} seeds each, mean per-class "
            f"IoU: cross-entropy alone {listing(plain)}; plus Dice {listing(both)}. The border ring is the "
            f"hardest class either way, {min(both):.3f} at best against {max(both):.3f} for background"),
        practice.Check(
            "MECHANISM: cross-entropy's gradient share is exactly the pixel frequency",
            max(gaps(ce, freq)) < 1e-4,
            f"at uniform logits, gradient magnitude on the logits attributed to the true class of the pixel: "
            f"{listing(ce)}, against class frequencies {listing(freq)} -- equal to {max(gaps(ce, freq)):.1e}. "
            "Cross-entropy weighs a class by how many pixels it owns, which is the whole reason a rare "
            "class is under-served"),
        practice.Check(
            "ANSWER: Dice re-weights toward the rare classes, so those are the ones it helps",
            min(lift[1:]) > 2.0 > 1.0 > lift[0],
            f"the same decomposition for the lesson's `dice_loss`: {listing(dice)}. Adding Dice multiplies "
            f"the interior's share of the gradient by {lift[1]:.2f} and the border's by {lift[2]:.2f} while "
            f"cutting background's to {lift[0]:.2f} -- Dice normalises each class by its own area, so its "
            "per-class weight barely depends on that class's pixel count"),
        practice.Check(
            "CONTROL: 2 IoU points is far below what this recipe reproduces, delta included",
            spread > 4 * TOLERANCE and max(delta) < spread,
            f"mIoU over the {len(scores)} runs spans {min(scores):.4f} to {max(scores):.4f}, a {spread:.4f} "
            f"spread -- {spread / TOLERANCE:.0f}x the {TOLERANCE} the exercise asks to match. The per-class "
            f"change from adding Dice is {fmt(delta)}, every one of them smaller than that spread, so the "
            "training numbers cannot rank the classes and the gradient decomposition above is what answers "
            "the question"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
