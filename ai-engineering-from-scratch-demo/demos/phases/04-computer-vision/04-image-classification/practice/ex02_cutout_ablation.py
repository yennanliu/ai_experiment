"""Exercise 2 — cutout ablation.

    **(Medium)** Implement Cutout — zero out a random 8x8 square in each training
    image — and run an ablation vs no augmentation, hflip+crop,
    hflip+crop+cutout, hflip+crop+mixup. Report val accuracy for each.

Reading of the exercise: the four arms are run at the lesson's own settings on
the lesson's own dataset, and val accuracy is reported as asked. It does not
separate them -- all four reach 1.000 -- so the report is completed with the
measurement that says why that is not a tie to be broken: one arm re-run under a
second seed moves its final val loss further than the four arms differ from each
other, which means no ranking read off this ablation would be real. "Zero out a
random 8x8 square" is then made precise, because two readings of it differ: a
hole composed after `standardize` is filled with the dataset mean, one composed
before is filled with -2.0 in normalised units, and one composed before
`random_crop` is not an 8x8 square by the time it reaches the model.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "04-image-classification"

CLASSES, EPOCHS, HOLE, PER_CLASS, MEAN, STD = 10, 5, 8, 200, [0.5] * 3, [0.25] * 3   # main()'s own
ARMS = ("none", "hflip+crop", "hflip+crop+cutout", "hflip+crop+mixup")


def cutout(np, size=HOLE, value=0.0):
    """Fill a random size x size square, placed so it lies wholly inside the image."""
    def _fn(img):
        top, left = (np.random.randint(0, side - size + 1) for side in img.shape[:2])
        out = img.copy()
        out[top:top + size, left:left + size, :] = value
        return out
    return _fn


def run(torch, np, ref, loader_cls, arm, seed=0) -> dict:
    """One arm, five epochs, the lesson's own `compose`, optimiser and schedule.

    Cutout goes after `standardize`, so its hole is filled with the dataset mean.
    """
    x, y = ref.synthetic_cifar(num_per_class=PER_CLASS)
    split, plain = int(0.9 * len(x)), ref.standardize(MEAN, STD)
    stages = [] if arm == "none" else [ref.random_hflip(), ref.random_crop(pad=4), plain]
    pipeline = ref.compose(*stages, *([cutout(np)] if arm.endswith("cutout") else [])) if stages else plain
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = loader_cls(ref.ArrayDataset(x[:split], y[:split], pipeline), batch_size=128, shuffle=True)
    val = loader_cls(ref.ArrayDataset(x[split:], y[split:], plain), batch_size=256)
    model = ref.MiniClassifier(num_classes=CLASSES)
    optimiser = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4,
                                nesterov=True)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)
    history = {"val_loss": [], "accuracy": []}
    for _ in range(EPOCHS):
        ref.train_one_epoch(model, train, optimiser, "cpu", CLASSES, arm.endswith("mixup"))
        for key, value in zip(history, ref.evaluate(model, val, "cpu", CLASSES)[:2]):
            history[key].append(value)
        schedule.step()
    return history


def holes(np, ref) -> dict:
    """What the hole actually is, under the three places it can be composed."""
    np.random.seed(0)
    image = ref.synthetic_cifar(num_per_class=1, num_classes=1)[0][0]
    after, before = ref.compose(ref.standardize(MEAN, STD), cutout(np)), ref.compose(
        cutout(np), ref.standardize(MEAN, STD))
    traced = ref.compose(cutout(np, value=np.nan), ref.random_crop(pad=4))
    sizes = [int(np.isnan(traced(image)).sum()) for _ in range(200)]
    corners = {tuple(np.argwhere(after(image) == 0.0)[0][:2]) for _ in range(200)}
    return {"zeroed": int((after(image) == 0.0).sum()), "fill_before": float(before(image).min()),
            "corners": len(corners), "cropped": (min(sizes), max(sizes)),
            "intact": sizes.count(HOLE ** 2 * 3) / len(sizes)}


def solve():
    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    arms = {arm: run(torch, np, ref, DataLoader, arm) for arm in ARMS}
    return {"arms": arms, "holes": holes(np, ref),
            "repeat": run(torch, np, ref, DataLoader, "hflip+crop+cutout", seed=1)}


def verify(result):
    arms, hole, repeat = result["arms"], result["holes"], result["repeat"]
    final = {a: arms[a]["val_loss"][-1] for a in ARMS}
    spread, reseed = max(final.values()) - min(final.values()), abs(final["hflip+crop+cutout"] - repeat["val_loss"][-1])
    table = ", ".join(f"{a} {arms[a]['accuracy'][-1]:.3f}/{final[a]:.3f}" for a in ARMS)
    return [
        practice.Check(
            "ANSWER: all four arms reach val accuracy 1.000",
            {arms[a]["accuracy"][-1] for a in ARMS} == {1.0},
            f"five epochs each on the lesson's 2,000-image synthetic set, val accuracy / val loss: {table}. "
            "Epochs to first 1.000: " + ", ".join(f"{a} {arms[a]['accuracy'].index(1.0) + 1}" for a in ARMS)),
        practice.Check(
            "FINDING: val loss separates what val accuracy ties, and by far more than seed noise",
            spread > 3 * reseed,
            "the four final val losses fall in augmentation order -- "
            + " > ".join(f"{a} {final[a]:.3f}" for a in ARMS)
            + f", a span of {spread:.3f}, while re-running hflip+crop+cutout under a second seed moves its "
            f"own by only {reseed:.3f}: {spread / max(reseed, 1e-9):.0f}x the seed noise. The tie at 1.000 "
            "accuracy is the metric's resolution, not the arms being equal"),
        practice.Check(
            "MECHANISM: the hole is exactly 8x8x3 = 192 entries, and it moves",
            hole["zeroed"] == HOLE * HOLE * 3 and hole["corners"] > 100,
            f"composed after `standardize`, cutout zeroes exactly {hole['zeroed']} entries ({HOLE}x{HOLE}x3), "
            f"and its top-left corner took {hole['corners']} distinct positions over 200 draws out of the "
            f"{(32 - HOLE + 1) ** 2} a wholly-inside placement allows"),
        practice.Check(
            "CONTROL: composed before `standardize`, 'zero' is not zero",
            hole["fill_before"] == -2.0,
            f"placed before the standardiser it fills the hole with raw 0.0, which (0 - {MEAN[0]}) / "
            f"{STD[0]} turns into {hole['fill_before']} -- two standard deviations below the mean, an "
            "out-of-distribution patch rather than the neutral one the method wants"),
        practice.Check(
            "CONTROL: composed before `random_crop`, it is no longer an 8x8 square",
            hole["intact"] < 0.7 and hole["cropped"][0] < HOLE * HOLE * 3 < hole["cropped"][1],
            f"traced with a NaN fill through the reflect-padded `random_crop`, the surviving hole ran from "
            f"{hole['cropped'][0]} to {hole['cropped'][1]} entries over 200 draws, reaching the intended "
            f"{HOLE ** 2 * 3} only {hole['intact']:.0%} of the time: the crop slides part of the square off "
            "the image, and reflect padding copies a border-touching hole back in twice"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
