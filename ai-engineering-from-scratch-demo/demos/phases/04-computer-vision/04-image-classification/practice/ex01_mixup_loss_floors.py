"""Exercise 1 — mixup loss floors.

    **(Easy)** Train the same model with and without mixup for five epochs on the
    synthetic dataset. Plot train and val loss for both. Explain why train loss
    with mixup is higher yet val accuracy is similar or better.

Reading of the exercise: "the same model ... with and without mixup" is run
through the lesson's own `train_one_epoch`, which is the only honest way to
answer the question -- and running it exposes the reason the question is harder
than it looks. The two arms do not minimise the same functional: the mixup arm
scores `soft_cross_entropy` against a mixed target, the other scores
`cross_entropy(label_smoothing=0.1)`. Each has a different irreducible floor, so
the two numbers are not comparable until both floors are subtracted, and the
explanation the exercise asks for is exactly that floor. Both floors are
measured rather than asserted. "Val accuracy similar or better" is reported as
what it is on this dataset, which turns out to be a claim it cannot test.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "04-image-classification"

CLASSES, EPOCHS, ALPHA, SMOOTHING = 10, 5, 0.2, 0.1
MEAN, STD = [0.5] * 3, [0.25] * 3   # the lesson's own main() values


def loaders(ref, loader_cls):
    """The lesson's own pipeline, plus a clean un-augmented view of the training set."""
    x, y = ref.synthetic_cifar(num_per_class=200)
    split, eval_tf = int(0.9 * len(x)), ref.standardize(MEAN, STD)
    train_tf = ref.compose(ref.random_hflip(), ref.random_crop(pad=4), eval_tf)
    return (loader_cls(ref.ArrayDataset(x[:split], y[:split], train_tf), batch_size=128, shuffle=True),
            loader_cls(ref.ArrayDataset(x[split:], y[split:], eval_tf), batch_size=256),
            loader_cls(ref.ArrayDataset(x[:split], y[:split], eval_tf), batch_size=256))


def arm(torch, np, ref, loader_cls, use_mixup) -> dict:
    """One five-epoch run of the lesson's own loop, at the lesson's own settings."""
    torch.manual_seed(0)
    np.random.seed(0)
    train, val, clean = loaders(ref, loader_cls)
    model = ref.MiniClassifier(num_classes=CLASSES)
    optimiser = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4,
                                nesterov=True)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)
    history = {"train": [], "train_acc": [], "val": [], "val_acc": []}
    for _ in range(EPOCHS):
        scores = (*ref.train_one_epoch(model, train, optimiser, "cpu", CLASSES, use_mixup),
                  *ref.evaluate(model, val, "cpu", CLASSES)[:2])
        schedule.step()
        for key, value in zip(history, scores):
            history[key].append(value)
    clean_loss, clean_acc, _cm = ref.evaluate(model, clean, "cpu", CLASSES)
    return dict(history, clean=clean_loss, clean_acc=clean_acc)


def mixup_floor(np) -> float:
    """E[H(lambda)] under Beta(alpha, alpha), times the chance the pair is cross-class."""
    lam = np.random.default_rng(0).beta(ALPHA, ALPHA, 200_000).clip(1e-12, 1 - 1e-12)
    return float(-(lam * np.log(lam) + (1 - lam) * np.log1p(-lam)).mean()) * (1 - 1 / CLASSES)


def smoothing_floor() -> float:
    """Closed form: the entropy of torch's smoothed target (1-e)*onehot + e/K."""
    top, rest = 1 - SMOOTHING + SMOOTHING / CLASSES, SMOOTHING / CLASSES
    return -(top * math.log(top) + (CLASSES - 1) * rest * math.log(rest))


def solve():
    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    return {"mixup": arm(torch, np, ref, DataLoader, True),
            "plain": arm(torch, np, ref, DataLoader, False),
            "floors": {"mixup": mixup_floor(np), "plain": smoothing_floor()}}


def verify(result):
    mix, plain, floor = result["mixup"], result["plain"], result["floors"]
    raw, excess = mix["train"][-1] - plain["train"][-1], {k: result[k]["train"][-1] - floor[k] for k in floor}
    series = lambda run, key: " ".join(f"{v:5.3f}" for v in run[key])   # noqa: E731 - a formatter
    return [
        practice.Check(
            "ANSWER: the premise holds, but by 0.015 -- far less than the explanation implies",
            mix["train"][-1] > plain["train"][-1],
            f"five epochs at the lesson's own settings. train loss mixup {series(mix, 'train')} / "
            f"plain {series(plain, 'train')}; val loss mixup {series(mix, 'val')} / plain "
            f"{series(plain, 'val')}; final train {mix['train'][-1]:.3f} vs {plain['train'][-1]:.3f}"),
        practice.Check(
            "FINDING: the two numbers are not the same objective, and the floors differ 2.4x",
            floor["plain"] > floor["mixup"] and excess["mixup"] > 5 * excess["plain"],
            f"mixup minimises soft cross-entropy against a mixed target (floor {floor['mixup']:.3f}); the "
            f"other, cross-entropy with label_smoothing={SMOOTHING} (floor {floor['plain']:.4f}). Above its own "
            f"floor mixup sits {excess['mixup']:.3f} and plain {excess['plain']:.3f}, "
            f"{excess['mixup'] / excess['plain']:.1f}x further out, against a raw gap of {raw:.3f}"),
        practice.Check(
            "MECHANISM: the floor is the entropy of the label the model is asked to predict",
            0.15 < floor["mixup"] < 0.30,
            f"with one lambda ~ Beta({ALPHA}, {ALPHA}) per batch, E[H(lambda)] is "
            f"{floor['mixup'] / (1 - 1 / CLASSES):.4f} over 200,000 draws and nine pairs in ten cross classes, "
            f"so {floor['mixup']:.4f}. A perfect model scores exactly that -- the target really is a mixture, "
            "so the loss is higher without the fit being worse"),
        practice.Check(
            "CONTROL: score the same weights on clean data and the gap disappears",
            mix["clean"] < 0.2 * mix["train"][-1] and mix["clean_acc"] == 1.0,
            f"the same weights on un-mixed, un-augmented training images with plain cross-entropy give "
            f"{mix['clean']:.3f} at accuracy {mix['clean_acc']:.3f}, against the {mix['train'][-1]:.3f} "
            "reported while training: the elevated number describes what it was scored on, not the fit"),
        practice.Check(
            "CONTROL: the reference's mixup train accuracy is a coin flip, not an accuracy",
            abs(sum(mix["train_acc"]) / EPOCHS - 0.5) < 0.15 < plain["train_acc"][-1],
            f"`train_one_epoch` scores argmax of the *mixed* image against the un-permuted y, so it reports "
            f"{series(mix, 'train_acc')}, averaging {sum(mix['train_acc']) / EPOCHS:.3f} -- near the 0.500 of "
            f"guessing which label dominates -- while the same weights are at {mix['clean_acc']:.3f} on clean "
            f"data. The plain arm's {plain['train_acc'][-1]:.3f} is a real accuracy"),
        practice.Check(
            "CONTROL: this dataset cannot test 'val accuracy similar or better'",
            mix["val_acc"][-1] == plain["val_acc"][-1] == 1.0,
            f"both arms end at 1.000 on 200 held-out images (mixup {series(mix, 'val_acc')}, plain "
            f"{series(plain, 'val_acc')}). `synthetic_cifar` gives class c the frequency 2 + c, so there is no "
            "generalisation gap to close; that half of the question needs a dataset the model can overfit"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
