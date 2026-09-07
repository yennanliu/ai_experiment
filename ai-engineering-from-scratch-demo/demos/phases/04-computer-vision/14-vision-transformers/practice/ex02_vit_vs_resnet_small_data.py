"""Exercise 2 — vit vs resnet small data.

    **(Medium)** Fine-tune a pretrained `timm` ViT-S/16 on the synthetic-CIFAR dataset from Lesson 4. Compare against ResNet-18 fine-tuning on the same data. Report training time and final accuracy.

Reading of the exercise: two of its premises do not survive contact. First,
`timm` is not installed in this repo's `vision` group and no weights may be
downloaded, so "pretrained" is unavailable and both arms start from random init
-- the CONTROL below records the exact `ModuleNotFoundError` rather than papering
over it, and what is compared is therefore architecture plus optimiser, not
transfer. Second, and much more damaging, the requested statistic cannot rank
anything: `synthetic_cifar` gives class c the spatial frequency `2 + c`, which is
linearly separable in raw pixel space, so a multinomial logistic regression on the
un-featurised 3072 pixels already scores a perfect validation accuracy and both
networks reach it too. Four runs -- two architectures x two seeds -- finish at the
same number, so "final accuracy" has no resolving power here and the honest
answer is to say so and report what does differ. The obvious fallback, how many
steps each takes to first hit 100%, turns out to flip sign between seeds, so it
is reported with its spread and asserted only as noise. Training time is the one
statistic that survives, and it points the opposite way from parameter count.
The ViT is the lesson's own class at `image_size=32, patch_size=8` (16 patch
tokens, the same sequence length as the lesson's own 64x64/16 demo); ResNet-18 is
`torchvision.models.resnet18(weights=None, num_classes=10)`.

Structure: `dataset` builds Lesson 4's `synthetic_cifar` once and standardises it
with that lesson's own mean/std; `arm` runs one architecture at one seed, timing
only the optimiser steps and sampling validation accuracy every fourth step;
`controls` fits the raw-pixel logistic regression and records whether `timm` can
be imported at all (through `importlib`, so the file's static imports stay inside
its declared `vision` dependency group). At 138 code lines this sits above D14's
120-line target and 12 clear of the ceiling: six checks over two architectures,
two seeds each, plus the two controls that show what the arms do not settle.
"""

from __future__ import annotations

import importlib
import time

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "14-vision-transformers"
CLASSIFICATION = "04-image-classification"

CLASSES, PER_CLASS, SPLIT, BATCH = 10, 60, 480, 48
STEPS, EVERY, LR, SEEDS = 32, 4, 3e-4, (0, 1)
MEAN, STD = 0.5, 0.25         # Lesson 4's own main() normalisation

curve = lambda arm: " ".join(f"{k}:{v:.2f}" for k, v in arm["curve"].items())    # noqa: E731


def dataset(torch, cls4):
    images, labels = cls4.synthetic_cifar(num_per_class=PER_CLASS, num_classes=CLASSES)
    x = (torch.from_numpy(images).permute(0, 3, 1, 2).float() - MEAN) / STD
    y = torch.from_numpy(labels).long()
    return x[:SPLIT], y[:SPLIT], x[SPLIT:], y[SPLIT:]


def arm(torch, data, build, seed) -> dict:
    x_train, y_train, x_val, y_val = data
    torch.manual_seed(seed)
    model = build()
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    gen = torch.Generator().manual_seed(seed)
    scores, first, seconds = {}, 0, 0.0
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, SPLIT, (BATCH,), generator=gen)
        model.train()
        clock = time.perf_counter()
        optimiser.zero_grad()
        torch.nn.functional.cross_entropy(model(x_train[idx]), y_train[idx]).backward()
        optimiser.step()
        seconds += time.perf_counter() - clock
        if step % EVERY == 0:
            model.eval()
            with torch.no_grad():
                scores[step] = (model(x_val).argmax(1) == y_val).float().mean().item()
            first = first or step * int(scores[step] == 1.0)
    return {"params": sum(p.numel() for p in model.parameters()), "seconds": seconds,
            "curve": scores, "first": first, "final": scores[STEPS], "val": len(y_val),
            "convs": sum(isinstance(m, torch.nn.Conv2d) for m in model.modules()),
            "linears": sum(isinstance(m, torch.nn.Linear) for m in model.modules())}


def controls(data) -> dict:
    from sklearn.linear_model import LogisticRegression
    x_train, y_train, x_val, y_val = (t.reshape(len(t), -1).numpy() if t.dim() > 1 else t.numpy()
                                      for t in data)
    probe = LogisticRegression(max_iter=2000).fit(x_train, y_train)
    try:
        available = f"timm {importlib.import_module('timm').__version__} is importable"
    except ImportError as exc:
        available = f"{type(exc).__name__}: {exc}"
    return {"pixels": probe.score(x_val, y_val), "timm": available, "features": x_train.shape[1]}


def solve():
    try:
        import torch
        from torchvision.models import resnet18
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    cls4 = parity.load_reference(PHASE, CLASSIFICATION, "main")
    torch.set_num_threads(2)
    data = dataset(torch, cls4)
    builders = {"vit": lambda: ref.ViT(image_size=32, patch_size=8, num_classes=CLASSES),
                "resnet18": lambda: resnet18(weights=None, num_classes=CLASSES)}
    return {"runs": {(name, seed): arm(torch, data, build, seed)
                     for name, build in builders.items() for seed in SEEDS},
            "controls": controls(data)}


def verify(result):
    runs, control = result["runs"], result["controls"]
    vit, res = [runs[("vit", s)] for s in SEEDS], [runs[("resnet18", s)] for s in SEEDS]
    finals = [run["final"] for run in runs.values()]
    ratio = min(r["seconds"] for r in res) / max(v["seconds"] for v in vit)
    return [
        practice.Check(
            "ANSWER: both reach the same final accuracy, and the ViT gets there 6x faster",
            min(finals) > 0.95 and ratio > 2.0,
            f"{STEPS} AdamW steps (lr {LR}, batch {BATCH}) on {SPLIT} training images scored against "
            f"{vit[0]['val']} held out, both from random init. ViT {vit[0]['params']:,} params, "
            f"{vit[0]['seconds']:.2f}s / {vit[1]['seconds']:.2f}s of optimiser time, final accuracy "
            f"{vit[0]['final']:.3f} / {vit[1]['final']:.3f}; ResNet-18 {res[0]['params']:,} params, "
            f"{res[0]['seconds']:.2f}s / {res[1]['seconds']:.2f}s, final {res[0]['final']:.3f} / "
            f"{res[1]['final']:.3f} -- at least {ratio:.1f}x the wall clock for the same answer"),
        practice.Check(
            "FINDING: 'final accuracy' cannot rank these two — all four runs land on the same number",
            max(finals) - min(finals) < 0.02,
            f"the four final accuracies are {finals}, a spread of {max(finals) - min(finals):.3f}. Lesson 4's "
            f"`synthetic_cifar` gives class c the frequency 2 + c on a fixed 32x32 grid, so the classes are "
            f"separated before any network sees them; the comparison the exercise asks for is unfalsifiable on "
            f"this data and needs a dataset both models can still get wrong"),
        practice.Check(
            "FINDING: the obvious fallback statistic is noise — steps-to-100% flips sign between seeds",
            (vit[0]["first"] < res[0]["first"]) != (vit[1]["first"] < res[1]["first"]),
            f"sampling validation accuracy every {EVERY} steps, the first step at 1.000 is ViT "
            f"{vit[0]['first']} / {vit[1]['first']} against ResNet-18 {res[0]['first']} / {res[1]['first']} -- "
            f"the winner changes with the seed. The curves are ViT {curve(vit[0])} and ResNet-18 "
            f"{curve(res[0])}, both non-monotone, so any 'converges faster' claim from one seed is a coin flip"),
        practice.Check(
            "MECHANISM: wall clock does not follow parameter count — it follows the layer stack",
            res[0]["params"] > 3 * vit[0]["params"] and ratio > 2.0,
            f"ResNet-18 carries {res[0]['params'] / vit[0]['params']:.1f}x the ViT's parameters "
            f"({res[0]['params']:,} vs {vit[0]['params']:,}) and still costs {ratio:.1f}x the time per step. "
            f"It is {res[0]['convs']} sequentially dependent convolutions over feature maps that start at 32x32; "
            f"the ViT is {vit[0]['linears']} `nn.Linear` modules over 17 tokens of width 192, a shape BLAS likes. "
            f"Parameters price memory, not latency"),
        practice.Check(
            "CONTROL: no network is being tested — logistic regression on raw pixels scores the same",
            control["pixels"] >= min(finals),
            f"a multinomial logistic regression on the {control['features']} un-featurised pixels of the same "
            f"{SPLIT} training images scores {control['pixels']:.3f} on the same validation split, matching or "
            f"beating both networks. Whatever the ViT and the ResNet learned, the task did not require it, so "
            f"neither arm's accuracy is evidence about architecture"),
        practice.Check(
            "CONTROL: 'pretrained' is not available here — both arms are random init, and say so",
            control["timm"].startswith("ModuleNotFoundError"),
            f"`importlib.import_module('timm')` gives {control['timm']}; timm is in no dependency group in "
            f"`pyproject.toml`, and downloading ImageNet weights is out of scope, so ResNet-18 is built with "
            f"`weights=None`. The exercise's real subject -- whether ImageNet pretraining rescues a ViT on "
            f"{SPLIT} images -- is untestable here, and this is a from-scratch comparison instead"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
