"""Exercise 3 — r2plus1d stride repair.

    **(Hard)** Build an R(2+1)D-18 by replacing every Conv2d in a ResNet-18 with `Conv2Plus1D`. Inflate the first conv's weights from an ImageNet-pretrained ResNet-18. Train on the motion dataset from exercise 2 and beat FramePool.

Reading of the exercise: two of its three instructions do not survive contact with
the lesson's own `Conv2Plus1D`. Taken literally, "replace every Conv2d" silently
deletes every stride -- the class accepts only `(in_c, out_c, kernel_size)` -- so
the network still runs (the downsample branches lose their stride too, so the
residual adds still line up) but never downsamples: 512x4x8x8 reaches the pool
instead of 512x4x1x1, at 44x the arithmetic. The repair used here stays inside the
lesson's code: build `Conv2Plus1D` unchanged, then re-set `spatial.stride`, which
`nn.Conv3d` reads at call time. And "inflate the first conv's weights from an
ImageNet-pretrained ResNet-18" is impossible in principle, not merely offline: the
lesson's `mid_c` formula makes conv1 a 3 -> 110 -> 64 sandwich, so ResNet-18's
(64, 3, 7, 7) conv1 has no tensor to land in -- 9,408 values against 16,170 and
49,280 -- and even the lesson's own `inflate_2d_to_3d` produces a third shape,
(64, 3, 3, 7, 7), that fits neither. This is why the R(2+1)D paper trains from
scratch. Nothing is downloaded here: every model is built with `weights=None` and
starts from random init, so the ImageNet initialisation the exercise asks for is
simply absent, and that is a real handicap this run accepts. What is left of the
exercise -- beat FramePool -- does hold, and by a wide margin.

The dataset, the training loop and the batched scorer are exercise 2's own
`clips`, `fit` and `score`, and the MAC counter is exercise 1's own `scan`, all
imported from the sibling files rather than copied -- so "the motion dataset from
exercise 2" is literally that: 16x16 frames, T=4, 24 clips per class, scored on 24
more per class drawn from a second seed. Both arms get the same budget, 6 epochs
of Adam(lr=1e-3) at batch 8, ~35 s in total on two CPU threads, which is why
the clip is this small. `convert` does the recursive module surgery (BatchNorm2d,
MaxPool2d and AdaptiveAvgPool2d have to become their 3D forms or
`BasicBlock.forward` fails on a 5-D tensor); `shape_of` walks ResNet-18's own
forward path to the map that reaches avgpool and `profile` totals conv MACs, both
on torch's meta device, so no gigaflop is ever executed. The Kinetics figure in
the last check is arithmetic only -- measured MACs times clips times epochs -- and
is labelled an estimate because it ignores video decoding, which is what actually
bounds the real recipe.
"""

from __future__ import annotations

import math
import pathlib

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "12-video-understanding"

EPOCHS, SEEDS, FIELDS = 6, (0, 1), ("imagenet", "inflated", "spatial", "temporal")
KIN_SIZE, KIN_T, KIN_CLIPS, KIN_EPOCHS, TFLOPS = 112, 16, 240_000, 45, 50e12
COUNTER, DATASET = "ex01_inflated_flop_ratio.py", "ex02_pooled_motion_ceiling.py"
COMMAND = ("torchrun --nproc_per_node=8 references/video_classification/train.py --data-path "
           "<kinetics400> --model r2plus1d_18 --batch-size 16 --lr 0.64 --epochs 45 --amp")

pull = lambda arms, key: [arms[seed][key] for seed in SEEDS]                        # noqa: E731
joined = lambda values, fmt: "/".join(format(v, fmt) for v in values)               # noqa: E731
rel = lambda arms: [g / s for g, s in zip(pull(arms, "gap"), pull(arms, "scale"))]  # noqa: E731
sibling = lambda name: practice.load_module(pathlib.Path(__file__).with_name(name))  # noqa: E731
shape_of = lambda model, sample: tuple(model.layer4(model.layer3(model.layer2(      # noqa: E731
    model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(sample)))))))).shape)


def convert(nn, ref, module, strided=True):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            block = ref.Conv2Plus1D(child.in_channels, child.out_channels, child.kernel_size[0])
            block.spatial.stride = (1, *child.stride) if strided else block.spatial.stride
            setattr(module, name, block)
        elif isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.BatchNorm3d(child.num_features))
        elif isinstance(child, nn.MaxPool2d):
            setattr(module, name, nn.MaxPool3d((1, child.kernel_size, child.kernel_size),
                                               (1, child.stride, child.stride),
                                               (0, child.padding, child.padding)))
        elif isinstance(child, nn.AdaptiveAvgPool2d):
            setattr(module, name, nn.AdaptiveAvgPool3d((1, 1, 1)))
        else:
            convert(nn, ref, child, strided)
    return module


def profile(torch, nn, scan, model, shape) -> dict:
    model.to("meta")
    with torch.no_grad():
        return {"macs": scan(torch, nn, model, shape, "meta")["conv"],
                "pool_in": shape_of(model, torch.zeros(shape, device="meta"))}


def graded(torch, score, model, x, y, axis) -> dict:
    ahead, back = score(torch, model, x), score(torch, model, x.flip(axis))
    return {"accuracy": float((ahead.argmax(1) == y).float().mean()),
            "gap": float((ahead - back).abs().max()), "scale": float(ahead.abs().max())}


def solve():
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        from torchvision.models import resnet18
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref, ex01, ex02 = parity.load_reference(PHASE, LESSON, "main"), sibling(COUNTER), sibling(DATASET)
    torch.set_num_threads(2)
    classes, frames, side = ex02.CLASSES, ex02.T, ex02.SIZE
    build = lambda strided=True: convert(  # noqa: E731 - the factory `fit` seeds and calls
        nn, ref, resnet18(weights=None, num_classes=classes), strided)
    plain, conv1 = resnet18(weights=None, num_classes=classes), build().conv1
    clip, kinetics = (1, 3, frames, side, side), (1, 3, KIN_T, KIN_SIZE, KIN_SIZE)
    train_x, train_y = ex02.clips(np, torch, ex02.PER_CLASS, 0)
    test_x, test_y = ex02.clips(np, torch, ex02.PER_CLASS, 1)
    video, test_video = train_x.permute(0, 2, 1, 3, 4), test_x.permute(0, 2, 1, 3, 4)
    pool = ex02.fit(torch, lambda: ref.FramePool(num_classes=classes, pretrained=False),
                    train_x, train_y, SEEDS[0], EPOCHS)
    return {"pool": graded(torch, ex02.score, pool, test_x, test_y, 1), "clips": len(test_y),
            "ceiling": 2.0 / classes,
            "setup": f"{EPOCHS} epochs of Adam(lr={ex02.LR}) on {len(train_y)} {side}x{side} T={frames} clips",
            "arms": {seed: graded(torch, ex02.score, ex02.fit(torch, build, video, train_y, seed,
                                  EPOCHS), test_video, test_y, 2) for seed in SEEDS},
            "repaired": profile(torch, nn, ex01.scan, build(), clip),
            "literal": profile(torch, nn, ex01.scan, build(False), clip),
            "kinetics": profile(torch, nn, ex01.scan, build(), kinetics)["macs"],
            "shapes": {"imagenet": tuple(plain.conv1.weight.shape), "spatial": tuple(conv1.spatial.weight.shape),
                       "inflated": tuple(ref.inflate_2d_to_3d(plain.conv1, 3).weight.shape),
                       "temporal": tuple(conv1.temporal.weight.shape)},
            "params": {"resnet": sum(p.numel() for p in plain.parameters()),
                       "r2plus1d": sum(p.numel() for p in build().parameters())}}


def verify(result):
    arms, pool, shapes, params = result["arms"], result["pool"], result["shapes"], result["params"]
    beat, ceiling = pull(arms, "accuracy"), result["ceiling"]
    fits = [math.prod(shapes[name]) for name in FIELDS]
    blow = result["literal"]["macs"] / result["repaired"]["macs"]
    share, flops = params["r2plus1d"] / params["resnet"], KIN_CLIPS * KIN_EPOCHS * 3 * result["kinetics"] * 2
    return [
        practice.Check(
            "ANSWER: the repaired R(2+1)D-18 beats FramePool, and clears the 2/3 ceiling FramePool cannot",
            all(value > ceiling + 1e-6 for value in beat) and min(beat) > pool["accuracy"],
            f"{result['setup']} from exercise 2, scored on {result['clips']} held-out clips from a second seed: R(2+1)D-18 reaches "
            f"{joined(beat, '.3f')} on seeds {SEEDS} against FramePool's {pool['accuracy']:.3f} at the identical budget, past the "
            f"{ceiling:.3f} that order-blind pooling caps out at -- both from random init, nothing downloaded"),
        practice.Check(
            "FINDING: the exercise's inflation step cannot be done at all, offline or online",
            fits[0] not in fits[2:] and fits[1] not in fits[2:],
            f"mid_c makes conv1 a 3 -> {shapes['spatial'][0]} -> {shapes['temporal'][0]} sandwich, so ResNet-18's conv1 weight "
            f"{shapes['imagenet']} ({fits[0]:,} values) fits neither the spatial {shapes['spatial']} ({fits[2]:,}) nor the temporal "
            f"{shapes['temporal']} ({fits[3]:,}), and inflate_2d_to_3d makes it {shapes['inflated']} ({fits[1]:,}), a third shape fitting neither"),
        practice.Check(
            "MECHANISM: taken literally the replacement deletes every stride -- it runs, at 44x the cost",
            blow > 10 and result["literal"]["pool_in"][-1] > result["repaired"]["pool_in"][-1],
            f"Conv2Plus1D takes only (in_c, out_c, kernel_size), so a literal swap drops the stride from all 7 strided convs -- the 3 downsample "
            f"1x1s included, which is why the residual adds still line up and nothing raises. The map reaching avgpool is "
            f"{result['literal']['pool_in']} instead of {result['repaired']['pool_in']}, a clip costs {result['literal']['macs'] / 1e9:.3f} GMAC "
            f"instead of {result['repaired']['macs'] / 1e9:.3f}, {blow:.1f}x -- re-setting spatial.stride repairs it in place"),
        practice.Check(
            "MECHANISM: the win is a temporal kernel that is not permutation-invariant",
            min(rel(arms)) > 0.1 > pool["gap"] / pool["scale"],
            f"reversing a test clip moves the trained R(2+1)D's logits by {joined(pull(arms, 'gap'), '.2f')} against scales of "
            f"{joined(pull(arms, 'scale'), '.2f')} -- relative {joined(rel(arms), '.2f')} -- while the same reversal moves FramePool's by "
            f"{pool['gap']:.1e} at scale {pool['scale']:.2f}. The kx1x1 temporal conv sees frame order; a mean over time cannot"),
        practice.Check(
            "CONTROL: (2+1)D buys no parameters back, and at Kinetics scale it is a GPU job",
            abs(share - 3.0) < 0.1 and flops / (TFLOPS * 3600) > 1.0,
            f"the converted network holds {params['r2plus1d']:,} parameters against ResNet-18's {params['resnet']:,}, {share:.3f}x -- the lesson's "
            f"Key Terms call (2+1)D 'fewer parameters', but mid_c is chosen to *match* a 3D conv, which exercise 1 measures at exactly 3.0000x. At "
            f"Kinetics settings ({KIN_T} frames, {KIN_SIZE}x{KIN_SIZE}) it measures {result['kinetics'] / 1e9:.1f} GMAC per clip, so "
            f"{KIN_CLIPS:,} clips x {KIN_EPOCHS} epochs x 3 for the backward is {flops / 1e18:.2f} EFLOP -- an estimated "
            f"{flops / (TFLOPS * 3600):,.0f} GPU-hours at {TFLOPS / 1e12:.0f} TFLOP/s assumed, an optimistic floor since `{COMMAND}` is bound by "
            f"video decoding, not the model"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
