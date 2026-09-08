"""Exercise 1 — probe vs finetune gap.

    **(Easy)** Train a `ResNet18` as a linear probe (backbone frozen) and as a
    full fine-tune on the same synthetic-CIFAR dataset. Report both accuracies
    side by side. Explain which gap tells you the features transfer well and
    which tells you they do not.

Reading of the exercise: the exercise reads the two accuracies as a diagnosis of
*pretrained* features, and that premise cannot survive a hermetic test. The
lesson's own `make_feature_extractor` asks torchvision for
`ResNet18_Weights.IMAGENET1K_V1`, which is a 46 MB fetch of
`resnet18-f37072fd.pth` that CI has no network for, so the one line rebound below
builds the same torchvision ResNet-18 with `weights=None` and everything else --
the freeze loop, the head swap, `discriminative_param_groups`, `freeze_bn_stats`,
`train_and_eval`, `synthetic_dataset` -- is the lesson's own code, imported.
Both arms therefore start from noise, and no gap between them can speak about
transfer; exercise 3 pretrains a checkpoint of its own and answers that question
there. What this file answers is the measurable half: the two accuracies side by
side at the lesson's own learning rates, why they are equal, and what the word
"frozen" actually froze. Scaled to 300 images at 32x32 for three epochs so the
lesson's tests stay in budget. The module-level lambdas hold the comprehensions
and boolean chains that would otherwise push `verify` past D14's complexity
ceiling of 8.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "05-transfer-learning"

SIZE, PER_CLASS, TRAIN, EPOCHS, CLASSES, SEED = 32, 40, 300, 3, 10, 0
PROBE_LR, TUNE_LR, DECAY, STAGES = 3e-2, 1e-3, 0.3, 6

tied = lambda a, b: min(a, b) > 0.95 and abs(a - b) < 0.02                            # noqa: E731
adapted = lambda p, f: (p["backbone"] == 0.0 and p["stats"] > 0.5 and p["n_stats"] == 9600   # noqa: E731
                        and f["stats"] == 0.0 and f["accuracy"] > 0.95)
laddered = lambda rung, t: (abs(rung[0][1] / rung[-1][1] - DECAY ** (STAGES - 1)) < 1e-9  # noqa: E731
                            and t["head"] > 20 * t["backbone"])


def displacement(before, model, keys) -> float:
    """Relative L2 movement of a named slice of the state dict, ||dW|| / ||W_0||."""
    now = model.state_dict()
    moved = sum(float(((now[k] - before[k]) ** 2).sum()) for k in keys)
    return (moved ** 0.5) / (sum(float((before[k] ** 2).sum()) for k in keys) ** 0.5 + 1e-12)


def arm(torch, ref, data, factory, lr, freeze_bn=False) -> dict:
    """One run of the lesson's own `train_and_eval`, plus what moved while it ran."""
    torch.manual_seed(SEED)
    model = factory(CLASSES)
    stats = [n for n, b in model.named_buffers() if b.dtype.is_floating_point]
    backbone = [n for n, _ in model.named_parameters() if not n.startswith("fc")]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with parity.quiet():
        acc = ref.train_and_eval(model, *data, "cpu", epochs=EPOCHS, base_lr=lr,
                                 freeze_bn=freeze_bn)
    return {"accuracy": acc, "trainable": ref.trainable_param_count(model),
            "backbone": displacement(before, model, backbone),
            "stats": displacement(before, model, stats),
            "head": displacement(before, model, ["fc.weight", "fc.bias"]),
            "n_stats": sum(before[k].numel() for k in stats)}


def solve():
    try:
        import numpy as np
        import torch
        from sklearn.linear_model import LogisticRegression
        from torch.utils.data import DataLoader
        from torchvision.models import resnet18
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    torch.set_num_threads(2)
    np.random.seed(SEED)
    ref = parity.load_reference(PHASE, LESSON, "main")
    ref.resnet18 = lambda weights=None, **kw: resnet18(weights=None, **kw)   # never download
    x, y = ref.synthetic_dataset(num_per_class=PER_CLASS, size=SIZE)
    data = (DataLoader(ref.ArrayDataset(x[:TRAIN], y[:TRAIN]), batch_size=16, shuffle=True),
            DataLoader(ref.ArrayDataset(x[TRAIN:], y[TRAIN:]), batch_size=64))
    means = x.reshape(len(x), -1, 3).mean(axis=1)
    baseline = LogisticRegression(max_iter=2000).fit(means[:TRAIN], y[:TRAIN])
    rungs = ref.discriminative_param_groups(ref.make_fine_tune(CLASSES), base_lr=TUNE_LR)
    return {"probe": arm(torch, ref, data, ref.make_feature_extractor, PROBE_LR),
            "frozen_bn": arm(torch, ref, data, ref.make_feature_extractor, PROBE_LR, True),
            "finetune": arm(torch, ref, data, ref.make_fine_tune, TUNE_LR),
            "pixel_mean": float(baseline.score(means[TRAIN:], y[TRAIN:])),
            "ladder": [(g["name"], g["lr"]) for g in rungs], "held_out": len(x) - TRAIN,
            "params": sum(p.numel() for p in ref.make_fine_tune(1000).parameters())}


def verify(result):
    probe, tuned, frozen = result["probe"], result["finetune"], result["frozen_bn"]
    rungs, ratio = result["ladder"], result["finetune"]["trainable"] / result["probe"]["trainable"]
    return [
        practice.Check(
            "ANSWER: probe and full fine-tune land on the same accuracy",
            tied(probe["accuracy"], tuned["accuracy"]),
            f"{EPOCHS} epochs on {TRAIN} images at {SIZE}x{SIZE} through the lesson's own "
            f"train_and_eval at its own LRs, {result['held_out']} held out: probe (base_lr "
            f"{PROBE_LR}, {probe['trainable']:,} trainable) {probe['accuracy']:.3f} vs fine-tune "
            f"(base_lr {TUNE_LR}, {tuned['trainable']:,}, {ratio:,.0f}x more) "
            f"{tuned['accuracy']:.3f} -- gap {tuned['accuracy'] - probe['accuracy']:+.3f}"),
        practice.Check(
            "FINDING: there are no pretrained features here, so no gap can diagnose transfer",
            result["params"] == 11_689_512,
            f"the lesson asks for ResNet18_Weights.IMAGENET1K_V1, a 46 MB fetch a hermetic test "
            f"cannot make, so both arms start from torchvision's own randomly initialised "
            f"ResNet-18 -- {result['params']:,} parameters at 1,000 classes, "
            f"{tuned['trainable']:,} once the lesson swaps in a 10-class head. A "
            "probe-vs-fine-tune gap is a claim about pretrained features; there are none here"),
        practice.Check(
            "FINDING: 'backbone frozen' froze the weights and nothing else, and it did not matter",
            adapted(probe, frozen),
            f"after the probe run the backbone parameters have moved exactly "
            f"{probe['backbone']:.1e} in relative L2 while its {probe['n_stats']:,} BatchNorm "
            f"running statistics moved {probe['stats']:.3f} (head {probe['head']:.3f}). Pinning "
            f"them with the lesson's own freeze_bn_stats holds that at {frozen['stats']:.1e} and "
            f"the arm still scores {frozen['accuracy']:.3f} with its head moving "
            f"{frozen['head']:.3f}: the channel is real, this task just does not need it"),
        practice.Check(
            "MECHANISM: the LR ladder makes 'full fine-tune' a probe with a slower head",
            laddered(rungs, tuned),
            f"discriminative_param_groups splits ResNet-18 into {len(rungs)} groups; at base_lr "
            f"{TUNE_LR} the {rungs[0][0]} group runs at {rungs[0][1]:.2e}, {DECAY}^{STAGES - 1} "
            f"= {DECAY ** (STAGES - 1):.5f} of the head's {rungs[-1][1]:.2e}. Over the run the "
            f"backbone weights moved {tuned['backbone']:.2e} and the head {tuned['head']:.3f}, "
            f"{tuned['head'] / tuned['backbone']:.0f}x further"),
        practice.Check(
            "CONTROL: three numbers per image get most of the way, so the task is near-trivial",
            result["pixel_mean"] > 0.7,
            f"a logistic regression on nothing but each image's three per-channel means scores "
            f"{result['pixel_mean']:.3f} on the same held-out split against the ResNet's "
            f"{probe['accuracy']:.3f}. synthetic_dataset gives class c its own colour centre, so "
            "the label is largely in the mean colour: this dataset cannot tell good features from "
            "an easy task"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
