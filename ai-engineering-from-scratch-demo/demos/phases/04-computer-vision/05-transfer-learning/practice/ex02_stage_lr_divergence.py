"""Exercise 2 — stage lr divergence.

    **(Medium)** Introduce a bug on purpose: set `base_lr = 1e-1` on the backbone
    stage instead of the head. Show the training loss explode, then recover by
    applying the `discriminative_param_groups` helper. Record the LR at which
    each stage starts diverging.

Reading of the exercise: the bug is read exactly as written -- the lesson's own
six groups with the ladder reversed, so `conv1_bn1` runs at 1e-1 and `fc` at
1e-1 * 0.3^5 -- and it does not explode. It trains. The premise is backwards for
this architecture: every backbone stage of a ResNet is a convolution followed by
BatchNorm, which makes the loss invariant to the scale of those weights, while
`fc` is the one stage with nothing after it. (The other reading of the bug, a
flat 1e-1 on every parameter, does explode -- peak 31.0, 11x its first step, when
run while this file was written -- but that is the head being included, and the
per-stage sweep below localises it without a fourth arm.) So the "recovery" is
measured too, and `discriminative_param_groups` at base_lr = 1e-1 explodes,
because the ladder hands its *largest* LR to the head: the helper redistributes
the budget, it does not shrink it. Recovery comes from lowering base_lr, which is
a different fix. "The LR at which each stage starts diverging" is then measured
per stage on a decade grid, one stage at a time -- only that stage is handed to
the optimiser, so only it moves -- with divergence defined up front as a peak
loss above 3x the first step's. Scaled to 200 images at 24x24 and bursts of 16
steps so the sweep fits the test budget; divergence shows up early or not at all. The lesson's factories ask torchvision for ResNet18_Weights.IMAGENET1K_V1,
a 46 MB fetch a hermetic test cannot make, so `ref.resnet18` is rebound to build
the same torchvision graph with `weights=None`; the divergence thresholds below
are therefore properties of the architecture and its initialisation, not of any
checkpoint. The module-level lambdas hold the comprehensions and boolean chains
that would otherwise push `verify` past D14's complexity ceiling of 8.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "05-transfer-learning"

SIZE, PER_CLASS, CLASSES, SEED, BUG_LR, GOOD_LR, DECAY = 24, 20, 10, 0, 1e-1, 1e-3, 0.3
GRID, SWEEP_STEPS, ARM_STEPS, EXPLODE = (1e-2, 1e-1, 1.0, 10.0), 6, 16, 3.0
BACKBONE = ("conv1_bn1", "layer1", "layer2", "layer3", "layer4")

blew_up = lambda seq: not all(map(math.isfinite, seq)) or max(seq) > EXPLODE * seq[0]  # noqa: E731
limits_of = lambda sw: {s: next((lr for lr in GRID if sw[s][lr]["diverged"]), None) for s in sw}  # noqa: E731
peaks_of = lambda sweep: [sweep["fc"][lr]["peak"] for lr in GRID[1:]]                 # noqa: E731
worst = lambda sweep, lr: max(sweep[s][lr]["peak"] for s in BACKBONE)                 # noqa: E731
tail = lambda sweep: max(sweep[s][10.0]["last"] for s in BACKBONE[2:])                # noqa: E731
head_only = lambda lim: lim["fc"] == 1e-1 and all(lim[s] is None for s in BACKBONE)   # noqa: E731
decade = lambda peaks: all(4 < b / a < 25 for a, b in zip(peaks, peaks[1:]))          # noqa: E731
invert = lambda groups: [dict(g, lr=r["lr"]) for g, r in zip(groups, groups[::-1])]   # noqa: E731
stage_only = lambda stage, lr: lambda ref, m, base: [{"lr": lr, "params": next(         # noqa: E731
    g["params"] for g in ref.discriminative_param_groups(m, 1.0, 1.0) if g["name"] == stage)}]
helper = lambda ref, m, base: ref.discriminative_param_groups(m, base)               # noqa: E731
ARM_SPECS = (("inverted", BUG_LR, lambda ref, m, base: invert(helper(ref, m, base))),
             ("helper_hot", BUG_LR, helper), ("helper", GOOD_LR, helper))


def burst(torch, ref, batches, grouping, base, steps) -> dict:
    """A fresh ResNet-18 and `steps` steps of the lesson's own optimiser over one grouping.
    Only the groups handed to SGD move, so freezing is what the optimiser is *not* given."""
    torch.manual_seed(SEED)
    model = ref.make_fine_tune(CLASSES)
    optimiser = torch.optim.SGD(grouping(ref, model, base), momentum=0.9, weight_decay=1e-4,
                                nesterov=True)
    model.train()
    losses = []
    for index in range(steps):
        x, y = batches[index % len(batches)]
        loss = torch.nn.functional.cross_entropy(model(x), y, label_smoothing=0.1)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
    return {"first": losses[0], "peak": max(losses), "last": losses[-1],
            "diverged": blew_up(losses)}


def solve():
    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader
        from torchvision.models import resnet18
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    torch.set_num_threads(2)
    np.random.seed(SEED)
    ref = parity.load_reference(PHASE, LESSON, "main")
    ref.resnet18 = lambda weights=None, **kw: resnet18(weights=None, **kw)   # never download
    x, y = ref.synthetic_dataset(num_per_class=PER_CLASS, size=SIZE)
    loader = DataLoader(ref.ArrayDataset(x, y), batch_size=16, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    batches = [batch for _, batch in zip(range(ARM_STEPS), loader)]
    ladder = [(g["name"], g["lr"])
              for g in ref.discriminative_param_groups(ref.make_fine_tune(CLASSES), BUG_LR)]
    sweep = {s: {lr: burst(torch, ref, batches, stage_only(s, lr), 0.0, SWEEP_STEPS)
                 for lr in GRID} for s, _ in ladder}
    runs = {name: burst(torch, ref, batches, grouping, base, ARM_STEPS)
            for name, base, grouping in ARM_SPECS}
    return dict(runs, sweep=sweep, ladder=ladder)


def verify(result):
    sweep, ladder, limits = result["sweep"], result["ladder"], limits_of(result["sweep"])
    (bug, hot, good), peaks, top = [result[k] for k, _, _ in ARM_SPECS], peaks_of(sweep), GRID[-1]
    return [
        practice.Check(
            "ANSWER: 1e-1 on the backbone does not explode -- it trains",
            bug["last"] < bug["first"] and bug["peak"] < 2 * bug["first"],
            f"the bug as written -- the lesson's six groups with the ladder reversed, "
            f"{ladder[0][0]} at {BUG_LR} and fc at {BUG_LR * DECAY ** 5:.2e} -- takes the loss "
            f"{bug['first']:.2f} -> {bug['last']:.2f} over {ARM_STEPS} steps, peaking at "
            f"{bug['peak']:.2f}, {bug['peak'] / bug['first']:.2f}x its first step"),
        practice.Check(
            "CONTROL: the helper is not the recovery -- at base_lr 1e-1 it explodes too",
            hot["peak"] > 5 * hot["first"] and good["last"] < 0.6 * good["first"],
            f"discriminative_param_groups(base_lr={BUG_LR}) peaks at {hot['peak']:.1f}, "
            f"{hot['peak'] / hot['first']:.0f}x its first step, because the ladder hands its "
            f"*largest* LR to fc; at the lesson's own {GOOD_LR} it takes the loss "
            f"{good['first']:.2f} -> {good['last']:.2f} instead. The recovery is the base_lr"),
        practice.Check(
            "ANSWER: fc diverges at 1e-1; no backbone stage diverges anywhere on the grid",
            head_only(limits),
            f"smallest LR in {GRID} whose peak loss exceeds {EXPLODE:g}x the first step's, one "
            f"stage in the optimiser at a time, {SWEEP_STEPS} steps each: fc at {limits['fc']:g}; "
            f"none of {', '.join(BACKBONE)} at any LR up to {top:g}, "
            f"{top / limits['fc']:.0f}x past fc's threshold"),
        practice.Check(
            "MECHANISM: BatchNorm makes conv weights scale-free; the head has no BatchNorm",
            worst(sweep, top) < 0.05 * peaks[-1] and decade(peaks),
            f"at LR {top:g} the worst backbone peak is {worst(sweep, top):.2f} while fc peaks at "
            f"{peaks[-1]:.0f}. fc's peak by LR: {GRID[1]:g} -> {peaks[0]:.1f}, {GRID[2]:g} -> "
            f"{peaks[1]:.1f}, {GRID[3]:g} -> {peaks[2]:.1f}, rising {peaks[1] / peaks[0]:.1f}x then "
            f"{peaks[2] / peaks[1]:.1f}x per decade -- the logits are linear in fc's weights, while "
            "scaling a conv only rotates what BatchNorm renormalises"),
        practice.Check(
            "CONTROL: the backbone's tolerance is not indifference -- it still learns at LR 10",
            tail(sweep) < 0.8 * sweep["layer2"][top]["first"],
            f"with only that stage in the optimiser at LR {top:g}, the worst of layer2, layer3 and "
            f"layer4 ends at {tail(sweep):.2f} after {SWEEP_STEPS} steps, down from "
            f"{sweep['layer2'][top]['first']:.2f} -- an LR {top / limits['fc']:.0f}x past the one "
            "that destroys fc, and they still learn"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
