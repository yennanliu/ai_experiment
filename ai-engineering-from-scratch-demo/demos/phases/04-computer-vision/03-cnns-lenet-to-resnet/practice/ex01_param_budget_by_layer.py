"""Exercise 1 — param budget by layer.

    **(Easy)** Count parameters by hand for `TinyResNet` layer by layer. Compare
    against `sum(p.numel() for p in net.parameters())`. Where does the majority
    of the parameter budget go — convs, BN, or the classifier head?

Reading of the exercise: "by hand" means a closed-form predictor written from
the architecture alone -- 9*Cin*Cout per 3x3 conv, 2*C per BatchNorm, the 1x1
projection where a group changes shape -- evaluated without touching the built
module, then compared group by group against `sum(p.numel() ...)`. Matching only
the grand total would hide a compensating pair of errors, so every group is
compared separately. The "where does it go" question is answered three ways:
by module type (the literal question), by group (which shows the answer is
really "the last group"), and against LeNet-5 (which shows the answer is a
property of the global-average-pool head, not of CNNs).
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "03-cnns-lenet-to-resnet"

GROUPS = (("layer1", 32, 32, 1), ("layer2", 32, 64, 2),
          ("layer3", 64, 128, 2), ("layer4", 128, 256, 2))
BLOCKS, SIDE = 2, 32


def hand_count(in_c, out_c, stride) -> int:
    """One BasicBlock, from the formula: two 3x3 convs, two BNs, maybe a 1x1."""
    convs = 9 * in_c * out_c + 2 * out_c + 9 * out_c * out_c + 2 * out_c
    return convs + ((in_c * out_c + 2 * out_c) if (stride != 1 or in_c != out_c) else 0)


def predicted() -> dict:
    out = {"stem": 27 * 32 + 2 * 32, "head": 256 * 10 + 10}
    for name, in_c, out_c, stride in GROUPS:
        out[name] = hand_count(in_c, out_c, stride) + hand_count(out_c, out_c, 1) * (BLOCKS - 1)
    return out


def measured(net) -> dict:
    return {name: sum(p.numel() for p in mod.parameters()) for name, mod in net.named_children()}


def by_type(nn, net) -> dict:
    """Parameters grouped by the module that owns them -- the exercise's question."""
    wanted = {"Conv2d": nn.Conv2d, "BatchNorm2d": nn.BatchNorm2d, "Linear": nn.Linear}
    return {key: sum(p.numel() for m in net.modules() if isinstance(m, cls)
                     for p in m.parameters()) for key, cls in wanted.items()}


def macs(torch, nn, net) -> dict:
    """Multiply-accumulates per group, so the budget can be read against the compute."""
    seen, current = {}, [""]
    record = lambda m, _i, o: seen.__setitem__(          # noqa: E731 - a hook, not a helper
        current[0], seen.get(current[0], 0) + m.weight.numel()
        * (o.shape[-1] * o.shape[-2] if isinstance(m, nn.Conv2d) else 1))
    for module in net.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module.register_forward_hook(record)
    x = torch.zeros(1, 3, SIDE, SIDE)
    with torch.no_grad():
        for name, group in net.named_children():
            current[0] = name
            x = group(x)
    return seen


def solve():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.manual_seed(0)
    net, lenet = ref.TinyResNet(), ref.LeNet5()
    net.eval()
    return {"predicted": predicted(), "measured": measured(net), "by_type": by_type(nn, net),
            "total": sum(p.numel() for p in net.parameters()), "macs": macs(torch, nn, net),
            "buffers": sum(b.numel() for b in net.buffers()), "lenet_fc": sum(
                p.numel() for m in lenet.modules() if isinstance(m, nn.Linear) for p in m.parameters()),
            "lenet_total": sum(p.numel() for p in lenet.parameters())}


def verify(result):
    kinds, total, groups = result["by_type"], result["total"], result["measured"]
    share = {k: v / total for k, v in kinds.items()}
    off = {k: result["predicted"][k] - v for k, v in groups.items()}
    flops, lenet = result["macs"], result["lenet_fc"] / result["lenet_total"]
    return [
        practice.Check(
            "ANSWER: the convs hold it, by three orders of magnitude",
            share["Conv2d"] > 0.99 > share["BatchNorm2d"] + share["Linear"] and sum(kinds.values()) == total,
            f"of {total:,} parameters: convs {kinds['Conv2d']:,} ({share['Conv2d']:.2%}), BatchNorm "
            f"{kinds['BatchNorm2d']:,} ({share['BatchNorm2d']:.2%}), classifier head {kinds['Linear']:,} "
            f"({share['Linear']:.2%}) -- the three account for every parameter"),
        practice.Check(
            "ANSWER: the hand count matches every group exactly, not just the total",
            set(off.values()) == {0},
            "predicted vs measured per group: " + ", ".join(f"{k} {result['predicted'][k]:,} (off by {v})"
            for k, v in off.items()) + f"; hand total {sum(result['predicted'].values()):,} vs {total:,}"),
        practice.Check(
            "FINDING: 'the convs' really means 'the last group'",
            groups["layer4"] > total - groups["layer4"],
            f"layer4 alone is {groups['layer4']:,} = {groups['layer4'] / total:.1%} of the net, more than the "
            f"other five children combined ({total - groups['layer4']:,}); the four groups run "
            + " -> ".join(f"{groups[n]:,}" for n, *_ in GROUPS)),
        practice.Check(
            "MECHANISM: parameters quadruple per group while the compute stays flat",
            groups["layer4"] > 50 * groups["layer1"] and flops["layer4"] < flops["layer1"],
            f"layer1 -> layer4 is {groups['layer4'] / groups['layer1']:.0f}x the parameters and "
            f"{flops['layer4'] / flops['layer1']:.2f}x the multiply-accumulates "
            f"({flops['layer1'] / 1e6:.1f}M -> {flops['layer4'] / 1e6:.1f}M): doubling channels costs 4x the "
            "weights, and the stride-2 that comes with it divides H*W by 4"),
        practice.Check(
            "CONTROL: LeNet-5 gives the opposite answer, so this is about the head",
            lenet > 0.9,
            f"LeNet-5 puts {result['lenet_fc']:,} of {result['lenet_total']:,} ({lenet:.1%}) in its "
            f"fully-connected head, against {share['Linear']:.2%} here; TinyResNet's AdaptiveAvgPool2d(1) "
            "collapses 256x4x4 to 256 before the Linear, which is what deletes the classifier's budget"),
        practice.Check(
            "CONTROL: sum(p.numel()) is not the checkpoint size",
            result["buffers"] > kinds["BatchNorm2d"],
            f"the BatchNorms also carry {result['buffers']:,} non-learnable numbers (running_mean/var/"
            f"num_batches) that no parameter count sees -- more than their {kinds['BatchNorm2d']:,} learnable "
            f"ones, and {result['buffers'] / total:.2%} of the total that still has to be saved and shipped"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
