"""Exercise 2 — bottleneck resnet50.

    **(Medium)** Implement the Bottleneck block (1x1 -> 3x3 -> 1x1 with skip) and
    use it to build a ResNet-50-style network for CIFAR. Compare params against
    `TinyResNet`.

Reading of the exercise: "ResNet-50-style **for CIFAR**" is the constraint that
decides the architecture -- the [3, 4, 6, 3] block layout and the 4x expansion
are ResNet-50's, but the 7x7/stride-2 stem and the max-pool are ImageNet's and
would throw away three quarters of a 32x32 image before the first block, so the
stem becomes a 3x3/stride-1 conv and the first group keeps stride 1. "Compare
params" is read as a comparison that survives a follow-up question: the raw
ratio, then a width-matched build that shows how much of the gap is the block
and how much is just the wider channels, then the multiply-accumulates the
parameter count does not measure.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "03-cnns-lenet-to-resnet"

EXPANSION, SIDE, CLASSES = 4, 32, 10
WIDTHS, BLOCKS = (64, 128, 256, 512), (3, 4, 6, 3)


def bottleneck_class(nn, functional):
    """The block, as a factory: `nn` only exists after the guarded import."""
    class Bottleneck(nn.Module):
        def __init__(self, in_c, width, stride=1):
            super().__init__()
            out_c = width * EXPANSION
            self.conv1 = nn.Conv2d(in_c, width, 1, bias=False)
            self.conv2 = nn.Conv2d(width, width, 3, stride=stride, padding=1, bias=False)
            self.conv3 = nn.Conv2d(width, out_c, 1, bias=False)
            self.bn1, self.bn2 = nn.BatchNorm2d(width), nn.BatchNorm2d(width)
            self.bn3 = nn.BatchNorm2d(out_c)
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)) if stride != 1 or in_c != out_c else nn.Identity()

        def forward(self, x):
            out = functional.relu(self.bn1(self.conv1(x)))
            out = functional.relu(self.bn2(self.conv2(out)))
            return functional.relu(self.bn3(self.conv3(out)) + self.shortcut(x))

    return Bottleneck


def build(nn, block, widths) -> object:
    """ResNet-50's [3, 4, 6, 3] layout on a CIFAR stem: 3x3 stride 1, no max-pool."""
    layers = [nn.Sequential(nn.Conv2d(3, widths[0], 3, padding=1, bias=False),
                            nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))]
    in_c = widths[0]
    for group, (width, repeats) in enumerate(zip(widths, BLOCKS)):
        for index in range(repeats):
            layers.append(block(in_c, width, stride=2 if index == 0 and group else 1))
            in_c = width * EXPANSION
    layers.append(nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_c, CLASSES)))
    return nn.Sequential(*layers)


def count(module) -> int:
    return sum(p.numel() for p in module.parameters())


def residual_gap(torch, nn, block) -> float:
    """Zero the last BN and a real residual block must collapse onto its shortcut."""
    probe, x = block(256, 64), torch.randn(2, 256, 8, 8)
    nn.init.zeros_(probe.bn3.weight)
    nn.init.zeros_(probe.bn3.bias)
    probe.eval()
    with torch.no_grad():
        return float((probe(x) - torch.relu(probe.shortcut(x))).abs().max())


def solve():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.manual_seed(0)
    block = bottleneck_class(nn, functional)
    net, tiny = build(nn, block, WIDTHS), ref.TinyResNet()
    thin, wide = build(nn, block, tuple(w // 3 for w in WIDTHS)), block(256, 64)
    convs = lambda m: sum(1 for x in m.modules() if isinstance(x, nn.Conv2d))  # noqa: E731
    with torch.no_grad():
        shape = tuple(net.eval()(torch.zeros(2, 3, SIDE, SIDE)).shape)
    return {"params": count(net), "tiny": count(tiny), "thin": count(thin), "shape": shape,
            "gap": residual_gap(torch, nn, block), "basic": count(ref.BasicBlock(256, 256)),
            "bottleneck": count(wide), "convs": convs(net), "thin_convs": convs(thin),
            "tiny_convs": convs(tiny),
            "inner": [c.weight.numel() for c in (wide.conv1, wide.conv2, wide.conv3)]}


def verify(result):
    ratio, inner = result["params"] / result["tiny"], result["inner"]
    cheaper = result["basic"] / result["bottleneck"]
    return [
        practice.Check(
            "ANSWER: 23.5M against TinyResNet's 2.8M -- 8.4x, over 2.6x the conv layers",
            result["shape"] == (2, CLASSES) and 8 < ratio < 9,
            f"the [3, 4, 6, 3] bottleneck net has {result['params']:,} parameters against TinyResNet's "
            f"{result['tiny']:,} ({ratio:.2f}x) over {result['convs']} convs against {result['tiny_convs']}"),
        practice.Check(
            "ANSWER: it is a residual block, not just a stack that looks like one",
            result["gap"] == 0.0 and result["shape"] == (2, CLASSES),
            "zero the last BatchNorm and the 1x1 -> 3x3 -> 1x1 branch vanishes: the block then equals "
            f"relu(shortcut(x)) to {result['gap']} max abs difference, so a block can start as a no-op"),
        practice.Check(
            "FINDING: at the same input and output width the bottleneck is 17x cheaper",
            cheaper > 15,
            f"BasicBlock(256 -> 256) costs {result['basic']:,} parameters; a Bottleneck through width 64 back "
            f"to 256 costs {result['bottleneck']:,} ({cheaper:.1f}x less) for the same interface"),
        practice.Check(
            "MECHANISM: the only 3x3 runs at a quarter width",
            inner[1] > inner[0] == inner[2] and sum(inner) < 9 * 256 * 256,
            f"the weights split 1x1 {inner[0]:,} / 3x3 {inner[1]:,} / 1x1 {inner[2]:,}: the 3x3 is 9*(C/4)^2, "
            f"not 9*C^2 = {9 * 256 * 256:,}, and that 16x saving pays for both projections many times over"),
        practice.Check(
            "CONTROL: most of the 8.4x is width, not the block",
            0.9 < result["thin"] / result["tiny"] < 1.0,
            f"dividing every width by 3 gives {result['thin']:,} parameters "
            f"({result['thin'] / result['tiny']:.2f}x TinyResNet) while keeping all {result['thin_convs']} conv "
            f"layers -- {result['thin_convs'] / result['tiny_convs']:.1f}x TinyResNet's depth for the same "
            "budget, so 'ResNet-50 is bigger' is a statement about channels, not about the block"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
