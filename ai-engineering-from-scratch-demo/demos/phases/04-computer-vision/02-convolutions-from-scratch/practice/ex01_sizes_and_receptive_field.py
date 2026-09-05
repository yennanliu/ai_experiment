"""Exercise 1 — output sizes and receptive field down a four-conv stack.

    **(Easy)** Given a 128x128 grayscale input and a stack of `[Conv3x3(s=1,p=1),
    Conv3x3(s=2,p=1), Conv3x3(s=1,p=1), Conv3x3(s=2,p=1)]`, compute the output
    spatial size and the receptive field at each layer by hand. Verify with a
    PyTorch `nn.Sequential` of dummy convs.

Reading of the exercise: checks 1-2 are the two hand calculations, each verified the way the
exercise asks — sizes against a real `nn.Sequential`, and the receptive field against the input
pixels that actually reach one output, read off `autograd`. Checks 3-4 then ask what that
number is worth: inside the image it is heavily weighted toward its own centre, and at the
border it is not 13 at all.
"""

from __future__ import annotations

from harness import parity, practice

try:
    import torch
except ImportError as exc:                       # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None
torch.set_num_threads(1)

PHASE, LESSON = "04-computer-vision", "02-convolutions-from-scratch"
SIDE = 128
LAYERS = ((3, 1, 1), (3, 2, 1), (3, 1, 1), (3, 2, 1))       # kernel, stride, padding


def stack(bias=False):
    net = torch.nn.Sequential(*[torch.nn.Conv2d(1, 1, k, stride=s, padding=p, bias=bias)
                                for k, s, p in LAYERS])
    with torch.no_grad():
        for layer in net:
            layer.weight.fill_(1.0)
    return net


def torch_sizes() -> list:
    x, out = torch.zeros(1, 1, SIDE, SIDE), [SIDE]
    for layer in stack():
        x = layer(x)
        out.append(int(x.shape[-1]))
    return out


def influence(row, col):
    """Which input pixels reach one output, and how strongly — the honest receptive field."""
    x = torch.zeros(1, 1, SIDE, SIDE, requires_grad=True)
    stack()(x)[0, 0, row, col].backward()
    return x.grad[0, 0].numpy()


def extent(numpy, grid) -> tuple:
    rows, cols = numpy.nonzero(grid)
    return int(rows.max() - rows.min() + 1), int(cols.max() - cols.min() + 1)


def solve():
    numpy = parity.try_numpy()
    ref = parity.load_reference(PHASE, LESSON, "main")
    hand, size = [SIDE], SIDE
    for kernel, stride, pad in LAYERS:
        size = ref.output_size(size, kernel, pad, stride)
        hand.append(size)
    fields = [ref.receptive_field([(k, s) for k, s, _p in LAYERS[:n + 1]])
              for n in range(len(LAYERS))]
    middle = influence(hand[-1] // 2, hand[-1] // 2)
    rows, cols = numpy.nonzero(middle)
    window = middle[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    half = window.shape[0] // 2
    corner = influence(0, 0)
    odd = SIDE - 1
    for kernel, stride, pad in LAYERS:
        odd = ref.output_size(odd, kernel, pad, stride)
    return {"hand": hand, "torch": torch_sizes(), "fields": fields, "odd": odd,
            "middle": extent(numpy, middle), "corner": extent(numpy, corner),
            "peak": float(window[half, half]), "edge": float(window[0, 0]),
            "core": float(window[half - 2:half + 3, half - 2:half + 3].sum() / window.sum()),
            "area": 25 / window.size, "reach": float(corner.sum() / middle.sum())}


def verify(result):
    hand, fields = result["hand"], result["fields"]
    return [
        practice.Check("ANSWER: the sizes are 128, 128, 64, 64, 32, and torch agrees",
                       hand == result["torch"] == [128, 128, 64, 64, 32],
                       f"by hand through the lesson's own `output_size`: {hand}; the same stack "
                       f"as an `nn.Sequential` of four `nn.Conv2d`: {result['torch']}. Each "
                       f"stride-2 layer halves and each stride-1 layer with p=1 preserves"),
        practice.Check("ANSWER: the receptive field is 3, 5, 9, 13, and 13 is what actually "
                       "reaches the output",
                       fields == [3, 5, 9, 13] and result["middle"] == (13, 13),
                       f"the lesson's `receptive_field` gives {fields} layer by layer; "
                       f"differentiating one centre output back to the input marks a "
                       f"{result['middle'][0]}x{result['middle'][1]} block of pixels with non-zero "
                       f"gradient. The formula and the measurement agree"),
        practice.Check("FINDING: inside that 13x13 the unit is nowhere near uniform",
                       result["peak"] / result["edge"] > 100 and result["core"] > 2.5 * result["area"],
                       f"with every weight set to 1, the input pixel at the centre of the window "
                       f"carries {result['peak']:.0f} of influence against {result['edge']:.0f} at "
                       f"its corner — {result['peak'] / result['edge']:.0f}x. The central 5x5 is "
                       f"{100 * result['area']:.1f}% of the area and "
                       f"{100 * result['core']:.1f}% of the total weight. The theoretical "
                       f"receptive field is an outer bound, not what the unit integrates"),
        practice.Check("FINDING: at the border it is not 13 either — most of that window is "
                       "padding",
                       result["corner"] == (7, 7) and result["reach"] < 0.3,
                       f"the output at (0, 0) reaches a "
                       f"{result['corner'][0]}x{result['corner'][1]} block, not 13x13, and sums to "
                       f"{result['reach']:.3f} of the influence the centre unit has. The rest of "
                       f"its window fell on the zeros `pad2d` inserted, so 'the receptive field at "
                       f"each layer' is a statement about interior units only"),
        practice.Check("CONTROL: the size chain is not invertible — 127 and 128 both end at 32",
                       result["odd"] == hand[-1],
                       f"running a {SIDE - 1}x{SIDE - 1} input through the same four layers ends "
                       f"at {result['odd']}, exactly where {SIDE} ends. `output_size` floors, so "
                       f"the output shape does not tell you the input shape — which is why "
                       f"decoders take a skip connection rather than recomputing it"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
