"""Exercise 5 — the law is model-independent in error, not in accuracy.

    Read the OpenCLIP scaling-laws paper (arXiv:2212.07143, Cherti et al.).
    Reproduce their conclusion for data scaling from the figures: at fixed model
    size, what is the log-linear relationship between ImageNet zero-shot
    accuracy and training data size?

Reading of the exercise: the figures are not reachable from here, so the fit is
built from the four ImageNet zero-shot top-1 numbers published on the OpenCLIP
checkpoint cards -- ViT-B/32 and ViT-L/14, each at LAION-400M and LAION-2B --
transcribed as constants and labelled as such. Everything asserted below rests
on *ratios* of those numbers rather than their absolute values, and the fit is
run in both of the paper's two framings so they can be compared rather than
assumed equivalent.

**ANSWER: at fixed model size, +5.29 accuracy points per decade of data.**
ViT-B/32 goes 62.9 -> 66.6 across a 5x data increase, which is 0.699 decades.
ViT-L/14 goes 72.8 -> 75.3, giving **+3.58** points per decade.

**ANSWER: those two slopes disagree by 32.4%, and the same step in error does
not.** Error falls by a factor **0.900** for B/32 and **0.908** for L/14 over
the identical data step -- agreement to **0.87%**. Stated as a power law,
error ~ D^-0.065 and D^-0.060. The conclusion is model-independent in error and
model-dependent in accuracy, which is why the paper states it as a power law.

**FINDING: the log-linear accuracy form names a finite crossing.** Extrapolated,
it reaches 100% top-1 at **4.1e15** images for B/32 and **1.6e16** for L/14 --
numbers, not infinities. The power law in error never reaches 100%. Two fits to
the same two points differ in what they say is possible.

**FINDING: and nothing reachable distinguishes them.** At 100x LAION-2B --
200 billion images, 100x the largest run in the paper -- the two forms differ by
**1.9** points for B/32 and **1.2** for L/14. The choice between them is settled
by the asymptote, not by any measurement anyone will make.

**FINDING: the step is not a data-size step.** The two checkpoints say so in
their own names: `laion400m_e32` is 32 epochs of 400M = 12.8B samples seen,
`laion2b_s34b_b79k` is 34B. Crediting the whole 3.7 points to 5x the data means
crediting it with a **2.66x** compute increase as well.

Structure: `CARDS` is the transcribed table, `fit` returns both framings for one
model, `predict` extrapolates each framing, and `crossing` solves the linear
form for 100%.
"""

from __future__ import annotations

import math

from harness import practice

# Transcribed from the OpenCLIP checkpoint cards: ImageNet-1k zero-shot top-1 (%)
# for the LAION-400M and LAION-2B runs of each model, plus samples seen (billions).
CARDS = {
    "ViT-B/32": {"small": 62.9, "large": 66.6, "samples": (12.8, 34.0)},
    "ViT-L/14": {"small": 72.8, "large": 75.3, "samples": (12.8, 34.0)},
}
SMALL_M, LARGE_M = 400.0, 2000.0          # LAION-400M and LAION-2B, in millions
EXTRAPOLATION = 100.0                      # multiples of LAION-2B


def decades(low, high):
    return math.log10(high / low)


def fit(card):
    """Both framings of the same two points: a line in accuracy, a power law in error."""
    span = decades(SMALL_M, LARGE_M)
    low, high = card["small"], card["large"]
    ratio = (100 - high) / (100 - low)
    return {"slope": round((high - low) / span, 3),
            "error_ratio": round(ratio, 5),
            "alpha": round(-math.log10(ratio) / span, 4)}


def predict(card, shape, factor=EXTRAPOLATION):
    """Each framing's accuracy at `factor` times LAION-2B."""
    span = decades(1.0, factor)
    linear = card["large"] + shape["slope"] * span
    power = 100 - (100 - card["large"]) * 10 ** (-shape["alpha"] * span)
    return round(linear, 1), round(power, 1)


def crossing(card, shape):
    """The data size at which the linear form reaches 100% top-1, in images."""
    return SMALL_M * 1e6 * 10 ** ((100 - card["small"]) / shape["slope"])


def solve():
    shapes = {name: fit(card) for name, card in CARDS.items()}
    slopes = [shape["slope"] for shape in shapes.values()]
    ratios = [shape["error_ratio"] for shape in shapes.values()]
    samples = CARDS["ViT-B/32"]["samples"]
    return {
        "span": round(decades(SMALL_M, LARGE_M), 3), "shapes": shapes,
        "slope_gap_pct": round(abs(min(slopes) / max(slopes) - 1) * 100, 1),
        "error_gap_pct": round(abs(max(ratios) / min(ratios) - 1) * 100, 2),
        "gains": {name: round(card["large"] - card["small"], 1)
                  for name, card in CARDS.items()},
        "crossings": {name: crossing(CARDS[name], shape)
                      for name, shape in shapes.items()},
        "far": {name: predict(CARDS[name], shape) for name, shape in shapes.items()},
        "compute_factor": round(samples[1] / samples[0], 2),
    }


def verify(result):
    shapes, far = result["shapes"], result["far"]
    return [
        practice.Check(
            "ANSWER: at fixed model size, +5.29 accuracy points per decade of data",
            all([shapes["ViT-B/32"]["slope"] == 5.294, shapes["ViT-L/14"]["slope"] == 3.577,
                 result["span"] == 0.699, result["gains"] == {"ViT-B/32": 3.7,
                                                              "ViT-L/14": 2.5}]),
            f"ViT-B/32 goes 62.9 -> 66.6 and ViT-L/14 72.8 -> 75.3 across a 5x data "
            f"increase, {result['span']} decades, so the slopes are "
            f"{ {n: s['slope'] for n, s in shapes.items()} } points per decade",
        ),
        practice.Check(
            "ANSWER: the two slopes disagree by 32.4%, and the same step in error does not",
            all([result["slope_gap_pct"] == 32.4, result["error_gap_pct"] == 0.87,
                 shapes["ViT-B/32"]["error_ratio"] == 0.90027,
                 shapes["ViT-L/14"]["error_ratio"] == 0.90809,
                 shapes["ViT-B/32"]["alpha"] == 0.0653]),
            f"over the identical data step the error falls by "
            f"{ {n: s['error_ratio'] for n, s in shapes.items()} } -- agreement to "
            f"{result['error_gap_pct']}% -- while the points-per-decade slopes disagree by "
            f"{result['slope_gap_pct']}%. As a power law that is error ~ D^-"
            f"{shapes['ViT-B/32']['alpha']} and D^-{shapes['ViT-L/14']['alpha']}",
        ),
        practice.Check(
            "FINDING: the log-linear accuracy form names a finite crossing",
            all([round(result["crossings"]["ViT-B/32"] / 1e15, 1) == 4.1,
                 round(result["crossings"]["ViT-L/14"] / 1e16, 1) == 1.6]),
            f"extrapolated, the line reaches 100% top-1 at "
            f"{result['crossings']['ViT-B/32']:.1e} images for B/32 and "
            f"{result['crossings']['ViT-L/14']:.1e} for L/14 -- numbers, not infinities. "
            "The power law in error never reaches 100%, so two fits to the same two points "
            "disagree about what is possible",
        ),
        practice.Check(
            "FINDING: and nothing reachable distinguishes them",
            all([far["ViT-B/32"] == (77.2, 75.3), far["ViT-L/14"] == (82.5, 81.3),
                 round(far["ViT-B/32"][0] - far["ViT-B/32"][1], 1) == 1.9]),
            f"at {EXTRAPOLATION:.0f}x LAION-2B -- 200 billion images -- the line predicts "
            f"{far['ViT-B/32'][0]}% and the power law {far['ViT-B/32'][1]}% for B/32, "
            f"{far['ViT-L/14'][0]}% and {far['ViT-L/14'][1]}% for L/14. Gaps of 1.9 and 1.2 "
            "points: the choice is settled by the asymptote, not by a measurement",
        ),
        practice.Check(
            "FINDING: the step is not a data-size step",
            result["compute_factor"] == 2.66,
            f"the checkpoints name their own budgets: laion400m_e32 is 32 epochs of 400M = "
            f"12.8B samples seen, laion2b_s34b_b79k is 34B. Crediting the 3.7 points to 5x "
            f"the data credits it with a {result['compute_factor']}x compute increase too",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
