"""Exercise 3 — about 128 in-domain samples is enough, and padding with generic text undoes it.

    Compute the calibration-dataset size needed to calibrate AWQ for a medical
    domain model. Why is more data not always better?

Reading of the exercise: the lesson's `code/` has no calibration at all, so
the size is measured on a scaled-down AWQ run over one layer. The layer is
16 x 256 Gaussian weights, INT4 round-to-nearest with a zero point, group
128. Per-input-channel scales are s = mean|x|^alpha from the calibration
activations, with alpha grid-searched over 21 values the way the AWQ paper
does it. Medical and generic text each light up 3 of 256 input channels
(~1%, the lesson's "~1% most-salient weights"): 5% of tokens spike to
sigma 40, and the two domains use disjoint channels. A sample is 8 tokens.
Error is the exact expected output MSE on medical traffic, relative to plain
round-to-nearest, averaged over 5 seeded calibration draws. The numbers are
properties of this toy. The shape of the curve is the transferable part.

**ANSWER: about 128 samples, set by how often the outlier channels fire, not
by a fixed count.** Relative error by in-domain sample count: 0.93 at 1,
0.81 at 4, 0.72 at 16, 0.69 at 32, 0.68 at 64, 0.67 at 128, 0.66 at 512. 128
is the smallest count within 2% of the 512-sample plateau; the seed-to-seed
spread there is 0.04, against 0.34 at 1 sample. What 128 buys is about 51
observed spikes per salient channel (128 x 8 x 0.05). The general rule is
tokens ~ 50 / (outlier rate). That makes "hundreds" (lesson) and "500-2000"
(the Ship It skill) the same rule at different outlier rates, not a number
to copy. The AWQ paper itself reports good perplexity from 16 sequences,
against GPTQ's 192.

**FINDING: more data is not better when it is not domain data.** Keep the
128 medical samples and add generic ones: +128 generic -> 0.74, +512 -> 0.93,
+1152 -> 0.98. At 10x the data, nearly all of AWQ's gain is gone, because the
scales follow the average activation and the generic spikes compete with the
medical ones in the alpha search. Generic-only calibration scores 1.18,
*worse* than not calibrating. Once in-domain data has converged it also
stops paying: 512 samples improve on 128 by 1.5%, for 4x the calibration
cost.

**FINDING: this toy is harsher about mismatch than the paper.** AWQ reports
+0.5-0.6 perplexity when calibrating on the wrong one of PubMed and Enron, and
GPTQ +2.3-4.9. Here the two domains share no salient channels, so this is
the worst case for mismatch. The ordering it shows (in-domain < padded <
none < generic) is the claim, not the magnitudes.

Structure: `calibrate()` is the scale search; `error()` is exact because the
toy activations are independent, E[(dW x)^2] = sum dW^2 E[x^2].
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "09-production-quantization"
D_IN, D_OUT, GROUP, TOKENS, FIRE, SPIKE = 256, 16, 128, 8, 0.05, 40.0
MED, GEN = (5, 70, 200), (11, 90, 150)
ALPHAS = [i / 20 for i in range(21)]
SIZES, PADS, SEEDS = (1, 4, 16, 32, 64, 128, 256, 512), (0, 128, 512, 1152), 5


def token(rng, salient):
    return [rng.gauss(0, SPIKE if j in salient and rng.random() < FIRE else 1.0)
            for j in range(D_IN)]


def moments(samples):
    """Per-channel mean |x| and mean x^2."""
    n, cols = len(samples), list(zip(*samples))
    return [sum(map(abs, c)) / n for c in cols], [sum(v * v for v in c) / n for c in cols]


def quantize(weights, scale):
    """INT4 with a zero point, per row and group of 128, on W * s, then undone."""
    out = []
    for row in weights:
        q = []
        for g in range(0, D_IN, GROUP):
            ws = [row[j] * scale[j] for j in range(g, g + GROUP)]
            lo, step = min(ws), (max(ws) - min(ws)) / 15
            q += [(lo + round((w - lo) / step) * step) / scale[g + i] for i, w in enumerate(ws)]
        out.append(q)
    return out


def error(weights, quant, second):
    return sum((w - q) ** 2 * second[j] for row, qrow in zip(weights, quant)
               for j, (w, q) in enumerate(zip(row, qrow)))


def calibrate(weights, samples):
    absmean, second = moments(samples)
    scored = []
    for alpha in ALPHAS:
        quant = quantize(weights, [max(a, 1e-6) ** alpha for a in absmean])
        scored.append((error(weights, quant, second), alpha, quant))
    return min(scored)[2]


def relative(weights, truth, base, medical, generic):
    """Mean and spread of medical-traffic error over SEEDS calibration draws."""
    runs = []
    for seed in range(SEEDS):
        rng = random.Random(1000 + seed)
        samples = [token(rng, MED) for _ in range(medical * TOKENS)]
        samples += [token(rng, GEN) for _ in range(generic * TOKENS)]
        runs.append(error(weights, calibrate(weights, samples), truth) / base)
    return round(sum(runs) / SEEDS, 4), round(max(runs) - min(runs), 4)


def layer():
    """Seeded weights and the exact per-channel E[x^2] of medical traffic."""
    rng = random.Random(0)
    weights = [[rng.gauss(0, 1) for _ in range(D_IN)] for _ in range(D_OUT)]
    return weights, [FIRE * SPIKE**2 + (1 - FIRE) if j in MED else 1.0 for j in range(D_IN)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    weights, truth = layer()
    base = error(weights, quantize(weights, [1.0] * D_IN), truth)
    sizes = {n: relative(weights, truth, base, n, 0) for n in SIZES}
    plateau = sizes[SIZES[-1]][0]
    return {
        "sizes": sizes,
        "knee": next(n for n in SIZES if sizes[n][0] <= plateau * 1.02),
        "padded": {p: relative(weights, truth, base, 128, p)[0] for p in PADS},
        "generic_only": relative(weights, truth, base, 0, 128)[0],
        "ref_has_calibration": any("calib" in name.lower() for name in dir(ref)),
    }


def verify(result):
    sizes, pad = result["sizes"], result["padded"]
    means = [sizes[n][0] for n in SIZES]
    spikes = round(result["knee"] * TOKENS * FIRE)
    return [
        practice.Check(
            "ANSWER: about 128 samples, set by how often the outlier channels fire",
            result["knee"] == 128 and means == sorted(means, reverse=True)
            and sizes[128][1] < sizes[1][1] / 5 and not result["ref_has_calibration"],
            f"relative error by samples {sizes}; knee {result['knee']} (within 2% of the "
            f"512 plateau) = {spikes} spikes per salient channel; code/ has no calibration",
        ),
        practice.Check(
            "FINDING: more data is not better when it is not domain data",
            list(pad.values()) == sorted(pad.values()) and pad[1152] > 0.95
            and result["generic_only"] > 1.0 and means[-1] > 0.98 * sizes[128][0],
            f"128 medical + generic padding {pad}; generic only {result['generic_only']} "
            f"(>1 = worse than no calibration); 512 vs 128 in-domain "
            f"{sizes[512][0]} vs {sizes[128][0]}",
        ),
        practice.Check(
            "FINDING: this toy is harsher about mismatch than the paper",
            sizes[128][0] < pad[128] < 1.0 < result["generic_only"],
            "ordering in-domain < padded < no calibration (1.0) < generic holds; AWQ's own "
            "PubMed/Enron swap costs only +0.5-0.6 perplexity, so magnitudes do not transfer",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
