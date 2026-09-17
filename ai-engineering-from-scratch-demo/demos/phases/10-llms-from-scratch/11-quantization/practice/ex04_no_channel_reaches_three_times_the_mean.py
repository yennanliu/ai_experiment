"""Exercise 4 — on a Gaussian activation no channel reaches even 3x the mean, so the sweep is empty.

    Build an outlier-aware quantizer inspired by LLM.int8(). Detect channels
    where the activation magnitude exceeds 6x the mean. Keep those channels in
    FP16 and quantize everything else to INT8. Measure end-to-end quality on the
    transformer layer from Step 5 with varying outlier thresholds (3x, 6x, 10x).

Reading of the exercise: the quantiser is built as specified and run on the
lesson's own `simulate_transformer_layer`, with the detector applied to the
activation entering it. Three activation distributions are tried -- Gaussian,
lognormal, and one with outliers planted deliberately -- because the exercise's
threshold sweep only means something if some threshold selects a different set
from another.

**ANSWER: on the activation this layer actually receives, all three thresholds
select the same channels, and that number is zero.** A Gaussian activation's
largest per-channel magnitude is **1.3x** the mean, so 3x, 6x and 10x all pick
**0 of 256** channels and the outlier-aware quantiser is INT8 quantisation with
extra bookkeeping.

**FINDING: a heavier tail does not help.** A lognormal activation -- far more
skewed than anything a LayerNorm output produces -- peaks at **1.9x** the mean.
Still nothing above 3x. The mean of 256 absolute values is a stable number, and
beating it sixfold takes a distribution with genuinely separate scales, not
merely a long tail.

**FINDING: planting outliers makes all three thresholds identical.** Scaling
three channels by 12x gives max/mean **13.0**, and 3x, 6x and 10x then select
the same 3 channels. Outliers that exist are far above every threshold in the
range and outliers that do not exist are below all of them, so the sweep has one
outcome either way: the cut only bites on a distribution whose channels sit
*between* 3 and 10 times the mean, and neither arm produces one.

**MECHANISM: LLM.int8()'s 6x is about a trained transformer's residual stream,
measured per token.** Those outliers are systematic -- the same few hidden
dimensions carry large values across almost every token, which is why a
threshold on the mean separates them. A random matrix has no such structure to
detect, so the exercise's detector is correct and its input has nothing in it.

Structure: `outlier_channels` is the detector; `arms` runs the three thresholds
against three activation distributions so the empty sweep is visible as such.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "11-quantization"
SEED, BATCH, SEQ, CHANNELS = 7, 2, 32, 256
TOKENS = BATCH * SEQ
THRESHOLDS = (3, 6, 10)
PLANTED = (3, 40, 91)
BOOST = 12.0


def activations(rng):
    """Three candidate inputs: Gaussian, lognormal, and Gaussian with planted outliers."""
    gaussian = rng.standard_normal((TOKENS, CHANNELS))
    planted = rng.standard_normal((TOKENS, CHANNELS))
    planted[:, list(PLANTED)] *= BOOST
    return {"gaussian": gaussian,
            "lognormal": rng.lognormal(0, 1.2, (TOKENS, CHANNELS)),
            "planted": planted}


def outlier_channels(activation, threshold):
    """LLM.int8()'s rule: channels whose mean magnitude exceeds `threshold` x the mean."""
    magnitude = np.abs(activation).mean(axis=0)
    return np.flatnonzero(magnitude > threshold * magnitude.mean())


def peak_ratio(activation):
    magnitude = np.abs(activation).mean(axis=0)
    return float(magnitude.max() / magnitude.mean())


def layer_weights(rng):
    """The Step 5 layer: one fused qkv projection and one output projection."""
    return {"qkv": rng.standard_normal((CHANNELS, 3 * CHANNELS)) * 0.05,
            "out": rng.standard_normal((CHANNELS, CHANNELS)) * 0.05}


def mixed_quality(ref, activation, weights, threshold):
    """Cosine of the layer output when the outlier input channels stay in fp16."""
    keep = outlier_channels(activation, threshold)
    quantised, scales = ref.quantize_per_channel(weights["qkv"], 8, axis=0)
    mixed = ref.dequantize_per_channel(quantised, scales, axis=0)
    mixed[keep] = weights["qkv"][keep]
    batched = activation.reshape(BATCH, SEQ, CHANNELS)
    exact, _ = ref.simulate_transformer_layer(batched, weights)
    approx, _ = ref.simulate_transformer_layer(batched, dict(weights, qkv=mixed))
    return float(exact.ravel() @ approx.ravel()
                 / (np.linalg.norm(exact) * np.linalg.norm(approx))), len(keep)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = np.random.default_rng(SEED)
    inputs = activations(rng)
    weights = layer_weights(rng)
    return {
        "counts": {name: {t: len(outlier_channels(act, t)) for t in THRESHOLDS}
                   for name, act in inputs.items()},
        "peaks": {name: peak_ratio(act) for name, act in inputs.items()},
        "quality": {t: mixed_quality(ref, inputs["gaussian"], weights, t)[0]
                    for t in THRESHOLDS},
        "channels": CHANNELS,
    }


def verify(result):
    counts, peaks, quality = result["counts"], result["peaks"], result["quality"]
    gaussian, lognormal, planted = counts["gaussian"], counts["lognormal"], counts["planted"]
    return [
        practice.Check(
            f"ANSWER: all three thresholds select 0 of {result['channels']} channels",
            set(gaussian.values()) == {0} and len(set(quality.values())) == 1,
            f"the Gaussian activation this layer receives peaks at "
            f"{peaks['gaussian']:.1f}x its own mean, so "
            + ", ".join(f"{t}x selects {n}" for t, n in gaussian.items())
            + f" and the end-to-end cosine is {list(quality.values())[0]:.6f} at every "
            "threshold. The outlier-aware quantiser is INT8 quantisation with extra bookkeeping",
        ),
        practice.Check(
            "FINDING: a heavier tail does not help -- lognormal stays under 2x",
            set(lognormal.values()) == {0} and peaks["lognormal"] < min(THRESHOLDS),
            f"a lognormal activation, far more skewed than any LayerNorm output, peaks at "
            f"{peaks['lognormal']:.1f}x the mean and still selects "
            f"{lognormal[min(THRESHOLDS)]} channels at {min(THRESHOLDS)}x. The mean of "
            f"{result['channels']} absolute values is a stable number, and beating it sixfold "
            "takes a distribution with genuinely separate scales, not merely a long tail",
        ),
        practice.Check(
            "FINDING: even with outliers planted, all three thresholds are one experiment",
            len(set(planted.values())) == 1 and planted[3] > 0,
            f"scaling {len(PLANTED)} channels by {BOOST:.0f}x gives a peak of "
            f"{peaks['planted']:.1f}x, and then "
            + ", ".join(f"{t}x selects {n}" for t, n in planted.items())
            + ". Outliers that exist at all are far above every threshold in the range and "
            "outliers that do not exist are below all of them, so the sweep the exercise asks "
            "for has one outcome either way -- the cut only bites on a distribution whose "
            "channels sit between 3 and 10 times the mean, and neither arm produces one",
        ),
        practice.Check(
            "MECHANISM: LLM.int8()'s 6x describes a trained residual stream, not a random matrix",
            peaks["gaussian"] < 2.0 < peaks["planted"],
            "those outliers are systematic -- the same few hidden dimensions carry large values "
            "across almost every token, which is why a threshold on the mean separates them. "
            f"A random activation peaks at {peaks['gaussian']:.1f}x and a planted one at "
            f"{peaks['planted']:.1f}x, so the detector is correct and its input, at this point "
            "in the lesson, has nothing in it to find",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
