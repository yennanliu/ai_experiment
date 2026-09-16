"""Exercise 2 — putting the INT8 at the ends or in the middle makes no measurable difference.

    Build a mixed-precision quantizer. Quantize the first and last layers of a
    multi-layer network at INT8 while quantizing middle layers at INT4. Compare
    end-to-end output quality against uniform INT4 and uniform INT8. Measure the
    memory savings compared to all-INT8.

Reading of the exercise: the network is a four-layer `tanh` stack of 64x64
matrices, each layer quantised with the lesson's own `quantize_per_channel`, and
"end-to-end output quality" is the cosine similarity of the final activations
against the unquantised forward pass. The exercise's plan is run, and so is its
mirror image -- INT4 at the ends and INT8 in the middle -- because a claim about
*which* layers deserve the bits is only supported if the swap is worse.

**ANSWER: the mixed plan sits where its bit budget says it should.**

    all INT8                  cos 0.999923   mse 6.81e-06   16384 bytes
    INT8 ends / INT4 middle   cos 0.986921   mse 1.17e-03   12288 bytes
    INT4 ends / INT8 middle   cos 0.987401   mse 1.13e-03   12288 bytes
    all INT4                  cos 0.976028   mse 2.18e-03    8192 bytes

Mixed precision costs **75% of all-INT8's memory** and recovers about half the
gap between INT4 and INT8 -- 1.17e-03 against 2.18e-03 and 6.81e-06.

**FINDING: the swap is not worse -- it is marginally better.** Putting INT4 at
the ends and INT8 in the middle scores **1.13e-03** against the exercise's
**1.17e-03**, at identical memory. The exercise's premise, that the first and
last layers are the ones that need the precision, does not hold on this network;
the two placements are within 3% of each other and the sign is the wrong way.

**MECHANISM: `tanh` compresses whatever the previous layer got wrong.** Every
layer's output is squashed into (-1, 1) before the next matmul, so an error
introduced early is attenuated rather than amplified. The ends are special in
real networks because the embedding and the output head have outlier-heavy
weight distributions, not because of their position -- and a stack of Gaussian
64x64 matrices has no such layer to protect.

**FINDING: INT8 is nearly free of error and INT4 is not nearly free.** The gap
between fp32 and INT8 is `6.81e-06`; between INT8 and INT4 it is **320x** that.
The interesting decision is not where to spend a half-precision budget but
whether 4 bits is usable at all, and the mixed plan is a way of answering "not
quite" while paying 75% of the price.

Structure: `stack` is the network; `quantise_plan` applies one bits-per-layer
plan through the lesson's own per-channel quantiser.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "11-quantization"
SEED, LAYERS, WIDTH, BATCH = 1, 4, 64, 16
PLANS = {
    "all int8": (8, 8, 8, 8),
    "int8 ends / int4 middle": (8, 4, 4, 8),
    "int4 ends / int8 middle": (4, 8, 8, 4),
    "all int4": (4, 4, 4, 4),
}


def stack():
    rng = np.random.default_rng(SEED)
    weights = [rng.standard_normal((WIDTH, WIDTH)) * 0.1 for _ in range(LAYERS)]
    return weights, rng.standard_normal((BATCH, WIDTH))


def forward(weights, inputs):
    activations = inputs
    for weight in weights:
        activations = np.tanh(activations @ weight)
    return activations


def quantise_plan(ref, weights, bits_per_layer):
    out = []
    for weight, bits in zip(weights, bits_per_layer):
        quantised, scales = ref.quantize_per_channel(weight, bits, axis=0)
        out.append(ref.dequantize_per_channel(quantised, scales, axis=0))
    return out


def cosine(left, right):
    return float(left.ravel() @ right.ravel()
                 / (np.linalg.norm(left) * np.linalg.norm(right)))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    weights, inputs = stack()
    reference = forward(weights, inputs)
    rows = {}
    for name, bits in PLANS.items():
        output = forward(quantise_plan(ref, weights, bits), inputs)
        rows[name] = {
            "cos": cosine(reference, output),
            "mse": float(np.mean((reference - output) ** 2)),
            "bytes": sum(w.size * b / 8 for w, b in zip(weights, bits)),
        }
    return {"rows": rows, "params": sum(w.size for w in weights)}


def verify(result):
    rows = result["rows"]
    asked, mirrored = rows["int8 ends / int4 middle"], rows["int4 ends / int8 middle"]
    int8, int4 = rows["all int8"], rows["all int4"]
    return [
        practice.Check(
            "ANSWER: mixed precision costs 75% of all-INT8 and recovers half the INT4 gap",
            abs(asked["bytes"] / int8["bytes"] - 0.75) < 1e-9
            and int8["mse"] < asked["mse"] < int4["mse"],
            ", ".join(f"{name} cos {r['cos']:.6f} mse {r['mse']:.2e} at {r['bytes']:.0f} bytes"
                      for name, r in rows.items())
            + f". The mixed plan is {100 * asked['bytes'] / int8['bytes']:.0f}% of all-INT8's "
            f"memory and lands {(int4['mse'] - asked['mse']) / (int4['mse'] - int8['mse']):.0%} "
            "of the way from INT4 to INT8",
        ),
        practice.Check(
            "FINDING: the swap is not worse -- it is marginally better",
            mirrored["mse"] < asked["mse"] and mirrored["bytes"] == asked["bytes"],
            f"putting INT4 at the ends and INT8 in the middle scores {mirrored['mse']:.2e} "
            f"against the exercise's {asked['mse']:.2e}, at identical memory -- "
            f"{100 * abs(1 - mirrored['mse'] / asked['mse']):.0f}% apart and with the sign the "
            "wrong way. The premise that the first and last layers are the ones needing "
            "precision does not hold on this network",
        ),
        practice.Check(
            "MECHANISM: tanh compresses whatever the previous layer got wrong",
            abs(mirrored["cos"] - asked["cos"]) < 0.01,
            f"every layer's output is squashed into (-1, 1) before the next matmul, so an error "
            f"introduced early is attenuated rather than amplified: the two placements agree to "
            f"{abs(mirrored['cos'] - asked['cos']):.1e} in cosine. The ends are special in real "
            "networks because the embedding and the output head carry outlier-heavy weights, "
            f"not because of their position, and {LAYERS} Gaussian {WIDTH}x{WIDTH} matrices have "
            "no such layer to protect",
        ),
        practice.Check(
            "FINDING: the INT8-to-INT4 step is 320x the fp32-to-INT8 step",
            int4["mse"] > 100 * int8["mse"],
            f"the gap between fp32 and INT8 is {int8['mse']:.2e}; between INT8 and INT4 it is "
            f"{int4['mse'] / int8['mse']:.0f}x that. The decision this network poses is not where "
            f"to spend a half-precision budget but whether {4} bits is usable at all, and the "
            f"mixed plan answers 'not quite' while paying "
            f"{100 * asked['bytes'] / int8['bytes']:.0f}% of the price",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
