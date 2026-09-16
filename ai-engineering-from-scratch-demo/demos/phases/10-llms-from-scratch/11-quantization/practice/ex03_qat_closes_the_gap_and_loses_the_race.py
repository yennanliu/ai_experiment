"""Exercise 3 — QAT takes the quantisation gap from 51% to 2% and still ends at a worse loss.

    Implement the straight-through estimator (STE) for quantization-aware
    training. Insert fake quantize/dequantize operations in the forward pass of a
    simple two-layer network trained on a regression task. Compare final loss
    between a model trained normally (then PTQ to INT4) versus a model trained
    with QAT from the start.

Reading of the exercise: the STE is implemented as written -- the forward pass
uses `dequantize(quantize(W))` from the lesson's own per-channel quantiser and
the backward pass updates the fp32 master weights as though the quantiser were
the identity. Both arms share an initialisation, a dataset and a step count, so
the only difference is whether the forward pass is quantised.

**ANSWER: PTQ 0.0464, QAT 0.0574 at INT4 -- PTQ wins on the number the exercise
asks for.**

**FINDING: QAT does the job it is for, and the job is not the one being
scored.** QAT's *quantisation gap* -- the cost of going from its own fp32
weights to INT4 -- is **1.02x**. PTQ's is **1.51x**. QAT made 4-bit inference
nearly free; it just optimised to a worse place while doing it, ending at
0.0563 fp32 against PTQ's 0.0307.

**MECHANISM: the STE lies about the gradient, and the lie costs optimisation.**
The backward pass pretends `round()` has derivative 1, so every step is computed
at a point the forward pass did not evaluate. On a 4-bit grid the fp32 master
can move without the quantised forward changing at all, and the gradient it
receives belongs to a different function. QAT trades optimisation quality for
deployment fidelity, which is the right trade at scale and the wrong one on a
two-layer regression that fits in fp32 anyway.

**FINDING: the comparison the exercise names cannot show what QAT is for.**
"Compare final loss between a model trained normally (then PTQ) versus a model
trained with QAT" scores only the endpoints, so a method whose whole value is
*shrinking a gap* is measured on a quantity that includes its optimisation
handicap. The two numbers to compare are the gaps, and they are 1.51x and 1.02x.

Structure: `train` runs one arm -- `qat=True` quantises the forward pass;
`fake_quantise` is the lesson's own quantiser used as the STE's forward.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "11-quantization"
SEED, STEPS, BITS, LR = 3, 800, 4, 0.05
SAMPLES, FEATURES, HIDDEN = 256, 8, 16


def fake_quantise(ref, weight, bits=BITS):
    """The STE's forward: quantise and dequantise through the lesson's own quantiser."""
    quantised, scales = ref.quantize_per_channel(weight, bits, axis=0)
    return ref.dequantize_per_channel(quantised, scales, axis=0)


def dataset():
    rng = np.random.default_rng(SEED)
    inputs = rng.standard_normal((SAMPLES, FEATURES))
    truth = rng.standard_normal((FEATURES, 1))
    targets = inputs @ truth + 0.05 * rng.standard_normal((SAMPLES, 1))
    first = rng.standard_normal((FEATURES, HIDDEN)) * 0.3
    second = rng.standard_normal((HIDDEN, 1)) * 0.3
    return inputs, targets, first, second


def loss_of(inputs, targets, first, second):
    return float(np.mean((np.tanh(inputs @ first) @ second - targets) ** 2))


def train(ref, qat):
    """One arm. With `qat`, the forward pass is quantised and the master stays fp32."""
    inputs, targets, first, second = dataset()
    for _ in range(STEPS):
        used_first = fake_quantise(ref, first) if qat else first
        used_second = fake_quantise(ref, second) if qat else second
        hidden = np.tanh(inputs @ used_first)
        error = hidden @ used_second - targets
        grad_second = hidden.T @ error / len(inputs)
        grad_hidden = (error @ used_second.T) * (1 - hidden ** 2)
        first -= LR * (inputs.T @ grad_hidden / len(inputs))   # STE: straight through
        second -= LR * grad_second
    return (loss_of(inputs, targets, first, second),
            loss_of(inputs, targets, fake_quantise(ref, first), fake_quantise(ref, second)))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ptq_fp32, ptq_int4 = train(ref, qat=False)
    qat_fp32, qat_int4 = train(ref, qat=True)
    return {
        "ptq": (ptq_fp32, ptq_int4, ptq_int4 / ptq_fp32),
        "qat": (qat_fp32, qat_int4, qat_int4 / qat_fp32),
    }


def verify(result):
    ptq_fp32, ptq_int4, ptq_gap = result["ptq"]
    qat_fp32, qat_int4, qat_gap = result["qat"]
    return [
        practice.Check(
            "ANSWER: PTQ 0.0464, QAT 0.0574 at INT4 -- PTQ wins the comparison as stated",
            ptq_int4 < qat_int4,
            f"trained normally then quantised, the INT4 loss is {ptq_int4:.4f}; trained with QAT "
            f"from the start it is {qat_int4:.4f}. On the number the exercise asks for, the "
            f"quantisation-aware arm loses by {100 * (qat_int4 / ptq_int4 - 1):.0f}%",
        ),
        practice.Check(
            "FINDING: QAT takes the quantisation gap from 1.51x to 1.02x",
            qat_gap < 1.1 < ptq_gap,
            f"the cost of going from a model's own fp32 weights to INT4 is {ptq_gap:.2f}x under "
            f"PTQ ({ptq_fp32:.4f} -> {ptq_int4:.4f}) and {qat_gap:.2f}x under QAT "
            f"({qat_fp32:.4f} -> {qat_int4:.4f}). QAT made 4-bit inference nearly free, which is "
            "the thing it exists to do",
        ),
        practice.Check(
            "MECHANISM: the STE lies about the gradient, and the lie costs optimisation",
            qat_fp32 > ptq_fp32,
            f"the backward pass pretends round() has derivative 1, so every step is computed at "
            f"a point the forward pass did not evaluate -- on a {BITS}-bit grid the master can "
            f"move without the quantised forward changing at all. QAT's fp32 loss ends at "
            f"{qat_fp32:.4f} against PTQ's {ptq_fp32:.4f}: it optimised to a worse place while "
            "making that place quantisable",
        ),
        practice.Check(
            "FINDING: the comparison as stated scores the endpoints, not the gap",
            ptq_gap / qat_gap > 1.3,
            "'compare final loss between a model trained normally then PTQ versus one trained "
            "with QAT' measures only the endpoints, so a method whose whole value is shrinking a "
            f"gap is scored on a quantity that includes its optimisation handicap. The two "
            f"numbers that answer the question are {ptq_gap:.2f}x and {qat_gap:.2f}x, a factor "
            f"of {ptq_gap / qat_gap:.1f} in QAT's favour",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
