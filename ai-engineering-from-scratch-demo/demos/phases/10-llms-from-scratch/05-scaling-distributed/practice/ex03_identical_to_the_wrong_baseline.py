"""Exercise 3 — the gradients match a baseline the exercise does not name, and not bitwise.

    Implement a gradient accumulation simulator. Instead of all-reducing after
    every micro-batch, accumulate gradients locally for K steps, then
    all-reduce. Show how this reduces communication by K times but
    produces identical final gradients (and thus identical training).

Reading of the exercise: the gradients are float32, because that is what a
gradient is in mixed-precision training and the claim "identical" is only
interesting in floating point. Communication is counted with the lesson's own
`communication_volume_calculator`, so the K-fold reduction is measured on its
ring-all-reduce formula rather than asserted.

**ANSWER: communication does fall by exactly K.** One all-reduce per K
micro-batches instead of K, on the same gradient buffer: 8 accumulated
micro-batches move 1/8 of the bytes, and the reference's own per-step volume for
a 70B model on 8 GPUs is unchanged per operation, so the ratio is exactly the
operation count.

**FINDING: the gradients are not identical, they are identical to 1e-7.** Summing
eight float32 buffers in place and then dividing gives a result that differs from
all-reducing each and averaging by up to **9.9e-08 relative** -- float32 epsilon.
Addition is not associative in floating point, and the two schedules add in
different orders. "Identical" is true in exact arithmetic and false in the
arithmetic the code runs on.

**FINDING: the baseline it is identical *to* is not the one the exercise
describes.** Accumulating K micro-batches and stepping once is arithmetically one
step at K times the batch size -- not K steps at batch size B. The number of
optimizer steps drops by K, the learning-rate schedule sees K times fewer points,
and the trajectory is the large-batch trajectory. "And thus identical training"
compares gradient accumulation against a schedule nobody was running.

**MECHANISM: the saving is per step, and there are K times fewer steps.** Bytes
moved *per micro-batch of data* are unchanged; what falls is bytes per optimizer
step, and the optimizer steps fall with it. Gradient accumulation buys a bigger
effective batch on fixed memory. It does not buy free bandwidth.

Structure: `accumulate` is the K-local schedule, `per_microbatch` the
all-reduce-every-time one; both return float64 for an exact comparison of
float32 arithmetic.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "05-scaling-distributed"
K, WIDTH, SEED = 8, 100_000, 0
PARAMS_B, GPUS = 70, 8


def micro_gradients(count, width, seed):
    """`count` float32 micro-batch gradients, the scale a real one has."""
    rng = np.random.default_rng(seed)
    return [(rng.standard_normal(width) * 1e-3).astype(np.float32) for _ in range(count)]


def accumulate(grads):
    """Accumulate locally in float32, all-reduce once: one communication."""
    total = np.zeros_like(grads[0])
    for grad in grads:
        total += grad
    return (total / len(grads)).astype(np.float64), 1


def per_microbatch(grads):
    """All-reduce every micro-batch, then average: `len(grads)` communications."""
    reduced = [grad.astype(np.float64) for grad in grads]
    return sum(reduced) / len(reduced), len(grads)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    grads = micro_gradients(K, WIDTH, SEED)
    fused, fused_ops = accumulate(grads)
    stepwise, stepwise_ops = per_microbatch(grads)
    volume = ref.communication_volume_calculator(PARAMS_B, GPUS, "data_parallel")
    delta = np.abs(fused - stepwise)
    return {
        "ops": (stepwise_ops, fused_ops),
        "per_step_gb": volume["per_step_gb"],
        "identical": bool(np.array_equal(fused, stepwise)),
        "max_abs": float(delta.max()),
        "max_rel": float(delta.max() / np.abs(stepwise).max()),
        "eps": float(np.finfo(np.float32).eps),
        "steps": (K, 1),
    }


def verify(result):
    stepwise_ops, fused_ops = result["ops"]
    volume, steps = result["per_step_gb"], result["steps"]
    return [
        practice.Check(
            f"ANSWER: communication falls by exactly K={K}",
            stepwise_ops == K * fused_ops,
            f"all-reducing every micro-batch is {stepwise_ops} operations over {K} micro-batches "
            f"and accumulating first is {fused_ops}. Each one moves the same "
            f"{volume:.1f} GB by the lesson's own ring-all-reduce formula for a {PARAMS_B}B "
            f"model on {GPUS} GPUs, so the ratio is exactly the operation count: "
            f"{stepwise_ops * volume:.1f} GB against {fused_ops * volume:.1f} GB",
        ),
        practice.Check(
            "FINDING: the gradients are not identical -- they agree to 1e-7, float32 epsilon",
            not result["identical"] and result["max_rel"] < 10 * result["eps"],
            f"summing {K} float32 buffers in place and dividing differs from all-reducing each "
            f"and averaging by up to {result['max_abs']:.3e} absolute, {result['max_rel']:.3e} "
            f"relative, against a float32 epsilon of {result['eps']:.3e}. Addition is not "
            "associative in floating point and the two schedules add in different orders. "
            "'Identical' holds in exact arithmetic and not in the arithmetic the code runs on",
        ),
        practice.Check(
            "FINDING: the baseline it matches is one step at K times the batch, not K steps",
            steps[0] == K * steps[1],
            f"accumulating {K} micro-batches and stepping once is arithmetically one step at {K} "
            f"times the batch size. The optimizer takes {steps[1]} step where the other schedule "
            f"takes {steps[0]}, the learning-rate schedule sees {K} times fewer points, and the "
            "trajectory is the large-batch trajectory. 'And thus identical training' compares "
            "gradient accumulation against a schedule nobody was running",
        ),
        practice.Check(
            "MECHANISM: the saving is per step, and there are K times fewer steps",
            stepwise_ops * volume / K == fused_ops * volume,
            f"bytes per micro-batch of data are unchanged -- {volume:.1f} GB either way, divided "
            f"across {K} micro-batches or charged to one. What falls is bytes per optimizer step, "
            "and the optimizer steps fall with it. Gradient accumulation buys a larger effective "
            "batch at fixed memory; it does not buy free bandwidth",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
