"""Exercise 3 — "identical" is bitwise when K is a power of two and 4e-03 when it is not.

    Implement a gradient accumulation simulator. Instead of all-reducing after
    every micro-batch, accumulate gradients locally for K steps, then
    all-reduce. Show how this reduces communication by K times but
    produces identical final gradients (and thus identical training).

Reading of the exercise: both schedules run in float32 -- what a gradient is in
mixed-precision training, and the only dtype in which "identical" is an
interesting claim -- and differ only in the two things a real system differs in:
when the division by K happens, and in what order the micro-batches arrive.
float64 appears once, to compare the two float32 results. Communication is
counted with the lesson's own `communication_volume_calculator`.

**ANSWER: communication does fall by exactly K.** One all-reduce per K
micro-batches instead of K, on the same gradient buffer, and the reference's own
per-step volume for a 70B model on 8 GPUs is unchanged per operation, so the
ratio is exactly the operation count.

**FINDING: "identical" is a fact about K, not about accumulation.** At K=8 and
K=16 the two schedules agree bitwise on all 100,000 coordinates, because 1/K is
exact in binary and the two summation orders coincide. At K=6 and K=10 they
disagree on about **two thirds** of them.

**FINDING: how you summarise the disagreement changes it by 30,000x.** Where the
schedules differ, the largest *element-wise* relative gap is **3.8e-03** -- four
orders of magnitude above float32 epsilon. The pooled statistic that is usually
quoted, `max|delta| / max|g|`, reads **1.3e-07** on the very same arrays, which
is float32 epsilon and would have been reported as "identical to rounding". The
gap is largest where the accumulated gradient nearly cancels, and dividing by
the largest coordinate in the tensor hides exactly those coordinates.

**FINDING: arrival order breaks it even at K=8.** Delivering the same eight
micro-batches in reverse -- what a ring all-reduce does, since each worker forms
its partial sums in a rotated sequence -- makes **61,977** coordinates differ, at
2.9e-03 element-wise. Addition is not associative, so the bitwise result survives
only the schedule that happens to add in the same sequence.

**FINDING: the baseline it is identical *to* is not the one the exercise
describes.** Accumulating K micro-batches and stepping once is arithmetically one
step at K times the batch size, not K steps at batch size B: the optimizer takes
K times fewer steps, the learning-rate schedule sees K times fewer points, and
the trajectory is the large-batch one. What falls is bytes per optimizer step,
and the steps fall with it -- gradient accumulation buys a larger effective batch
at fixed memory, not free bandwidth.

Structure: `fused` accumulates in float32 and divides once, `stepwise` divides
each micro-batch as it arrives; `gap` reports both the element-wise and the
pooled relative difference so the two can be compared.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "05-scaling-distributed"
K, WIDTH, SEED = 8, 100_000, 0
SWEEP = (6, 8, 10, 16)
PARAMS_B, GPUS = 70, 8


def micro_gradients(count):
    """`count` float32 micro-batch gradients, the scale a real one has."""
    rng = np.random.default_rng(SEED)
    return [(rng.standard_normal(WIDTH) * 1e-3).astype(np.float32) for _ in range(count)]


def fused(grads):
    """Accumulate locally in float32, divide once, all-reduce once."""
    total = np.zeros_like(grads[0])
    for grad in grads:
        total += grad
    return total / np.float32(len(grads))


def stepwise(grads, order):
    """All-reduce every micro-batch: each arrives already divided by K."""
    total = np.zeros_like(grads[0])
    for index in order:
        total += grads[index] / np.float32(len(grads))
    return total


def gap(left, right):
    """Element-wise and pooled relative difference between two float32 results."""
    wide, other = left.astype(np.float64), right.astype(np.float64)
    delta, scale = np.abs(wide - other), np.abs(other)
    relative = np.divide(delta, scale, out=np.zeros_like(delta), where=scale > 0)
    return {"identical": bool(np.array_equal(left, right)),
            "differing": int(np.count_nonzero(left != right)),
            "max_rel": float(relative.max()),
            "pooled": float(delta.max() / scale.max()) if scale.max() else 0.0}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    sweep = {}
    for count in SWEEP:
        grads = micro_gradients(count)
        sweep[count] = gap(fused(grads), stepwise(grads, range(count)))
    grads = micro_gradients(K)
    return {
        "sweep": sweep,
        "reversed": gap(fused(grads), stepwise(grads, reversed(range(K)))),
        "ops": (K, 1),
        "per_step_gb": ref.communication_volume_calculator(PARAMS_B, GPUS, "data_parallel"
                                                           )["per_step_gb"],
        "eps": float(np.finfo(np.float32).eps),
        "width": WIDTH,
        "steps": (K, 1),
    }


def summary(sweep, eps):
    """Which K agree bitwise, which do not, and how the two error metrics compare."""
    powers = [k for k in SWEEP if sweep[k]["identical"]]
    odd = [k for k in SWEEP if not sweep[k]["identical"]]
    return (powers, odd, max(sweep[k]["max_rel"] for k in odd),
            all(sweep[k]["pooled"] < 10 * eps for k in odd))


def column(sweep, keys, field, fmt):
    """One comma-joined row of `field` across `keys` -- the detail strings' formatter."""
    return ", ".join(format(sweep[k][field], fmt) for k in keys)


def verify(result):
    sweep, flipped, eps = result["sweep"], result["reversed"], result["eps"]
    stepwise_ops, fused_ops = result["ops"]
    volume, steps, width = result["per_step_gb"], result["steps"], result["width"]
    powers, odd, worst, pooled_small = summary(sweep, eps)
    return [
        practice.Check(
            f"ANSWER: communication falls by exactly K={K}",
            stepwise_ops == K * fused_ops,
            f"all-reducing every micro-batch is {stepwise_ops} operations over {K} micro-batches "
            f"and accumulating first is {fused_ops}. Each one moves the same {volume:.1f} GB by "
            f"the lesson's own ring-all-reduce formula for a {PARAMS_B}B model on {GPUS} GPUs, so "
            f"the ratio is exactly the operation count: {stepwise_ops * volume:.1f} GB against "
            f"{fused_ops * volume:.1f} GB",
        ),
        practice.Check(
            "FINDING: identical is a fact about K being a power of two",
            powers == [8, 16] and odd == [6, 10],
            f"the two schedules agree bitwise on all {width:,} coordinates at K={powers} -- 1/K "
            f"is exact in binary there and the two summation orders coincide -- and disagree at "
            f"K={odd} on {column(sweep, odd, 'differing', ',')} of them. The exercise's 'identical "
            "final gradients' is true for the K it happens to be run at and false one micro-batch "
            "either side",
        ),
        practice.Check(
            "FINDING: element-wise 3.8e-03 against a pooled 1.3e-07 on the same arrays",
            worst > 1e-3 and pooled_small,
            f"where the schedules differ the largest element-wise relative gap is "
            f"{column(sweep, odd, 'max_rel', '.1e')}, while max|delta| / max|g| on the very same "
            f"arrays reads {column(sweep, odd, 'pooled', '.1e')} -- float32 epsilon is {eps:.1e}. "
            "The gap is largest where the accumulated gradient nearly cancels, and dividing by the "
            "largest coordinate in the tensor hides precisely those coordinates",
        ),
        practice.Check(
            "FINDING: arrival order breaks it even at the K where the dtype does not",
            sweep[K]["identical"] and not flipped["identical"],
            f"delivering the same {K} micro-batches in reverse -- what a ring all-reduce does, "
            f"since each worker forms its partial sums in a rotated sequence -- makes "
            f"{flipped['differing']:,} coordinates differ at {flipped['max_rel']:.1e} "
            f"element-wise, against {sweep[K]['differing']} in arrival order. Addition is not "
            "associative, so the bitwise result survives only the schedule that adds in the same "
            "sequence",
        ),
        practice.Check(
            "FINDING: the baseline it matches is one step at K times the batch, not K steps",
            steps[0] == K * steps[1],
            f"accumulating {K} micro-batches and stepping once is arithmetically one step at {K} "
            f"times the batch size. The optimizer takes {steps[1]} step where the other takes "
            f"{steps[0]}, the learning-rate schedule sees {K} times fewer points, and the "
            f"trajectory is the large-batch one. Bytes per micro-batch of data are unchanged at "
            f"{volume:.1f} GB either way; what falls is bytes per optimizer step, and the steps "
            "fall with it -- 'and thus identical training' compares gradient accumulation against "
            "a schedule nobody was running",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
