"""Exercise 3 — FP8 weights cost 2x, and in-domain calibration cuts NVFP4 error 65%, not to FP8.

    You see accuracy drop 3 points on MATH after NVFP4 weight conversion. Name
    two recovery paths: one quality-first (keep FP8 weights), one cost-first
    (calibrate with in-domain data).

Reading of the exercise: both paths are named in the exercise, so each one is
priced and measured. The price of each path comes from the reference
`cost_per_million_tokens` on B200 variants. The quality side cannot be measured
on MATH without a model, so a seeded stand-in is used: a 32x256 linear layer
(1% outlier weights) is quantized and scored on held-out in-domain activations,
where 8 of 256 channels run 20x hot. The formats are FP8 E4M3 per tensor,
per-tensor FP4 E2M1, and NVFP4, which is E2M1 in 16-element blocks with E4M3
block scales. Calibration picks each block's clip ratio, from 1.0 down to 0.6,
to minimise output error on 32 calibration inputs. The metric is relative
output error, which says nothing about MATH points.

**ANSWER: quality-first is FP8 weights on the same B200, at 2x the $/M;
cost-first is in-domain calibration, which keeps NVFP4's price and cuts its
error by 65%.** FP8 weights take the B200 row from $1.04 to $2.08/M, and
120 GB of weights still fits in 192 GB. Even at that price the B200 is 3.5x
cheaper than H200 FP8 ($7.29/M), the lesson's other quality-first option.
Relative output error is:

- FP8: 0.0007.
- NVFP4 with max-abs scales: 0.0126.
- NVFP4 calibrated on generic data: 0.0115.
- NVFP4 calibrated on in-domain data: 0.0044.

**FINDING: calibration has to be in-domain, and it still does not reach FP8.**
Generic calibration inputs, drawn without the hot channels, recover only 9%.
In-domain inputs recover 65%. That supports the lesson's "mitigates but does
not eliminate": the calibrated error is still 6.3x FP8's. Block microscaling
does most of the work: per-tensor FP4 scores 0.196, 15.6x worse than NVFP4.

**FINDING: the module cannot express the lesson's compromise.** `Stack` has
fields for weight and KV bits but none for activations. That means "FP8 weights
+ FP4 activations" is priced the same as FP8 weights, and no field records
calibration, so both recovery paths look identical in `main.py`.

Structure: `near()` rounds to a format grid; `nvfp4()` quantizes a row with
per-block clip ratios chosen by `clip()`; `rel_error()` scores a quantized
layer.
"""

from __future__ import annotations

import dataclasses
import math
import random

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "07-tensorrt-llm-blackwell"
E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
E4M3 = sorted(
    {m / 8 * 2**-6 for m in range(8)}
    | {(1 + m / 8) * 2 ** (e - 7) for e in range(1, 16) for m in range(8)} - {480.0}
)
ROWS, DIM, BLOCK, HOT, ALPHAS = 32, 256, 16, 8, [1 - 0.05 * k for k in range(9)]


def near(grid, x):
    return math.copysign(min(grid, key=lambda g: abs(g - abs(x))), x)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def fp4_block(block, alpha, unit):
    scale = near(E4M3, max(map(abs, block)) * alpha / 6 / unit) * unit or unit * E4M3[1]
    return [near(E2M1, min(max(x / scale, -6), 6)) * scale for x in block]


def clip(block, start, unit, calib):
    def cost(alpha):
        diff = [w - v for w, v in zip(block, fp4_block(block, alpha, unit))]
        return sum(dot(diff, x[start : start + BLOCK]) ** 2 for x in calib)

    return min(ALPHAS, key=cost) if calib else 1.0


def nvfp4(row, unit, calib=None):
    blocks = [(s, row[s : s + BLOCK]) for s in range(0, DIM, BLOCK)]
    return [q for s, b in blocks for q in fp4_block(b, clip(b, s, unit, calib), unit)]


def rel_error(w, q, xs):
    pairs = [(dot(wr, x), dot(qr, x)) for x in xs for wr, qr in zip(w, q)]
    err = sum((y - yq) ** 2 for y, yq in pairs)
    return round(err / sum(y * y for y, _ in pairs), 4)


def layer(seed=0):
    """Weights with 1% outliers; activations with HOT channels 20x hot in-domain."""
    rng = random.Random(seed)
    w = [[rng.gauss(0, 1) * (8 if rng.random() < 0.01 else 1) for _ in range(DIM)]
         for _ in range(ROWS)]  # fmt: skip
    hot = set(rng.sample(range(DIM), HOT))

    def acts(n, hot_set):
        return [[rng.gauss(0, 1) * (20 if j in hot_set else 1) for j in range(DIM)]
                for _ in range(n)]  # fmt: skip

    return w, acts(32, hot), acts(32, set()), acts(64, hot)


def formats(w, calib_in, calib_gen):
    amax = max(abs(x) for r in w for x in r)
    unit = amax / 6 / 448  # the FP32 scale under the E4M3 block scales
    tensor = {"fp8": E4M3, "fp4_tensor": E2M1}  # one scale per tensor, amax -> grid max
    out = {k: [[near(g, x * g[-1] / amax) * amax / g[-1] for x in r] for r in w]
           for k, g in tensor.items()}  # fmt: skip
    blocked = {"nvfp4": None, "generic": calib_gen, "in_domain": calib_in}
    return out | {k: [nvfp4(r, unit, c) for r in w] for k, c in blocked.items()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    b, h200 = ref.STACKS[3], ref.STACKS[2]
    w, calib_in, calib_gen, test = layer()
    price = ref.cost_per_million_tokens
    stacks = {"nvfp4": b, "fp8": dataclasses.replace(b, weight_bits=8), "h200": h200}
    quantized = formats(w, calib_in, calib_gen)
    return {
        "err": {k: rel_error(w, q, test) for k, q in quantized.items()},
        "cost": {k: round(price(36, s), 2) for k, s in stacks.items()},
        "fp8_fits": sum(ref.hbm_footprint_gb(120, 36, 8192, stacks["fp8"])) <= b.hbm_gb,
        "fields": [f.name for f in dataclasses.fields(b)],
    }


def verify(result):
    e, c = result["err"], result["cost"]
    cut = {k: round(1 - e[k] / e["nvfp4"], 2) for k in ("generic", "in_domain")}
    x = lambda a, b: round(e[a] / e[b], 1)  # noqa: E731
    ratios = (x("in_domain", "fp8"), x("fp4_tensor", "nvfp4"))
    return [
        practice.Check(
            "ANSWER: FP8 weights cost 2x the $/M; in-domain calibration keeps the price",
            (c, result["fp8_fits"])
            == ({"nvfp4": 1.04, "fp8": 2.08, "h200": 7.29}, True)
            and e["fp8"] < e["in_domain"] < e["nvfp4"],
            f"$/M {c}; relative output error {e}",
        ),
        practice.Check(
            "FINDING: calibration has to be in-domain, and it still does not reach FP8",
            cut["generic"] < 0.15 and cut["in_domain"] >= 0.6 and ratios == (6.3, 15.6),
            f"NVFP4 error cut by calibration {cut}; calibrated in-domain it is still "
            f"{ratios[0]}x FP8's, and per-tensor FP4 is {ratios[1]}x NVFP4",
        ),
        practice.Check(
            "FINDING: the module cannot express the lesson's compromise",
            not any(f.startswith(("act", "calib")) for f in result["fields"]),
            f"Stack fields {result['fields']}: no activation precision, no calibration",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
