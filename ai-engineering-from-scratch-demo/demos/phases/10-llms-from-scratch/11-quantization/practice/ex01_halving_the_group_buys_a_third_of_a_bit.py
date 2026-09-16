"""Exercise 1 — each halving of the group buys about 0.7 dB and doubles the scale storage.

    Implement group quantization. Instead of one scale per channel, use one scale
    per group of 128 weights within a channel. This is what GPTQ and AWQ actually
    use. Compare group sizes of 32, 64, 128, and 256 on the same weight matrix.
    Smaller groups give better quality but more storage overhead for scale
    factors.

Reading of the exercise: the weight matrix is the one the lesson's own
`full_quantization_comparison` builds -- `np.random.seed(42)`, 256x512, scaled
by 0.02 -- and error is the lesson's own `quantization_error`. The group-512 row
is added because at 512 the group *is* the channel, which makes the lesson's own
per-channel quantiser the bottom row of the same table rather than a different
method.

**ANSWER: monotone in both directions, and the exchange rate is steady.**

    group   32   20.26 dB   4096 scales   12.5% overhead
    group   64   19.36 dB   2048 scales    6.2%
    group  128   18.63 dB   1024 scales    3.1%
    group  256   18.00 dB    512 scales    1.6%
    group  512   17.45 dB    256 scales    0.8%   <- per-channel

Each halving of the group buys **0.55 to 0.90 dB** and doubles the number of
scales. There is no knee: the curve is close to linear in `log2(group)`, so
"smaller groups give better quality but more storage" has no setting at which
the trade turns.

**FINDING: the overhead is computed against the wrong denominator to sound
small.** A 16-bit scale per 32 4-bit weights is **12.5%** of the payload -- the
same as moving from 4 bits to 4.5. Quoted against an fp16 baseline instead it
would be 3%, which is how the number is usually reported.

**FINDING: the lesson's per-channel quantiser is group quantisation at group =
row length.** `quantize_per_channel(axis=0)` takes one scale from each row's
absolute maximum, which is exactly the group-512 row above. The exercise
presents group quantisation as a different technique; it is the same technique
at a different setting, and the lesson already ships one end of the sweep.

**FINDING: what the groups are buying is protection from the row maximum.** A
single outlier sets the scale for everything sharing it, so a group of 32
confines the damage to 31 neighbours instead of 511. On this Gaussian matrix the
largest per-row magnitude is only 1.74x the smallest, which is why the gain is a
steady fraction of a bit rather than the order of magnitude group quantisation
buys on real weights.

Structure: `group_quantise` is the quantiser at one group size; `sweep` runs it
across the five sizes and reports the lesson's own error metrics.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "11-quantization"
SEED, BITS = 42, 4
SHAPE, SCALE = (256, 512), 0.02
GROUPS = (32, 64, 128, 256, 512)
SCALE_BITS = 16


def weights():
    """The lesson's own `full_quantization_comparison` matrix."""
    np.random.seed(SEED)
    return np.random.randn(*SHAPE) * SCALE


def group_quantise(tensor, bits, group):
    """One symmetric scale per `group` weights within each row."""
    qmax = 2 ** (bits - 1) - 1
    out = np.zeros_like(tensor, dtype=float)
    scales = 0
    for row in range(tensor.shape[0]):
        for start in range(0, tensor.shape[1], group):
            block = tensor[row, start:start + group]
            peak = np.abs(block).max()
            step = peak / qmax if peak > 0 else 1.0
            out[row, start:start + group] = np.clip(
                np.round(block / step), -qmax - 1, qmax) * step
            scales += 1
    return out, scales


def sweep(ref, tensor):
    rows = {}
    for group in GROUPS:
        reconstructed, count = group_quantise(tensor, BITS, group)
        error = ref.quantization_error(tensor, reconstructed)
        rows[group] = {"snr": error["snr_db"], "mse": error["mse"], "scales": count,
                       "overhead": count * SCALE_BITS / (tensor.size * BITS)}
    return rows


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    tensor = weights()
    rows = sweep(ref, tensor)
    quantised, scales = ref.quantize_per_channel(tensor, BITS, axis=0)
    per_channel = ref.dequantize_per_channel(quantised, scales, axis=0)
    gains = [rows[a]["snr"] - rows[b]["snr"] for a, b in zip(GROUPS, GROUPS[1:])]
    return {
        "rows": rows,
        "gains": gains,
        "per_channel_snr": ref.quantization_error(tensor, per_channel)["snr_db"],
        "row_peaks": (float(np.abs(tensor).max(axis=1).min()),
                      float(np.abs(tensor).max(axis=1).max())),
        "fp16_overhead": rows[32]["scales"] * SCALE_BITS / (tensor.size * 16),
    }


def verify(result):
    rows, gains = result["rows"], result["gains"]
    low, high = result["row_peaks"]
    return [
        practice.Check(
            "ANSWER: each halving of the group buys 0.55-0.90 dB and doubles the scales",
            all(rows[a]["snr"] > rows[b]["snr"] for a, b in zip(GROUPS, GROUPS[1:]))
            and all(0.4 < gain < 1.0 for gain in gains),
            ", ".join(f"group {g} {r['snr']:.2f} dB at {r['scales']} scales "
                      f"({100 * r['overhead']:.1f}% overhead)" for g, r in rows.items())
            + ". The per-halving gains are "
            + ", ".join(f"{gain:.2f} dB" for gain in gains)
            + " -- close to linear in log2(group), so there is no setting at which the trade "
            "turns and nothing in the sweep identifies 128 as the right answer",
        ),
        practice.Check(
            "FINDING: 12.5% overhead is against the 4-bit payload, not against fp16",
            rows[32]["overhead"] > 0.1 and result["fp16_overhead"] < 0.04,
            f"a {SCALE_BITS}-bit scale per 32 {BITS}-bit weights is "
            f"{100 * rows[32]['overhead']:.1f}% of the payload -- the same cost as moving from "
            f"{BITS} to {BITS * (1 + rows[32]['overhead']):.1f} bits. Quoted against an fp16 "
            f"baseline it would be {100 * result['fp16_overhead']:.1f}%, which is how the number "
            "is usually reported and why group quantisation sounds free",
        ),
        practice.Check(
            "FINDING: the lesson's per-channel quantiser is the bottom row of this table",
            abs(result["per_channel_snr"] - rows[512]["snr"]) < 0.01,
            f"quantize_per_channel(axis=0) scores {result['per_channel_snr']:.2f} dB and "
            f"group-512 scores {rows[512]['snr']:.2f} dB, because a group of 512 on a 512-wide "
            "row is the row. The exercise presents group quantisation as what GPTQ and AWQ "
            "actually use, distinct from what the lesson ships; it is the same quantiser at a "
            "different group size, and the lesson already ships one end of the sweep",
        ),
        practice.Check(
            "FINDING: the groups buy protection from the row maximum, and this matrix has none",
            high / low < 2.0,
            f"a single outlier sets the scale for everything sharing it, so a group of 32 "
            f"confines the damage to 31 neighbours instead of 511. On this Gaussian matrix the "
            f"largest per-row magnitude is {high:.4f} and the smallest {low:.4f}, a ratio of "
            f"{high / low:.2f} -- which is why the gain is a steady fraction of a bit rather "
            "than the order of magnitude group quantisation buys on real weights",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
