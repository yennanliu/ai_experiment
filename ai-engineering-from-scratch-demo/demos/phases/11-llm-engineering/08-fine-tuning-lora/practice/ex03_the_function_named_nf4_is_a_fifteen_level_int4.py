"""Exercise 3 — the function named nf4 is a 15-level symmetric int4, and block size barely moves.

    **Quantization error analysis.** Take the trained model's weight matrices
    before and after quantize_to_nf4 / dequantize_from_nf4. Compute the mean
    squared error, max absolute error, and the correlation between original and
    reconstructed weights. Experiment with block_size values of 32, 64, 128,
    and 256.

Reading of the exercise: the matrix analysed is the first frozen weight of a
trained model -- 512x256, trained through `train_lora` with the seed fixed --
and all three statistics are computed over the whole matrix rather than per
block, because the exercise asks for one number each.

**ANSWER: the error is 7% of the weight RMS and the block size is worth 4% of
it.** Across block sizes 32, 64, 128 and 256 the MSE runs 6.056e-06, 6.351e-06,
6.506e-06 and 6.579e-06 -- an 8.6% spread for an 8x change in block size -- and
the correlation stays at 0.9975. The max absolute error is identical to five
decimal places at every block size.

**FINDING: one of the sixteen levels is unreachable.** `scales` is
`max|block| / 7.0` and the quantiser clamps to `(-8, 7)`, so the most negative
value in a block maps to -7 and -8 can never be produced. Only 15 of the 16
codes appear anywhere in the quantised tensor: a 4-bit format used at 3.91 bits.

**FINDING: it is not NF4, and the reconstruction proves it.** NF4 is a fixed
16-value codebook, so a dequantised tensor can hold at most 16 distinct numbers.
This one holds 28,125, because every block carries its own scale. The function
implements block-wise symmetric integer quantisation -- correct, useful, and not
what it is named.

**MECHANISM: the block size cannot matter much because the weights are
homogeneous.** `nn.Linear`'s default initialisation is uniform over
(-1/sqrt(fan_in), 1/sqrt(fan_in)), and LoRA freezes it, so every block's
`max|x|` is drawn from the same distribution. The per-block scale has almost
nothing to adapt to, and a larger block only loses the little it had.

**CONTROL: give it a matrix with structure and the block size earns its keep.**
Interleaving runs of 32 elements at 100x makes the boundary matter: the fraction
of the small elements reconstructed as exactly 0.0 is 0.069 at block 32 -- the
same as the real weight's baseline -- and 1.000 at block 64 and 256. A
32-element block gives them their own scale; anything larger rounds every one of
them away. The knob works; this fixture has no dynamic range for it to track.

Structure: `stats` computes the three numbers, `levels` counts the codes the
quantiser actually emits, and `skewed` is the structured control matrix.
"""

from __future__ import annotations

import torch

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "08-fine-tuning-lora"
BLOCKS = (32, 64, 128, 256)
SEED, RANK = 42, 8


def trained_weight(ref):
    torch.manual_seed(SEED)
    model = ref.create_demo_model()
    ref.inject_lora(model, ["0", "2", "4"], rank=RANK)
    ref.train_lora(model, ref.create_demo_data(500), epochs=5, lr=1e-3, batch_size=32)
    return dict(model.named_parameters())["0.linear.weight"].data.clone()


def stats(ref, weight, block_size):
    quantized, scales, shape, pad = ref.quantize_to_nf4(weight, block_size)
    restored = ref.dequantize_from_nf4(quantized, scales, shape, pad)
    error = weight - restored
    pair = torch.stack([weight.reshape(-1), restored.reshape(-1)])
    return {"mse": float(error.pow(2).mean()),
            "max": round(float(error.abs().max()), 5),
            "corr": round(float(torch.corrcoef(pair)[0, 1]), 6),
            "rel": round(float(error.pow(2).mean().sqrt() / weight.pow(2).mean().sqrt()), 4),
            "levels": sorted({int(v) for v in quantized.unique()}),
            "distinct": int(restored.unique().numel())}


def skewed(weight, period=32):
    """The control: alternating runs of 32 elements scaled 100x, so the dynamic
    range varies inside a 256-element block and not inside a 32-element one."""
    flat = weight.reshape(-1).clone()
    index = torch.arange(flat.numel())
    flat[(index // period) % 2 == 0] *= 100.0
    return flat.reshape(weight.shape)


def zeroed(ref, weight, block_size, period=32):
    """What fraction of the small runs is reconstructed as exactly 0."""
    flat = weight.reshape(-1)
    index = torch.arange(flat.numel())
    small = (index // period) % 2 == 1
    quantized, scales, shape, pad = ref.quantize_to_nf4(weight, block_size)
    restored = ref.dequantize_from_nf4(quantized, scales, shape, pad).reshape(-1)
    return round(float((restored[small] == 0).float().mean()), 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "lora")
    weight = trained_weight(ref)
    rows = {block: stats(ref, weight, block) for block in BLOCKS}
    control = skewed(weight)
    return {
        "shape": tuple(weight.shape),
        "rms": round(float(weight.pow(2).mean().sqrt()), 5),
        "mse": [f"{rows[b]['mse']:.3e}" for b in BLOCKS],
        "mse_spread": round(rows[256]["mse"] / rows[32]["mse"], 3),
        "max": [rows[b]["max"] for b in BLOCKS],
        "corr": [rows[b]["corr"] for b in BLOCKS],
        "rel": [rows[b]["rel"] for b in BLOCKS],
        "levels": rows[64]["levels"], "distinct": rows[64]["distinct"],
        "control_zeros": {b: zeroed(ref, control, b) for b in (32, 64, 256)},
        "flat_zeros": {b: zeroed(ref, weight, b) for b in (32, 256)},
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the error is 7% of the weight RMS and block size is worth 4% of it",
            all([max(result["rel"]) < 0.075, result["mse_spread"] < 1.1,
                 len(set(result["corr"])) == len(BLOCKS)]),
            f"over a {result['shape']} weight of RMS {result['rms']}, MSE by block size "
            f"{list(BLOCKS)} is {result['mse']} -- a {100 * (result['mse_spread'] - 1):.1f}% "
            f"spread for an 8x change -- with relative RMS error {result['rel']} and "
            f"correlation {result['corr']}",
        ),
        practice.Check(
            "FINDING: one of the sixteen levels is unreachable",
            all([len(result["levels"]) == 15, min(result["levels"]) == -7,
                 max(result["levels"]) == 7]),
            f"`scales` is max|block| / 7.0 and the quantiser clamps to (-8, 7), so the most "
            f"negative value in a block maps to -7 and -8 can never be produced. The codes "
            f"that appear are {result['levels']} -- {len(result['levels'])} of 16, a 4-bit "
            f"format used at {__import__('math').log2(len(result['levels'])):.2f} bits",
        ),
        practice.Check(
            "FINDING: it is not NF4, and the reconstruction proves it",
            result["distinct"] > 1000,
            f"NF4 is a fixed 16-value codebook, so a dequantised tensor could hold at most "
            f"16 distinct numbers. This one holds {result['distinct']:,}, because every "
            "block carries its own scale. The function is block-wise symmetric integer "
            "quantisation: correct, useful, and not what it is named",
        ),
        practice.Check(
            "MECHANISM: the max absolute error does not move at all",
            len(set(result["max"])) == 1,
            f"max absolute error by block size is {result['max']} -- identical to five "
            "decimal places across an 8x change. `nn.Linear`'s default initialisation is "
            "uniform over (-1/sqrt(fan_in), 1/sqrt(fan_in)) and LoRA freezes it, so every "
            "block's max|x| comes from the same distribution and there is nothing to adapt",
        ),
        practice.Check(
            "CONTROL: give it a matrix with structure and the block size earns its keep",
            all([result["control_zeros"][32] == result["flat_zeros"][32],
                 result["control_zeros"][64] == 1.0, result["control_zeros"][256] == 1.0,
                 max(result["flat_zeros"].values()) < 0.1]),
            f"interleaving runs of 32 elements at 100x makes the boundary matter: the "
            f"fraction of the small elements reconstructed as exactly 0.0 is "
            f"{result['control_zeros']} at block 32, 64 and 256. A 32-element block gives "
            f"them their own scale and anything larger rounds every one of them away. On "
            f"the real weight the same measurement is {result['flat_zeros']} at both ends. "
            "The knob works; this fixture has no dynamic range for it to track",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
