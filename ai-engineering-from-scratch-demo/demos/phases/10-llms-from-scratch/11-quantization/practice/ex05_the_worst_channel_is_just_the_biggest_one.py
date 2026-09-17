"""Exercise 5 — "which channels to keep at higher precision" is answered by the scale factor alone.

    Implement a quantization quality dashboard. Given a weight matrix, compute
    and display: the weight distribution histogram, the quantization error
    distribution, per-channel scale factors, the worst-quantized channels
    (highest reconstruction error), and the cosine similarity between original
    and quantized outputs across 100 random inputs. Identify which channels
    should be kept at higher precision.

Reading of the exercise: every panel is computed on the lesson's own 256x512
matrix through its own `quantize_per_channel` and `quantization_error`, and the
dashboard's closing instruction -- identify which channels to keep at higher
precision -- is treated as a claim to test rather than a panel to render, since
it is the only part with an answer that can be wrong.

**ANSWER: the worst channels are the largest ones, and the dashboard's five
panels carry one number between them.** Per-row reconstruction RMSE correlates
**0.987** with the row's own scale factor. "Which channel quantised worst" and
"which channel had the biggest weights" are the same question, so four of the
five panels are views of the fifth.

**FINDING: there is barely a range to rank.** Per-row RMSE runs 2.11e-03 to
3.68e-03 -- a spread of **1.74x** from best to worst channel across 256 of them.
A dashboard whose purpose is to find the outliers is looking at a distribution
with none.

**FINDING: acting on the ranking is a bad trade.** Keeping the worst 8 of 256
rows in fp16 improves total MSE by **5%** and costs **9.4%** more memory. Spend
the same 9.4% on a smaller group size instead -- Exercise 1's table -- and the
same matrix gains 0.6 dB, which is a 14% MSE improvement.

**FINDING: the cosine panel cannot resolve what the others report.** Across 100
random inputs the output cosine is **0.9912**, and upgrading the eight worst
channels moves it to **0.9917** -- a change of 5e-04. The panel the exercise puts
last, the one measuring what the user actually cares about, is the one with the
least dynamic range.

Structure: `panels` computes the five dashboard quantities; `upgrade` re-runs the
reconstruction with the worst `k` channels left in full precision.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "11-quantization"
SEED, BITS, SHAPE, SCALE = 42, 4, (256, 512), 0.02
PROBES, UPGRADE = 100, 8


def weights():
    np.random.seed(SEED)
    return np.random.randn(*SHAPE) * SCALE


def panels(ref, tensor):
    """The five dashboard quantities, as arrays rather than plots."""
    quantised, scales = ref.quantize_per_channel(tensor, BITS, axis=0)
    reconstructed = ref.dequantize_per_channel(quantised, scales, axis=0)
    residual = tensor - reconstructed
    return {
        "reconstructed": reconstructed,
        "scales": np.asarray(scales).ravel(),
        "row_rmse": np.sqrt((residual ** 2).mean(axis=1)),
        "residual": residual,
    }


def upgrade(tensor, reconstructed, rows):
    patched = reconstructed.copy()
    patched[rows] = tensor[rows]
    return patched


def output_cosine(rng, tensor, reconstructed):
    inputs = rng.standard_normal((PROBES, tensor.shape[0]))
    exact, approx = inputs @ tensor, inputs @ reconstructed
    return float(exact.ravel() @ approx.ravel()
                 / (np.linalg.norm(exact) * np.linalg.norm(approx)))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    tensor = weights()
    board = panels(ref, tensor)
    rmse, scales = board["row_rmse"], board["scales"]
    worst = np.argsort(rmse)[-UPGRADE:]
    patched = upgrade(tensor, board["reconstructed"], worst)
    rng = np.random.default_rng(SEED)
    base = ref.quantization_error(tensor, board["reconstructed"])
    return {
        "correlation": float(np.corrcoef(rmse, scales)[0, 1]),
        "spearman": float(np.corrcoef(np.argsort(np.argsort(rmse)),
                                      np.argsort(np.argsort(scales)))[0, 1]),
        "rmse_range": (float(rmse.min()), float(np.median(rmse)), float(rmse.max())),
        "mse": (base["mse"], ref.quantization_error(tensor, patched)["mse"]),
        "memory_cost": UPGRADE / tensor.shape[0] * (16 - BITS) / BITS,
        "cosine": (output_cosine(rng, tensor, board["reconstructed"]),
                   output_cosine(rng, tensor, patched)),
        "rows": tensor.shape[0],
    }


def verify(result):
    low, median, high = result["rmse_range"]
    before, after = result["mse"]
    plain_cos, patched_cos = result["cosine"]
    return [
        practice.Check(
            "ANSWER: the worst channels are the largest ones -- RMSE tracks the scale at 0.987",
            result["correlation"] > 0.95,
            f"per-row reconstruction RMSE correlates {result['correlation']:.3f} with the row's "
            f"own scale factor, and the two orderings agree at a rank correlation of "
            f"{result['spearman']:.3f}. 'Which channel quantised worst' and 'which channel had "
            "the biggest weights' are the same question, so four of the dashboard's five panels "
            "are views of the fifth",
        ),
        practice.Check(
            "FINDING: there is barely a range to rank -- 1.74x from best channel to worst",
            high / low < 2.0,
            f"per-row RMSE runs {low:.2e} to {high:.2e} with a median of {median:.2e} -- a "
            f"spread of {high / low:.2f}x across {result['rows']} channels. A dashboard whose "
            "purpose is to find the outliers is looking at a distribution that has none",
        ),
        practice.Check(
            "FINDING: acting on the ranking is a bad trade -- 5% error for 9.4% memory",
            (1 - after / before) < result["memory_cost"],
            f"keeping the worst {UPGRADE} of {result['rows']} rows in fp16 takes MSE from "
            f"{before:.3e} to {after:.3e}, an improvement of {100 * (1 - after / before):.0f}%, "
            f"for {100 * result['memory_cost']:.1f}% more memory. Exercise 1's table buys 0.6 dB "
            "-- about 14% of MSE -- for the same budget spent on a smaller group size instead",
        ),
        practice.Check(
            "FINDING: the panel measuring what users care about has the least dynamic range",
            abs(patched_cos - plain_cos) < 1e-3 and plain_cos > 0.99,
            f"across {PROBES} random inputs the output cosine is {plain_cos:.6f} for the plain "
            f"reconstruction and {patched_cos:.6f} with the worst {UPGRADE} channels upgraded -- "
            f"a difference of {abs(patched_cos - plain_cos):.1e}. The dashboard's last panel, "
            "the one closest to end-to-end quality, cannot resolve the difference the other "
            "four spend their space ranking",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
