"""Exercise 4 — doubling the codebook buys one bit and 0.09% of a percent.

    A 512x512 Show-o image is 1024 tokens. At vocab K=16384, the model emits
    1024 * log2(16384) = 14,336 bits (~1.75 KiB) of data. Stable Diffusion
    outputs 512*512*24 bits = 6,291,456 bits (~768 KiB) of raw pixels. What is
    the compression ratio and what quality does it buy?

Reading of the exercise: the ratio is arithmetic and the second half is the
question, so "what quality does it buy" is answered against the *other* codebook
this phase has priced -- Chameleon's K=8192 at the same token count, in Lesson
12.11 -- and against the rate-distortion scaling for a vector quantizer, which
is where the answer turns out to be almost nothing.

**ANSWER: 438.9x, 1.75 KiB against 768 KiB.** The two arms differ by a factor of
439 and the exercise's own byte figures are exact.

**FINDING: Show-o's larger codebook compresses 7.1% *less* than Chameleon's.**
Same 1,024 tokens, 14 bits each against 13 -- **14,336** bits against **13,312**
and **438.9x** against **472.6x**. Doubling K costs one bit per token, by
construction, and buys whatever one bit of per-patch capacity is worth.

**FINDING: one bit of per-patch capacity is worth 0.09% of the reconstruction
error.** A vector quantizer's distortion scales as K^(-2/d) in MSE for a
d-dimensional source, and a 16x16x3 patch has **d = 768**. Doubling K multiplies
the RMS error by 2^(-1/768) = **0.99910**. To halve the error you would need
**2^768** times the codebook.

**ANSWER: so what the compression buys is not fidelity.** At 768 dimensions the
codebook size is nearly irrelevant to reconstruction, and 438.9x is a rate at
which no photographic detail survives in either case. What the token budget buys
is a *sequence a language model can predict* -- 1,024 positions over a
14-bit alphabet, against 786,432 bytes that no autoregressive model can emit.
The ratio is the price of admission, not a quality setting.

Structure: `bits` is the two rates, `ratio` their quotient, `rms_gain` applies
the K^(-1/d) scaling, and `CODEBOOKS` compares this lesson's vocabulary with
Lesson 12.11's.
"""

from __future__ import annotations

import math

from harness import practice

TOKENS, SIDE, CHANNELS, DEPTH = 1024, 512, 3, 8
CODEBOOKS = {"Show-o": 16384, "Chameleon (12.11)": 8192}
PATCH_SIDE = 16
DIMENSION = PATCH_SIDE * PATCH_SIDE * CHANNELS


def bits(codebook, tokens=TOKENS):
    return int(tokens * math.log2(codebook))


def raw_bits(side=SIDE):
    return side * side * CHANNELS * DEPTH


def ratio(codebook):
    return round(raw_bits() / bits(codebook), 1)


def rms_gain(factor, dimension=DIMENSION):
    """RMS distortion multiplier when the codebook grows by `factor`."""
    return factor ** (-1 / dimension)


def solve():
    coded = {name: bits(size) for name, size in CODEBOOKS.items()}
    ratios = {name: ratio(size) for name, size in CODEBOOKS.items()}
    gain = rms_gain(2)
    return {
        "raw_bits": raw_bits(), "coded": coded, "ratios": ratios,
        "raw_kib": round(raw_bits() / 8 / 1024, 1),
        "coded_kib": round(coded["Show-o"] / 8 / 1024, 2),
        "bits_per_token": {name: int(math.log2(size)) for name, size in CODEBOOKS.items()},
        "extra_bits": coded["Show-o"] - coded["Chameleon (12.11)"],
        "compression_loss_pct": round(
            (1 - ratios["Show-o"] / ratios["Chameleon (12.11)"]) * 100, 1),
        "dimension": DIMENSION,
        "rms_multiplier": round(gain, 5),
        "rms_gain_pct": round((1 - gain) * 100, 4),
        "halving_factor": 2 ** DIMENSION,
        "positions": TOKENS,
        "raw_bytes": raw_bits() // 8,
    }


def verify(result):
    coded, ratios = result["coded"], result["ratios"]
    return [
        practice.Check(
            "ANSWER: 438.9x -- 1.75 KiB against 768 KiB",
            all([coded["Show-o"] == 14_336, result["raw_bits"] == 6_291_456,
                 ratios["Show-o"] == 438.9, result["coded_kib"] == 1.75,
                 result["raw_kib"] == 768.0]),
            f"{TOKENS:,} tokens of log2(16384) = 14 bits is {coded['Show-o']:,} bits "
            f"({result['coded_kib']} KiB) against {result['raw_bits']:,} "
            f"({result['raw_kib']} KiB) of raw pixels -- {ratios['Show-o']}x",
        ),
        practice.Check(
            "FINDING: Show-o's larger codebook compresses 7.1% less than Chameleon's",
            all([coded["Chameleon (12.11)"] == 13_312,
                 ratios["Chameleon (12.11)"] == 472.6,
                 result["extra_bits"] == 1024,
                 result["compression_loss_pct"] == 7.1,
                 result["bits_per_token"] == {"Show-o": 14, "Chameleon (12.11)": 13}]),
            f"the same {TOKENS:,} tokens at {result['bits_per_token']} bits give {coded} "
            f"bits and {ratios}. Doubling K costs exactly one bit a token -- "
            f"{result['extra_bits']:,} more in total -- and {result['compression_loss_pct']}% "
            "of the compression ratio",
        ),
        practice.Check(
            "FINDING: one bit of per-patch capacity is worth 0.09% of the error",
            all([result["dimension"] == 768, result["rms_multiplier"] == 0.9991,
                 result["rms_gain_pct"] == 0.0902]),
            f"a vector quantizer's distortion scales as K^(-2/d) in MSE, and a "
            f"{PATCH_SIDE}x{PATCH_SIDE}x{CHANNELS} patch has d = {result['dimension']}. "
            f"Doubling K multiplies the RMS error by {result['rms_multiplier']} -- a "
            f"{result['rms_gain_pct']}% reduction. Halving the error would need 2^"
            f"{result['dimension']} times the codebook",
        ),
        practice.Check(
            "ANSWER: so what the compression buys is not fidelity",
            all([result["positions"] == 1024, result["raw_bytes"] == 786_432,
                 ratios["Show-o"] > 400]),
            f"at {result['dimension']} dimensions the codebook size is nearly irrelevant to "
            f"reconstruction, and {ratios['Show-o']}x is a rate at which no photographic "
            f"detail survives either way. What the budget buys is a sequence a language model "
            f"can predict: {result['positions']:,} positions over a 14-bit alphabet, against "
            f"{result['raw_bytes']:,} bytes no autoregressive model can emit. The ratio is "
            "the price of admission, not a quality setting",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
