"""Exercise 1 — three of the four images get the same codes.

    Chameleon uses K=8192 codebook entries and 1024 tokens per 512x512 image.
    Estimate the compression ratio vs a 24-bit RGB image. Is it lossy? How
    lossy?

Reading of the exercise: the ratio is arithmetic, so the interesting half is
"how lossy", and that is answered by running the lesson's own quantizer on the
lesson's own four images rather than by describing quantization in general. The
patch extraction is duplicated here because `image_to_tokens` returns only the
codes, and the reconstruction error needs the patch it replaced.

**ANSWER: 472.6x, and yes.** 512x512 at 24 bits is 6,291,456 bits; 1,024 tokens
of log2(8192) = 13 bits each is 13,312. That is **0.0508 bits per pixel**
against 24 -- and roughly **20x below** a 1-bit-per-pixel JPEG, taken here as a
stated reference for "good quality". A rate that far below JPEG is not
compressing an image; it is describing one.

**FINDING: on the lesson's own tokenizer, three of four images are the same
image.** `red`, `green` and `gray` all quantize to **(34, 36, 36, 36)**; only
`blue` differs. Four visually distinct inputs produce **2** distinct code
sequences, and the downstream bigram cannot tell them apart at all.

**FINDING: the reconstruction error is comparable to the signal.** Mean absolute
error between each patch and its codebook entry is 3.19, 1.56, 1.56 and 1.69 on
a scale whose whole range is 0-9. The nearest codebook entry to `[8, 7, 7, 8]`
is `[6, 1, 4, 7]`.

**FINDING: the toy compresses 15x less than the thing it illustrates.** Its
8x8 grayscale image is 512 bits and its 4 tokens of log2(16) are 16 -- **32x**,
against Chameleon's 472.6x. The collisions above happen at a *fifteenth* of the
real compression, which is what makes them worth reporting rather than
dismissing as a toy artefact.

Structure: `bits` is the two rates, `patches` re-derives what `image_to_tokens`
quantized, `reconstruction` measures each patch against its codebook entry, and
`CODES` collects the lesson's four images.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "11-chameleon-early-fusion-tokens"
CODEBOOK_SIZE, TOKENS, SIDE, CHANNELS, DEPTH = 8192, 1024, 512, 3, 8
JPEG_BPP = 1.0
KINDS = ("red", "blue", "green", "gray")
TOY_SIDE, TOY_TOKENS, TOY_CODEBOOK = 8, 4, 16


def bits(side=SIDE, tokens=TOKENS, codebook=CODEBOOK_SIZE):
    raw = side * side * CHANNELS * DEPTH
    coded = tokens * math.log2(codebook)
    return raw, coded


def patches(image, size=4):
    """The 2x2 mean-pooled patches image_to_tokens quantizes, re-derived."""
    out = []
    for top in range(0, TOY_SIDE, size):
        for left in range(0, TOY_SIDE, size):
            out.append([sum(image[top + 2 * r + dr][left + 2 * c + dc]
                            for dr in range(2) for dc in range(2)) // 4
                        for r in range(2) for c in range(2)])
    return out


def reconstruction(ref, kind):
    """Mean absolute error between each patch and the codebook entry chosen for it."""
    image = ref.synth_image(kind)
    codes = ref.image_to_tokens(image)
    errors = [sum(abs(a - b) for a, b in zip(patch, ref.CODEBOOK[code - ref.IMG_OFFSET]))
              / len(patch) for patch, code in zip(patches(image), codes)]
    return tuple(codes), round(sum(errors) / len(errors), 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    raw, coded = bits()
    measured = {kind: reconstruction(ref, kind) for kind in KINDS}
    toy_raw = TOY_SIDE * TOY_SIDE * DEPTH          # grayscale: one channel
    toy_coded = TOY_TOKENS * math.log2(TOY_CODEBOOK)
    first = ref.synth_image("red")
    return {
        "raw_bits": raw, "coded_bits": int(coded),
        "ratio": round(raw / coded, 1),
        "bits_per_pixel": round(coded / (SIDE * SIDE), 4),
        "pixels_per_token": SIDE * SIDE // TOKENS,
        "vs_jpeg": round(JPEG_BPP / (coded / (SIDE * SIDE)), 1),
        "codes": {kind: row[0] for kind, row in measured.items()},
        "distinct": len({row[0] for row in measured.values()}),
        "errors": {kind: row[1] for kind, row in measured.items()},
        "first_patch": patches(first)[0],
        "first_code": ref.CODEBOOK[measured["red"][0][0] - ref.IMG_OFFSET],
        "toy_ratio": round(toy_raw / toy_coded, 1),
        "toy_gap": round((raw / coded) / (toy_raw / toy_coded), 1),
    }


def verify(result):
    codes, errors = result["codes"], result["errors"]
    return [
        practice.Check(
            "ANSWER: 472.6x, and yes -- 0.0508 bits per pixel against 24",
            all([result["raw_bits"] == 6_291_456, result["coded_bits"] == 13_312,
                 result["ratio"] == 472.6, result["bits_per_pixel"] == 0.0508,
                 result["pixels_per_token"] == 256, result["vs_jpeg"] == 19.7]),
            f"{SIDE}x{SIDE} at 24 bits is {result['raw_bits']:,} bits; {TOKENS:,} tokens of "
            f"log2({CODEBOOK_SIZE:,}) = 13 bits is {result['coded_bits']:,}. That is "
            f"{result['bits_per_pixel']} bits per pixel -- one token per "
            f"{result['pixels_per_token']} pixels -- and {result['vs_jpeg']}x below a "
            f"{JPEG_BPP:g}-bit-per-pixel JPEG taken as a stated reference",
        ),
        practice.Check(
            "FINDING: on the lesson's own tokenizer, three of four images are the same image",
            all([result["distinct"] == 2, len(KINDS) == 4,
                 codes["red"] == codes["green"] == codes["gray"] == (34, 36, 36, 36),
                 codes["blue"] == (38, 38, 38, 38)]),
            f"the four synthetic images quantize to {codes} -- {result['distinct']} distinct "
            f"sequences from {len(KINDS)} visually distinct inputs. red, green and gray are "
            "indistinguishable to everything downstream of the tokenizer, including the "
            "bigram trained on them",
        ),
        practice.Check(
            "FINDING: the reconstruction error is comparable to the signal",
            all([errors == {"red": 3.188, "blue": 1.562, "green": 1.562, "gray": 1.688},
                 result["first_patch"] == [8, 7, 7, 8],
                 result["first_code"] == [6, 1, 4, 7]]),
            f"mean absolute error between each patch and its chosen codebook entry is "
            f"{errors}, on a scale whose whole range is 0-9. The nearest entry to "
            f"{result['first_patch']} is {result['first_code']}",
        ),
        practice.Check(
            "FINDING: the toy compresses 15x less than the thing it illustrates",
            all([result["toy_ratio"] == 32.0, result["toy_gap"] == 14.8,
                 result["ratio"] > 400]),
            f"the toy's {TOY_SIDE}x{TOY_SIDE} grayscale image is "
            f"{TOY_SIDE * TOY_SIDE * DEPTH} bits and its {TOY_TOKENS} tokens of "
            f"log2({TOY_CODEBOOK}) are {TOY_TOKENS * 4} -- {result['toy_ratio']}x, against "
            f"Chameleon's "
            f"{result['ratio']}x, a gap of {result['toy_gap']}x. The collisions above happen "
            "at a fifteenth of the real compression",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
