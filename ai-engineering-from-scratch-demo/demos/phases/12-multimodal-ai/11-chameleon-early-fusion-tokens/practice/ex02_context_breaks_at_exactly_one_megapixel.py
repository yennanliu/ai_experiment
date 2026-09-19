"""Exercise 2 — context breaks at exactly one megapixel.

    A 4K image (3840x2160) at the same VQ-VAE density produces how many image
    tokens? Can a Chameleon-style model generate a 4K image in one inference
    call? What breaks first — context, tokenizer quality, or KV cache?

Reading of the exercise: "what breaks first" is answered by solving each of the
three for the resolution at which it fails, rather than by ranking them at 4K --
because at 4K two of them have already failed and the third never does, so the
ordering is only visible as a set of thresholds. The stated hardware is a
32-layer, 4096-wide model in bf16 with a 4,096-token context.

**ANSWER: 32,400 tokens, and no.** The density is one token per 256 pixels, so
3840x2160 is 32,400 -- **7.9x** a 4,096-token context.

**ANSWER: context breaks first, at exactly 1024x1024.** 4,096 tokens is one
megapixel, to the pixel. The KV cache at a 16 GiB budget holds 32,768 tokens --
**8x** more -- which is 2896x2896. Context fails at 1.0 megapixels and the cache
at 8.4.

**FINDING: tokenizer quality never breaks, because it does not depend on
resolution.** Each token carries 13 bits for its own 16x16 patch whatever the
image around it is; doubling the side doubles the token count and leaves the
bits per pixel at **0.0508**. The third candidate in the exercise's list is the
one quantity that is scale-invariant -- it was already broken at 512x512, and it
gets no worse.

**FINDING: the 4K cache is 15.82 GiB, which is the model again.** At 32 layers x
2 x 4096 x 2 bytes, a token costs **512 KiB** of cache, so a 4K image costs more
memory in KV than a 7B model costs in bf16 weights. The image is not a large
input; it is a second model.

Structure: `tokens_for` applies the density, `side_for` inverts it into a square
resolution, and `kv_bytes` prices the cache at the stated model shape.
"""

from __future__ import annotations

import math

from harness import practice

PIXELS_PER_TOKEN = 512 * 512 // 1024
FOUR_K = (3840, 2160)
CONTEXT = 4096
LAYERS, HIDDEN, DTYPE_BYTES = 32, 4096, 2
KV_BUDGET = 16 * 2 ** 30
BITS_PER_TOKEN = 13
WEIGHTS_GIB = 7e9 * 2 / 2 ** 30


def tokens_for(width, height, density=PIXELS_PER_TOKEN):
    return width * height // density


def side_for(tokens, density=PIXELS_PER_TOKEN):
    return int(math.isqrt(tokens * density))


def kv_bytes(tokens, layers=LAYERS, hidden=HIDDEN):
    return tokens * layers * 2 * hidden * DTYPE_BYTES


def solve():
    four_k = tokens_for(*FOUR_K)
    per_token = kv_bytes(1)
    cache_tokens = KV_BUDGET // per_token
    return {
        "density": PIXELS_PER_TOKEN, "four_k_tokens": four_k,
        "over_context": round(four_k / CONTEXT, 1),
        "context_side": side_for(CONTEXT),
        "context_megapixels": round(CONTEXT * PIXELS_PER_TOKEN / 1e6, 1),
        "cache_tokens": cache_tokens, "cache_side": side_for(cache_tokens),
        "cache_megapixels": round(cache_tokens * PIXELS_PER_TOKEN / 1e6, 1),
        "headroom": cache_tokens // CONTEXT,
        "per_token_kib": per_token // 1024,
        "four_k_cache_gib": round(kv_bytes(four_k) / 2 ** 30, 2),
        "weights_gib": round(WEIGHTS_GIB, 2),
        "bits_per_pixel": round(BITS_PER_TOKEN / PIXELS_PER_TOKEN, 4),
        "rates": [round(tokens_for(side, side) * BITS_PER_TOKEN / (side * side), 4)
                  for side in (512, 1024, 2048, FOUR_K[0])],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 32,400 tokens, and no -- 7.9x a 4,096-token context",
            all([result["density"] == 256, result["four_k_tokens"] == 32_400,
                 result["over_context"] == 7.9]),
            f"one token per {result['density']} pixels makes {FOUR_K[0]}x{FOUR_K[1]} "
            f"{result['four_k_tokens']:,} tokens, {result['over_context']}x a "
            f"{CONTEXT:,}-token context. A single inference call cannot hold the output",
        ),
        practice.Check(
            "ANSWER: context breaks first, at exactly 1024x1024",
            all([result["context_side"] == 1024, result["context_megapixels"] == 1.0,
                 result["cache_side"] == 2896, result["headroom"] == 8]),
            f"{CONTEXT:,} tokens is {result['context_side']}x{result['context_side']} -- "
            f"{result['context_megapixels']} megapixels, to the pixel -- while a "
            f"{KV_BUDGET // 2 ** 30} GiB cache holds {result['cache_tokens']:,} tokens, "
            f"{result['headroom']}x more, or {result['cache_side']}x{result['cache_side']} "
            f"({result['cache_megapixels']} megapixels)",
        ),
        practice.Check(
            "FINDING: tokenizer quality never breaks, because it does not depend on resolution",
            all([result["bits_per_pixel"] == 0.0508,
                 result["rates"] == [0.0508] * 4]),
            f"each token carries {BITS_PER_TOKEN} bits for its own 16x16 patch whatever "
            f"surrounds it, so doubling the side doubles the token count and leaves the rate "
            f"at {result['bits_per_pixel']} bits per pixel -- measured at four sides it is "
            f"{result['rates']}. The third candidate in the "
            "exercise's list is the one quantity that is scale-invariant -- already broken at "
            "512x512 and no worse at 4K",
        ),
        practice.Check(
            "FINDING: the 4K cache is 15.82 GiB, which is the model again",
            all([result["per_token_kib"] == 512, result["four_k_cache_gib"] == 15.82,
                 result["four_k_cache_gib"] > result["weights_gib"]]),
            f"at {LAYERS} layers x 2 x {HIDDEN:,} x {DTYPE_BYTES} bytes a token costs "
            f"{result['per_token_kib']} KiB of cache, so a 4K image costs "
            f"{result['four_k_cache_gib']} GiB -- more than the "
            f"{result['weights_gib']} GiB a 7B model costs in bf16 weights. The image is not "
            "a large input; it is a second model",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
