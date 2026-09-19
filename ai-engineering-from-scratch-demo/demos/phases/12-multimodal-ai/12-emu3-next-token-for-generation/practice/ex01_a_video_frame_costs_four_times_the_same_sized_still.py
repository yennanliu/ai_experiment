"""Exercise 1 — a video frame costs four times the same-sized still.

    Emu3 produces 4096 tokens per 512x512 image at 8x8 reduction. Compute the
    equivalent for 1024x1024 and 2048x2048. What happens to inference latency?

Reading of the exercise: the token counts come from the lesson's own `TokCost`,
and "what happens to latency" is checked against the lesson's own constant-rate
model rather than reported through it -- because autoregressive decode is not a
constant rate, and the difference between the two models is the whole answer at
these lengths.

**ANSWER: 16,384 and 65,536 tokens.** At the lesson's 30 tokens per second that
is **2:16**, **9:06** and **36:24**. Tokens are quadratic in the side, so the
latency table is quadratic too.

**FINDING: the constant rate is the optimistic bound, and it diverges.**
Attention over an autoregressive decode of N tokens touches N(N+1)/2 pairs:
**8.4M** at 4,096 and **2.1 billion** at 65,536. A 16x token count carries
**256x** the attention work, so the 36-minute figure is a floor and the real
curve bends away from it exactly where the exercise is asking.

**FINDING: one 2048x2048 image is 32 GiB of KV cache.** At 32 layers x 2 x 4096
wide in bf16 a token costs 512 KiB, so the largest row in the lesson's own table
does not fit on a single 80 GiB card alongside a 7B model's weights and
activations.

**FINDING: the video rows use a different reduction, so a video frame costs 4x
the same-sized still.** The image rows pass reduction 8 and the video rows pass
**4**, which is a 4x spatial factor before any frame arithmetic. A 4-second
512x512 clip is **131,072** tokens -- **2x** the 2048x2048 image and **32x** the
512x512 still -- and only 8 of that is the video.

Structure: `tokens_for` runs the lesson's own `TokCost`, `latency` is its
constant-rate model, `attention_pairs` is the quadratic term it omits, and
`kv_bytes` prices the cache.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "12-emu3-next-token-for-generation"
SIDES = (512, 1024, 2048)
REDUCTION, VIDEO_REDUCTION = 8, 4
RATE = 30.0
LAYERS, HIDDEN, DTYPE_BYTES = 32, 4096, 2


def tokens_for(ref, side, reduction=REDUCTION):
    return ref.TokCost(f"image {side}", side, reduction).tokens()


def clip_tokens(ref, side, seconds=4.0, fps=8, time_reduction=4):
    return ref.TokCost(f"video {side}", side, VIDEO_REDUCTION, seconds, fps,
                       time_reduction).tokens()


def latency(tokens, rate=RATE):
    seconds = tokens / rate
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def attention_pairs(tokens):
    return tokens * (tokens + 1) // 2


def kv_gib(tokens):
    return round(tokens * LAYERS * 2 * HIDDEN * DTYPE_BYTES / 2 ** 30, 1)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    counts = {side: tokens_for(ref, side) for side in SIDES}
    clip = clip_tokens(ref, 512)
    return {
        "counts": counts,
        "latency": {side: latency(count) for side, count in counts.items()},
        "token_growth": counts[SIDES[-1]] // counts[SIDES[0]],
        "pairs": {side: attention_pairs(count) for side, count in counts.items()},
        "pair_growth": attention_pairs(counts[SIDES[-1]])
        // attention_pairs(counts[SIDES[0]]),
        "kv": {side: kv_gib(count) for side, count in counts.items()},
        "clip": clip, "clip_latency": latency(clip),
        "vs_biggest_image": round(clip / counts[SIDES[-1]], 1),
        "vs_same_still": clip // counts[SIDES[0]],
        "reductions": (REDUCTION, VIDEO_REDUCTION),
        "spatial_factor": (REDUCTION // VIDEO_REDUCTION) ** 2,
    }


def verify(result):
    counts, pairs, kv = result["counts"], result["pairs"], result["kv"]
    return [
        practice.Check(
            "ANSWER: 16,384 and 65,536 tokens -- 2:16, 9:06 and 36:24 at 30 tokens/s",
            all([counts == {512: 4096, 1024: 16384, 2048: 65536},
                 result["latency"] == {512: "2:16", 1024: "9:06", 2048: "36:24"},
                 result["token_growth"] == 16]),
            f"at an {REDUCTION}x reduction the sides {list(SIDES)} give {counts} tokens and, "
            f"at the lesson's own {RATE:.0f} tokens per second, {result['latency']}. Tokens "
            f"are quadratic in the side, so a 4x side is {result['token_growth']}x the work",
        ),
        practice.Check(
            "FINDING: the constant rate is the optimistic bound, and it diverges",
            all([pairs[512] == 8_390_656, pairs[2048] == 2_147_516_416,
                 result["pair_growth"] == 255]),
            f"an autoregressive decode of N tokens touches N(N+1)/2 attention pairs: "
            f"{pairs[512]:,} at 4,096 and {pairs[2048]:,} at 65,536 -- "
            f"{result['pair_growth']}x for a {result['token_growth']}x token count. The "
            "36-minute figure is a floor, and the curve bends away from it exactly where the "
            "exercise is asking",
        ),
        practice.Check(
            "FINDING: one 2048x2048 image is 32 GiB of KV cache",
            all([kv[2048] == 32.0, kv[512] == 2.0]),
            f"at {LAYERS} layers x 2 x {HIDDEN:,} wide in bf16 a token costs 512 KiB, so the "
            f"sides {list(SIDES)} cost {kv} GiB. The largest row in the lesson's own table "
            "does not fit on a single 80 GiB card beside a 7B model's weights and activations",
        ),
        practice.Check(
            "FINDING: the video rows use a different reduction, so a frame costs 4x a still",
            all([result["reductions"] == (8, 4), result["spatial_factor"] == 4,
                 result["clip"] == 131_072, result["vs_biggest_image"] == 2.0,
                 result["vs_same_still"] == 32, result["clip_latency"] == "72:49"]),
            f"the image rows pass reduction {REDUCTION} and the video rows "
            f"{VIDEO_REDUCTION} -- a {result['spatial_factor']}x spatial factor before any "
            f"frame arithmetic. A 4-second 512x512 clip is {result['clip']:,} tokens "
            f"({result['clip_latency']}), {result['vs_biggest_image']}x the 2048x2048 image "
            f"and {result['vs_same_still']}x the 512x512 still -- and only 8 of that is the "
            "video",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
