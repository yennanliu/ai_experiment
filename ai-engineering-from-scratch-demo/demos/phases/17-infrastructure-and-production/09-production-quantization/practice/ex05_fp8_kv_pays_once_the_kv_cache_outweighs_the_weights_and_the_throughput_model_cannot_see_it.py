"""Exercise 5 — FP8 KV pays once the KV cache outweighs the weights, and the throughput model cannot see it.

    When does it make sense to combine AWQ weights with FP8 KV cache vs keeping
    KV at BF16?

Reading of the exercise: "makes sense" is measured on the two things FP8 KV
changes in the lesson's model. The first is HBM, via the reference
`memory_breakdown` and `gpu_check`. The second is decode bytes per step:
decode is bandwidth-bound, and the lesson says so, and each step reads the
weights plus the resident KV. AWQ + FP8 KV is built from the reference AWQ
row with `kv_bits=8`, since `FORMATS` has no such row. The accuracy side, which
the code does not model, is taken from vLLM's quantized-KV docs.

**ANSWER: combine them once concurrency x context puts more bytes in the KV
cache than in the 4-bit weights; keep BF16 KV below that, or when attention
accuracy is the risk.** The break-even is 133,514 tokens in flight for 70B
(65 sequences at 2k) and 42,220 for 7B (21 sequences). Past it, FP8 KV cuts
per-step bytes by more than 25%. At the lesson's 128 x 2k, 70B goes from
103.72 to 69.36 GB read per step, 1.50x fewer, and the total from 107.22 GB
(H200) to 72.86 GB (one H100). Its concurrency ceiling on an 80 GB card
doubles, from 77 to 154 2k sequences. At a single 2k sequence on a 7B, KV is
4.6% of the bytes and FP8 KV saves 2.3%, not worth any accuracy risk. vLLM's
docs name the risks: sliding-window layers are "more sensitive to KV-cache
quantization", FP8 KV defaults to unit scales without a calibration pass,
and with FlashAttention 3 the queries are quantized to FP8 too. So for a
short, low-concurrency, accuracy-marginal workload the answer is BF16 KV.

**FINDING: the code's throughput model cannot see the KV decision at all.**
`relative_throughput` is 16 / weight_bits, so AWQ and NVFP4 + FP8 KV are both
4.00x, and FP8 KV on AWQ would score the same as BF16 KV. Its bytes-per-step
model disagrees at every point where KV is large. At 70B, 256 x 8k, KV is 94%
of AWQ's bytes and FP8 KV reads 1.89x fewer.

**FINDING: FP8 KV does not rescue the long-context case.** At 70B, 256 x 8k,
AWQ + FP8 KV is still 313.4 GB, MULTI-GPU. Past a point the lever is
concurrency or context, not precision.

Structure: `point()` measures one (params, concurrency, context) cell for
both KV precisions; `break_even()` divides the AWQ weight bytes by the BF16 KV
bytes per token.
"""

from __future__ import annotations

import dataclasses

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "09-production-quantization"
CTX, H100 = 2048, 80


def formats(ref):
    awq = next(f for f in ref.FORMATS if f.name.startswith("AWQ"))
    return awq, dataclasses.replace(awq, name="AWQ + FP8 KV", kv_bits=8)


def point(ref, params, conc, ctx):
    bf16, fp8 = (ref.memory_breakdown(params, f, conc, ctx) for f in formats(ref))
    step16, step8 = bf16["weight"] + bf16["kv"], fp8["weight"] + fp8["kv"]
    return {
        "kv_share": round(bf16["kv"] / step16, 3), "step": (round(step16, 2), round(step8, 2)),
        "saved": round(1 - step8 / step16, 3), "speedup": round(step16 / step8, 2),
        "total": (round(bf16["total"], 2), round(fp8["total"], 2)),
        "gpu": (ref.gpu_check(bf16["total"]), ref.gpu_check(fp8["total"])),
    }


def break_even(ref, params):
    """Tokens in flight at which BF16 KV bytes equal the AWQ weight bytes."""
    awq = formats(ref)[0]
    per_token = ref.memory_breakdown(params, awq, 1, CTX)["kv"] / CTX
    tokens = int(ref.memory_breakdown(params, awq, 1, CTX)["weight"] / per_token)
    return tokens, round(tokens / CTX)


def ceiling(ref, fmt, params=70):
    one = ref.memory_breakdown(params, fmt, 1, CTX)
    return int((H100 - one["weight"] - one["act"]) / one["kv"])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    awq, awq8 = formats(ref)
    nvfp4 = next(f for f in ref.FORMATS if f.name.startswith("NVFP4"))
    return {
        "lesson": point(ref, 70, 128, CTX), "long": point(ref, 70, 256, 8192),
        "single": point(ref, 7, 1, CTX),
        "even": {p: break_even(ref, p) for p in (70, 7)},
        "ceiling": (ceiling(ref, awq), ceiling(ref, awq8)),
        "tput": {f.name: ref.relative_throughput(f) for f in (awq, awq8, nvfp4)},
    }


def verify(result):
    les, long_, one, even = result["lesson"], result["long"], result["single"], result["even"]
    return [
        practice.Check(
            "ANSWER: combine them once concurrency x context puts more bytes in KV than in weights",
            all([even == {70: (133514, 65), 7: (42220, 21)}, les["step"] == (103.72, 69.36),
                 les["speedup"] == 1.5, les["gpu"] == ("H200 141GB", "H100 80GB"),
                 result["ceiling"] == (77, 154), one["kv_share"] == 0.046,
                 one["saved"] == 0.023]),
            f"break-even tokens/sequences {even}; 70B 128x2k bytes/step {les['step']} "
            f"({les['speedup']}x), totals {les['total']} on {les['gpu']}, H100 ceiling "
            f"{result['ceiling']}; 7B 1x2k KV share {one['kv_share']}, saving {one['saved']}",
        ),
        practice.Check(
            "FINDING: the code's throughput model cannot see the KV decision at all",
            set(result["tput"].values()) == {4.0} and long_["kv_share"] == 0.94
            and long_["speedup"] == 1.89,
            f"relative_throughput {result['tput']}; at 70B 256x8k KV is "
            f"{long_['kv_share']:.0%} of bytes and FP8 KV reads {long_['speedup']}x fewer",
        ),
        practice.Check(
            "FINDING: FP8 KV does not rescue the long-context case",
            long_["gpu"] == ("MULTI-GPU", "MULTI-GPU") and round(long_["total"][1], 1) == 313.4,
            f"70B 256x8k totals {long_['total']} GB, {long_['gpu']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
