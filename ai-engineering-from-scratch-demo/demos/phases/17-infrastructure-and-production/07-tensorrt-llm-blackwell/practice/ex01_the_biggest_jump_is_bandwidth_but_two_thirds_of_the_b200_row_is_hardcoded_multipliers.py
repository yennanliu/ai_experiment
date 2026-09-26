"""Exercise 1 — the biggest jump is bandwidth, but two thirds of the B200 row is hardcoded multipliers.

    Run `code/main.py`. On a 120B MoE with 30% active parameters, compute the
    memory-bandwidth-limited decode throughput on H100 BF16, H100 FP8, and B200
    NVFP4/FP8. Where does the biggest jump come from?

Reading of the exercise: 30% of 120B is 36B active, which is exactly the
`print_stack(120, 36)` row the module prints. Throughput is the reference
`decode_throughput`; "where the jump comes from" is answered by switching the
B200 row's factors off one at a time with `dataclasses.replace`, so each factor
is measured rather than read off the KEY FINDING text.

**ANSWER: 47 -> 93 -> 1280 tok/s; the biggest jump is H100 FP8 -> B200, 13.76x,
and within it HBM bandwidth is the largest single factor.** BF16 -> FP8 on the
H100 is 2.0x. The B200 step decomposes into bandwidth 8.0/3.35 = 2.39x, NVFP4
weights 2.0x, MTP 1.8x and disaggregation 1.6x.

**FINDING: the memory-bandwidth-limited B200 number is 444 tok/s, not 1280.**
Bandwidth over bytes-per-token gives 444; the other 2.88x is `mtp_factor` and
`disagg_factor`, two constants multiplied in. 836 of the row's 1280 tok/s is
not bandwidth-limited decode at all.

**FINDING: KV precision and context length never reach the throughput.**
`decode_throughput` reads only weight bytes: the B200 row gives 1280 tok/s with
FP16, FP8 or FP4 KV alike, and has no `seq_len` argument. The lesson's "FP8 for
KV" decision is invisible to the number it prints.

**FINDING: "closer to 7x after overhead" is the GPU price, and the product is not
14x.** The code has no overhead term: 13.76x tok/s divided by the 4.80/2.50 =
1.92x hourly price is the 7.16x $/M gap. The printed factors (2.4, 2.0, 1.8,
2.0) multiply to 17.28, not ~14; 14 needs the row's 1.6 disaggregation.

**FINDING: GPT-OSS-120B is not 30% active.** Its model card gives 117B total and
5.1B active (4.4%), with MXFP4 MoE weights. At 5.1B active every row is 7.06x
faster than the 36B the module labels "GPT-OSS-120B MoE (30% active)".

Structure: `row()` is the reference function on a stack variant; `solve()`
builds the variants.
"""

from __future__ import annotations

import dataclasses

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "07-tensorrt-llm-blackwell"
ACTIVE, OSS_ACTIVE = 0.3 * 120, 5.1  # the exercise's 30%; the gpt-oss-120b model card


def row(ref, stack, active=ACTIVE, **change):
    return ref.decode_throughput(active, dataclasses.replace(stack, **change))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    h16, h8, b = (
        ref.STACKS[0],
        ref.STACKS[1],
        ref.STACKS[3],
    )  # H100 BF16, H100 FP8, B200
    tps = {k: round(row(ref, v), 2) for k, v in (("h16", h16), ("h8", h8), ("b", b))}
    no = {"mtp_factor": 1.0, "disagg_factor": 1.0}
    raw = row(ref, b, **no)
    return {
        "tps": tps,
        "raw": round(raw, 1),
        "jump": round(row(ref, b) / row(ref, h8), 2),
        "factors": {
            "bandwidth": round(row(ref, b, weight_bits=8, **no) / tps["h8"], 2),
            "nvfp4": round(raw / row(ref, b, weight_bits=8, **no), 2),
            "mtp": b.mtp_factor,
            "disagg": b.disagg_factor,
        },
        "kv": sorted({row(ref, b, kv_bits=k) for k in (16, 8, 4)}),
        "cost_gap": round(
            ref.cost_per_million_tokens(ACTIVE, h8)
            / ref.cost_per_million_tokens(ACTIVE, b),
            2,
        ),
        "price_ratio": round(b.price_per_gpu_hour / h8.price_per_gpu_hour, 2),
        "printed_product": round(2.4 * 2.0 * 1.8 * 2.0, 2),
        "oss_speedup": round(row(ref, b, OSS_ACTIVE) / tps["b"], 2),
    }


def verify(result):
    t, f = result["tps"], result["factors"]
    jump = result["jump"]
    return [
        practice.Check(
            "ANSWER: the biggest jump is H100 FP8 -> B200, and bandwidth is its largest factor",
            [round(v) for v in t.values()] == [47, 93, 1280]
            and jump == 13.76
            and max(f, key=f.get) == "bandwidth",
            f"{t} tok/s; BF16 -> FP8 {t['h8'] / t['h16']:.1f}x, FP8 -> B200 {jump}x "
            f"= factors {f}",
        ),
        practice.Check(
            "FINDING: the memory-bandwidth-limited B200 number is 444 tok/s, not 1280",
            result["raw"] == 444.4,
            f"bandwidth / weight bytes gives {result['raw']} tok/s; "
            f"{t['b'] - result['raw']:.0f} tok/s of the row come from MTP x disagg constants",
        ),
        practice.Check(
            "FINDING: KV precision and context length never reach the throughput",
            result["kv"] == [t["b"]],
            f"KV at 16, 8 and 4 bits all give {result['kv']} tok/s on the B200 row",
        ),
        practice.Check(
            "FINDING: 'closer to 7x after overhead' is the GPU price, and the product is not 14x",
            abs(jump / result["price_ratio"] - result["cost_gap"]) < 0.01
            and result["cost_gap"] == 7.16
            and result["printed_product"] == 17.28,
            f"{jump}x tok/s / {result['price_ratio']}x price = {result['cost_gap']}x $/M; "
            f"the printed factors multiply to {result['printed_product']}",
        ),
        practice.Check(
            "FINDING: GPT-OSS-120B is 5.1B active, not 30%",
            result["oss_speedup"] == 7.06,
            f"at 5.1B active the decode rate is {result['oss_speedup']}x the 36B row",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
