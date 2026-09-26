"""Exercise 2 — the savings fund 40 B200s; the load needs 7 by the code and 551 by the lesson.

    A customer spends $2M/year on H100 + vLLM. What is the break-even number of
    Blackwell GPUs they need to buy to amortize a migration to TRT-LLM in 12
    months, given the 7x economic gap?

Reading of the exercise: at a 7x gap the same tokens cost $2M / 7 after the
migration, so 12 months of savings is $2M x (1 - 1/7). The break-even count is
how many Blackwell GPUs that sum pays for. The lesson gives no purchase price,
so a GPU is costed at the module's own B200 price, $4.80/hour for a year. That
count is then compared with how many GPUs the workload actually needs. "H100 +
vLLM" is the module's H100 FP8 row, the one whose gap to B200 is 7.16x.

**ANSWER: $1.71M of savings pays for 40 B200s; the workload needs 7.** At
$42,048 per B200-year, 12 months of savings covers 40.8 GPUs. $2M at the
module's H100 FP8 price, $7.46/M, buys 2.68e11 tokens a year, and at the B200
row's 1280 tok/s that takes 6.64 GPUs. Buying 7 costs $294k, which the savings
repay in about 2 months, so the migration breaks even with 33 GPUs to spare.

**FINDING: the lesson's own $/M sizes the same fleet 83x larger.** At the
lesson's $0.09/M, $2M buys 22.2T tokens, or 704k tok/s sustained. At the B200
row's 1280 tok/s that needs 551 GPUs, far above the 40 the savings fund. The
module's H100 FP8 price is 82.9x the lesson's $0.09 and its B200 price is 52.1x
the lesson's $0.02. It models one decode stream per GPU with no batching, so
the break-even answer is 7 or 551 GPUs depending on which of the lesson's two
sets of numbers you trust.

**FINDING: the 7x is a GB200 NVL72 number, and an NVL72 is 72 GPUs.** The
lesson's figures give $0.09 / $0.012 = 7.5x for GB200 NVL72, and $0.09 / $0.02
= 4.5x for HGX B200. The 7x only holds on a rack NVIDIA specifies as 72
Blackwell GPUs, and $1.71M spread over 72 is $23.8k per GPU. At the module's
GB200 price of $54,312 a year, the savings cover 31.6 GPUs, less than half a
rack.

Structure: `fleet()` sizes the workload from a $/M price and a per-GPU rate;
`solve()` runs it on the module's rows and on the lesson's quoted prices.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "07-tensorrt-llm-blackwell"
SPEND, GAP, ACTIVE, HOURS = 2_000_000, 7, 36, 24 * 365
LESSON_H100, LESSON_B200, LESSON_NVL72, NVL72_GPUS = 0.09, 0.02, 0.012, 72


def fleet(tokens_per_year, tok_s):
    """GPU-years of serving at `tok_s` per GPU."""
    return tokens_per_year / (tok_s * 3600 * HOURS)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    h8, b, gb = ref.STACKS[1], ref.STACKS[3], ref.STACKS[4]
    savings = SPEND * (1 - 1 / GAP)
    gpu_year, gb_year = b.price_per_gpu_hour * HOURS, gb.price_per_gpu_hour * HOURS
    code_tokens = SPEND / ref.cost_per_million_tokens(ACTIVE, h8) * 1e6
    lesson_tokens = SPEND / LESSON_H100 * 1e6
    b_tps = ref.decode_throughput(ACTIVE, b)
    need = fleet(code_tokens, b_tps)
    return {
        "savings": round(savings),
        "gpu_year": round(gpu_year),
        "fundable": round(savings / gpu_year, 1),
        "code_tokens": code_tokens,
        "need": round(need, 2),
        "buy": math.ceil(need),
        "payback_months": round(math.ceil(need) * gpu_year / savings * 12, 1),
        "lesson_tok_s": round(lesson_tokens / (3600 * HOURS)),
        "lesson_need": math.ceil(fleet(lesson_tokens, b_tps)),
        "price_x": (
            round(ref.cost_per_million_tokens(ACTIVE, h8) / LESSON_H100, 1),
            round(ref.cost_per_million_tokens(ACTIVE, b) / LESSON_B200, 1),
        ),
        "gaps": (
            round(LESSON_H100 / LESSON_NVL72, 1),
            round(LESSON_H100 / LESSON_B200, 1),
        ),
        "per_rack_gpu": round(savings / NVL72_GPUS),
        "gb_fundable": round(savings / gb_year, 1),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: $1.71M of savings pays for 40 B200s; the workload needs 7",
            result["savings"] == 1_714_286
            and math.floor(result["fundable"]) == 40
            and result["buy"] == 7
            and result["payback_months"] < 3,
            f"savings ${result['savings']:,} / ${result['gpu_year']:,} per B200-year = "
            f"{result['fundable']} GPUs; {result['code_tokens']:.3g} tokens need "
            f"{result['need']} GPUs, repaid in {result['payback_months']} months",
        ),
        practice.Check(
            "FINDING: the lesson's own $/M sizes the same fleet 83x larger",
            result["lesson_need"] == 551 and result["price_x"] == (82.9, 52.1),
            f"$0.09/M makes $2M {result['lesson_tok_s']:,} tok/s, {result['lesson_need']} "
            f"B200s; the module's H100/B200 $/M are {result['price_x']}x the lesson's",
        ),
        practice.Check(
            "FINDING: the 7x is a GB200 NVL72 number, and an NVL72 is 72 GPUs",
            result["gaps"] == (7.5, 4.5) and result["gb_fundable"] < NVL72_GPUS / 2,
            f"lesson gaps NVL72 {result['gaps'][0]}x, HGX B200 {result['gaps'][1]}x; "
            f"savings are ${result['per_rack_gpu']:,} per rack GPU and fund "
            f"{result['gb_fundable']} GB200 GPU-years",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
