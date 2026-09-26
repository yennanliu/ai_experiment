"""Exercise 5 — no prefix length makes batch + cache cheaper, because its bill rises with the prefix.

    Compute break-even: at what shared-prefix length does batch + cache become
    cheaper than running overnight on your own reserved GPU?

Reading of the exercise: the workload is exercise 1's, 100k documents a
night with 2000 per-document tokens and 500 output tokens. The API side is the
reference's `cost_batch_cache` at the reference's prices. The GPU side uses
figures from sibling lessons of this phase. The price is $2.50 per H100-hour,
lesson 07's H100 + FP8 + vLLM stack. The throughput is 2,300 tok/s, the
middle of lesson 04's 2,200-2,400 for Llama 3.3 70B FP8 on one H100. A
reserved GPU is paid for all day, $60, and the job runs in an 8-hour night.
With prefix caching the shared prefix is prefilled once, so the GPU's cost
does not depend on its length. The comparison sets a 70B open model against
the reference's $3 / $15 API prices, as the exercise does, and it counts only
output tokens against GPU capacity.

**ANSWER: no prefix length makes batch + cache cheaper, because its bill
rises with the prefix.** Each prefix token adds $0.015 a night at 100k
documents (the batched cached read, $0.15 per million, times 100k). Even at
a zero-length prefix the bill is $675.00, 11.25 reserved H100-days. The
break-even is in volume, not prefix: at a 3K prefix, batch + cache is
cheaper only below 8,333 documents a night.

**FINDING: the break-even exists only as a ceiling, and only for small
jobs.** At 1,000 documents a night batch + cache stays cheaper up to a
350,960-token prefix, which is longer than any context window. Past that
prefix length the GPU wins. So the exercise's direction is backwards:
"shorter than" is the only break-even the model allows.

**FINDING: a longer prefix improves only the ratio to sync, which is what
the lesson's percentages measure.** From prefix 0 to 30,000 the stack falls
from 50.0% to 10.87% of the sync bill while its dollars rise from $675.00 to
$1,125.05. A percentage of the sync bill says nothing about beating a
fixed-cost GPU.

**FINDING: one H100 covers the night.** 2,300 tok/s for 8 hours is 66.24M
output tokens, 132,480 documents of 500 tokens, against 100k needed. This
counts decode only, so it is an upper bound.

Structure: `api()` is the reference cost; `breakeven_prefix()` solves the
line through two reference evaluations for the prefix that equals a GPU-day.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "15-batch-apis"
DOCS, PER_DOC, OUT, PREFIX = 100_000, 2000, 500, 3000
GPU_HOUR, DECODE_TPS, NIGHT_H = 2.50, 2300, 8  # lessons 07 and 04 of this phase
GPU_DAY = 24 * GPU_HOUR
PREFIXES = (0, 1000, 3000, 10_000, 30_000)


def api(ref, prefix, docs=DOCS):
    return ref.cost_batch_cache(docs, prefix, PER_DOC, OUT)


def breakeven_prefix(ref, docs):
    """Prefix at which the batch+cache bill equals one reserved GPU-day (linear in prefix)."""
    base, slope = api(ref, 0, docs), (api(ref, 1000, docs) - api(ref, 0, docs)) / 1000
    return (GPU_DAY - base) / slope


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    bills = {p: round(api(ref, p), 2) for p in PREFIXES}
    ratios = {
        p: round(api(ref, p) / ref.cost_sync(DOCS, p, PER_DOC, OUT), 4)
        for p in PREFIXES
    }
    per_doc = api(ref, PREFIX) / DOCS
    capacity = DECODE_TPS * 3600 * NIGHT_H
    return {
        "bills": bills,
        "ratios": ratios,
        "slope": round((bills[1000] - bills[0]) / 1000, 4),
        "gpu_days": round(bills[0] / GPU_DAY, 2),
        "volume_breakeven": int(GPU_DAY / per_doc),
        "prefix_at_100k": breakeven_prefix(ref, DOCS),
        "prefix_at_1k": round(breakeven_prefix(ref, 1000), -1),
        "capacity": (capacity, capacity // OUT),
    }


def verify(result):
    bills, ratios = list(result["bills"].values()), list(result["ratios"].values())
    tokens, docs = result["capacity"]
    return [
        practice.Check(
            "ANSWER: no prefix length makes batch + cache cheaper, because its bill rises "
            "with the prefix",
            all(
                [
                    bills == sorted(bills),
                    result["slope"] == 0.015,
                    bills[0] == 675.0,
                    result["gpu_days"] == 11.25,
                    result["prefix_at_100k"] < 0,
                    result["volume_breakeven"] == 8333,
                ]
            ),
            f"batch+cache bill by prefix {result['bills']}, +${result['slope']} per prefix "
            f"token; ${bills[0]} at prefix 0 is {result['gpu_days']} GPU-days of "
            f"${GPU_DAY:.0f}; at a {PREFIX} prefix it wins only below "
            f"{result['volume_breakeven']} documents",
        ),
        practice.Check(
            "FINDING: the break-even exists only as a ceiling, and only for small jobs",
            result["prefix_at_1k"] == 350_960,
            f"at 1,000 documents batch+cache is cheaper up to a {result['prefix_at_1k']:,.0f}"
            f"-token prefix; at {DOCS:,} the break-even prefix is "
            f"{result['prefix_at_100k']:,.0f}",
        ),
        practice.Check(
            "FINDING: a longer prefix improves only the ratio to sync, which is what the "
            "lesson's percentages measure",
            ratios == sorted(ratios, reverse=True)
            and ratios[0] == 0.5
            and ratios[-1] == 0.1087,
            f"ratio to sync by prefix {result['ratios']} while dollars rise "
            f"${bills[0]} -> ${bills[-1]}",
        ),
        practice.Check(
            "FINDING: one H100 covers the night",
            tokens == 66_240_000 and docs == 132_480 and docs > DOCS,
            f"{DECODE_TPS} tok/s x {NIGHT_H}h = {tokens:,} output tokens = {docs:,} "
            f"documents of {OUT} tokens, against {DOCS:,}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
