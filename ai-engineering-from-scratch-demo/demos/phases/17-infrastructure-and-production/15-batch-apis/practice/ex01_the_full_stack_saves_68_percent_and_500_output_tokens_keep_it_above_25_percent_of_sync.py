"""Exercise 1 — the full stack saves 68%, and 500 output tokens keep it above 25% of sync.

    Run `code/main.py`. For a 100k-doc pipeline with 3K-token system prompt
    and 500-token output, compute the savings of full stack (batch + cache) vs
    sync baseline.

Reading of the exercise: the exercise fixes docs, prefix and output but not
the per-document input, so the headline uses the reference's own
summarization value, 2000 tokens, and the answer is also swept over 0 and the
reference's other two values (300, 15000). Every cost comes from the
reference's `cost_sync` and `cost_batch_cache`.

**ANSWER: batch + cache takes $2,250.00 to $720.01, a saving of $1,529.99
(68.0%).** Over per-document input 0 / 300 / 2000 / 15000 tokens the stack
lands at 25.45% / 26.72% / 32.0% / 43.41% of the sync bill.

**FINDING: the ~10% the lesson promises is out of reach here.** Even with no
per-document tokens the 500 output tokens hold the stack at 25.45%: output
is only halved, never cached. With zero document tokens the ratio is
0.5 * (0.3P + 15O) / (3P + 15O), which reaches 10% only at P = 40 * O --
a 20,000-token prompt for this pipeline. The reference's own three runs print
24.3%, 17.1% and 41.3% under a banner that says "~10%". The Problem's
50k-doc, 4K-prompt pipeline ("$2,000 -> $180, ~9%") prints $1,050.00 ->
$255.01, 24.3%, and 14.0% even with no document tokens.

**FINDING: the reference assumes every cache read hits; Anthropic documents
30% to 98%.** Anthropic's batch docs say cache hits in a batch are
best-effort, "30% to 98%", and suggest the 1-hour cache because batches
outlast 5 minutes. The reference prices a 5-minute write once and 100% hits
after it. Price a miss as a fresh write and the stack is 32.46% of sync at
98%, 43.5% at 50% and 48.1% at 30%. Plain batch is 50%. Below a 21.7% hit
rate, `cache_control` makes the batch dearer than no cache at all.

Structure: `stack()` calls the reference cost functions; `hit_rate()` patches
the module's `CACHED_INPUT` to the blended read/write price and restores it.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "15-batch-apis"
DOCS, PREFIX, OUT, PER_DOC = 100_000, 3000, 500, 2000
SWEEP = (0, 300, 2000, 15000)
HITS = (0.3, 0.5, 0.98, 1.0)
REF_RUNS = (
    (50_000, 4000, 2000, 200),
    (200_000, 1500, 300, 50),
    (1_000, 6000, 15_000, 2000),
)


def stack(ref, docs, prefix, per_doc, out):
    """(sync, batch+cache, ratio) from the reference's own functions."""
    sync = ref.cost_sync(docs, prefix, per_doc, out)
    full = ref.cost_batch_cache(docs, prefix, per_doc, out)
    return round(sync, 2), round(full, 2), round(full / sync, 4)


def hit_rate(ref, rate):
    """Stack ratio when only `rate` of cache reads hit and a miss writes again."""
    original = ref.CACHED_INPUT
    ref.CACHED_INPUT = rate * original + (1 - rate) * ref.CACHE_WRITE_5MIN
    try:
        return stack(ref, DOCS, PREFIX, PER_DOC, OUT)[2]
    finally:
        ref.CACHED_INPUT = original


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    breakeven = (ref.BASE_INPUT - ref.CACHE_WRITE_5MIN) / (
        ref.CACHED_INPUT - ref.CACHE_WRITE_5MIN
    )
    return {
        "headline": stack(ref, DOCS, PREFIX, PER_DOC, OUT),
        "sweep": {d: stack(ref, DOCS, PREFIX, d, OUT)[2] for d in SWEEP},
        "forty_x": stack(ref, DOCS, 40 * OUT, 0, OUT)[2],
        "ref_runs": [round(stack(ref, *run)[2] * 100, 1) for run in REF_RUNS],
        "problem": stack(ref, 50_000, 4000, 2000, 200),
        "problem_floor": stack(ref, 50_000, 4000, 0, 200)[2],
        "hits": {h: hit_rate(ref, h) for h in HITS},
        "cache_breakeven": round(breakeven, 4),
        "plain_batch": ref.BATCH_DISCOUNT,
    }


def verify(result):
    sync, full, ratio = result["headline"]
    sweep, hits = result["sweep"], result["hits"]
    return [
        practice.Check(
            "ANSWER: batch + cache takes $2,250.00 to $720.01, a saving of $1,529.99 (68.0%)",
            (sync, full, ratio) == (2250.0, 720.01, 0.32)
            and list(sweep.values()) == [0.2545, 0.2672, 0.32, 0.4341],
            f"sync ${sync} -> full stack ${full}, saving ${sync - full:.2f}; ratio by "
            f"per-document tokens {sweep}",
        ),
        practice.Check(
            "FINDING: the ~10% the lesson promises is out of reach here",
            all(
                [
                    sweep[0] > 0.25,
                    round(result["forty_x"], 3) == 0.1,
                    result["ref_runs"] == [24.3, 17.1, 41.3],
                    result["problem"][:2] == (1050.0, 255.01),
                    result["problem_floor"] == 0.14,
                ]
            ),
            f"floor with no document tokens {sweep[0]}; a {40 * OUT}-token prompt reaches "
            f"{result['forty_x']}; the reference prints {result['ref_runs']}%; the "
            f"Problem pipeline is ${result['problem'][0]} -> ${result['problem'][1]}, "
            f"floor {result['problem_floor']}",
        ),
        practice.Check(
            "FINDING: the reference assumes every cache read hits; Anthropic documents 30% to 98%",
            all(
                [
                    hits[1.0] == ratio,
                    hits[0.98] == 0.3246,
                    hits[0.5] == 0.435,
                    hits[0.3] == 0.481,
                    hits[0.3] < result["plain_batch"],
                    result["cache_breakeven"] == 0.2174,
                ]
            ),
            f"stack ratio by hit rate {hits} against plain batch {result['plain_batch']}; "
            f"below a {result['cache_breakeven']:.1%} hit rate cache_control costs more "
            "than no cache",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
