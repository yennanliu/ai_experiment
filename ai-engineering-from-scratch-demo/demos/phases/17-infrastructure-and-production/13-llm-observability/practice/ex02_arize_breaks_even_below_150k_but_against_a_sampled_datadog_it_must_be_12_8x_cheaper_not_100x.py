"""Exercise 2 — Arize breaks even below $150K, but against a sampled Datadog it must be 12.8x cheaper, not 100x.

    At 5M traces/day with Datadog quotes $150K/month, compute break-even for
    Arize AX.

Reading of the exercise: break-even is the Arize AX contract price at which
its all-in monthly cost equals the alternative. All-in means the Arize fee
plus the S3 bill for the Parquet it reads, since zero-copy stores traces in
your own lake. The alternative is computed twice: Datadog at the quote with
full retention, and Datadog at the quote's per-GB rate under the lesson's own
sampling rule. Volumes come from the reference `simulate_day()` at 5M traces.

**ANSWER: Arize breaks even at a $149,984/month contract, and against a
sampled Datadog at $11,725.** 5M traces at the reference's 4,500 bytes is
22.5 GB/day, 675 GB/month, so the quote is $222/GB, or $1.00 per 1,000
traces. Your own S3 at 30-day retention is $15.53/month. That leaves
$149,984 for the Arize contract. The lesson's "100x cheaper" would be
$1,500/month. But the lesson's own rule, "5% success + errors + $$$",
keeps 391,366 of 5M traces (7.83%), which cuts the Datadog bill to $11,741. Full-retention Arize must come in below $11,725, a
12.8x discount, to beat that. Sampling alone cuts the bill 92%, with no
vendor change; the 100x claim is worth 99%.

**FINDING: the 100x is a ratio of two constants, and the S3 bill cuts it
to 17.9x.** `OBSERVABILITY_INGEST_PER_GB / ARIZE_AX_PER_GB` = 0.50 / 0.005 =
100 exactly, at every volume and strategy. The `arize` column leaves out
the S3 column, which is 4.6x larger ($15.53 against $3.38 at 5M/day).
Counting it, the reference's own rates give $337.50 against $18.90, or 17.9x.

**FINDING: the code prices Datadog 444x below the quote, so its own table
contradicts its own "hundreds of $/day".** At $0.50/GB, 5M traces/day costs
$337.50/month, and the 1M-trace table shows $67.50/month, $2.25/day, for 100%
retention. The printed "Read:" line says hundreds of dollars a day. That is
true only at the quote's $222/GB, where 1M/day is $1,000/day.

**FINDING: Arize's published plans stop at 50K spans a month.** At 5M
traces/day that is 14.4 minutes of traffic, even at one span per trace. On
arize.com/pricing (read 2026-09-26), AX Pro is $50/month for 50K spans and
Enterprise is "Custom". So this volume's price is a negotiation, and the
number above is the most that negotiation can reach before Arize stops paying off.

Structure: `monthly()` turns a `simulate_day()` row into GB/month; each
break-even subtracts the S3 bill from the alternative's cost.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "13-llm-observability"
TRACES, QUOTE, DAYS = 5_000_000, 150_000, 30
PRO_SPANS = 50_000  # AX Pro, arize.com/pricing, read 2026-09-26


def monthly(row):
    return row["gb_per_day"] * DAYS


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    full = ref.simulate_day(ref.STRATEGIES[0], TRACES)
    sampled = ref.simulate_day(ref.STRATEGIES[3], TRACES)
    gb = monthly(full)
    per_gb = QUOTE / gb
    s3 = full["s3_month"]
    datadog_sampled = monthly(sampled) * per_gb
    one_m = ref.simulate_day(ref.STRATEGIES[0])
    return {
        "gb": gb,
        "per_gb": per_gb,
        "per_1k": QUOTE / (TRACES * DAYS) * 1000,
        "s3": s3,
        "break_even": QUOTE - s3,
        "claim": QUOTE / 100,
        "kept": sampled["retained"],
        "datadog_sampled": datadog_sampled,
        "break_even_sampled": datadog_sampled - s3,
        "constant": ref.OBSERVABILITY_INGEST_PER_GB / ref.ARIZE_AX_PER_GB,
        "arize": full["arize_month"],
        "mono": full["monolithic_month"],
        "all_in": full["monolithic_month"] / (full["arize_month"] + s3),
        "underprice": per_gb / ref.OBSERVABILITY_INGEST_PER_GB,
        "day_1m": one_m["monolithic_month"] / DAYS,
        "day_1m_quote": monthly(one_m) * per_gb / DAYS,
        "pro_minutes": PRO_SPANS / TRACES * 24 * 60,
    }


def verify(result):
    r = result
    discount = QUOTE / r["break_even_sampled"]
    return [
        practice.Check(
            "ANSWER: break-even at a $149,984 contract, and $11,725 against a sampled Datadog",
            all(
                [
                    r["gb"] == 675,
                    round(r["per_gb"], 2) == 222.22,
                    round(r["per_1k"], 2) == 1.0,
                    round(r["break_even"]) == 149_984,
                    r["kept"] == 391_366,
                    round(r["break_even_sampled"]) == 11_725,
                    round(discount, 1) == 12.8,
                ]
            ),
            f"{r['gb']:.0f} GB/month at ${r['per_gb']:.2f}/GB; S3 ${r['s3']:.2f}; break-even "
            f"${r['break_even']:,.0f} (100x claim: ${r['claim']:,.0f}); sampling keeps "
            f"{r['kept']:,} traces, Datadog ${r['datadog_sampled']:,.0f}, so Arize must be "
            f"under ${r['break_even_sampled']:,.0f}, {discount:.1f}x cheaper",
        ),
        practice.Check(
            "FINDING: the 100x is a ratio of two constants, and the S3 bill cuts it to 17.9x",
            round(r["constant"], 9) == 100
            and round(r["all_in"], 1) == 17.9
            and r["s3"] > 4 * r["arize"],
            f"0.50/0.005 = {r['constant']:.0f}; S3 ${r['s3']:.2f} vs arize ${r['arize']:.2f}; "
            f"all-in ${r['mono']:.2f} vs ${r['arize'] + r['s3']:.2f} = {r['all_in']:.1f}x",
        ),
        practice.Check(
            "FINDING: the code prices Datadog 444x below the quote, contradicting its 'hundreds of $/day'",
            round(r["underprice"]) == 444
            and r["day_1m"] == 2.25
            and round(r["day_1m_quote"]) == 1000,
            f"${r['per_gb']:.2f}/GB against $0.50 ({r['underprice']:.1f}x); 100% retention at 1M/day "
            f"is ${r['day_1m']:.2f}/day in the code, ${r['day_1m_quote']:,.0f}/day at the quote",
        ),
        practice.Check(
            "FINDING: Arize's published plans stop at 50K spans a month, 14.4 minutes of this load",
            round(r["pro_minutes"], 1) == 14.4,
            f"AX Pro's {PRO_SPANS:,} spans cover {r['pro_minutes']:.1f} minutes of a 5M-trace day; "
            "Enterprise is 'Custom', so the break-even is a ceiling for that negotiation",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
