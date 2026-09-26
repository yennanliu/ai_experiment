"""Exercise 5 — on-demand wins, and Provisioned Throughput cannot break even even at full load.

    Read the Azure OpenAI and Bedrock pricing pages. For a 100M-token/month
    Claude workload, which is cheaper — direct Anthropic API, Bedrock
    on-demand, or Bedrock Provisioned Throughput?

Reading of the exercise: the pages were read on 2026-09-26 and what they
actually publish is stored in `PAGES`; where a page withholds a number, the
reference's own figure stands in and says so. 100M tokens are split 3:1
input:output, the mix of the reference's first demo workload, and a month
is 730 hours.

**ANSWER: on-demand, by 25x -- direct API and Bedrock on-demand tie at $600,
Provisioned Throughput costs $15,330.** Anthropic's price list has Claude
Sonnet 4.6 at $3/$15 per M, the rate the reference uses for Bedrock, so both
come to 75M x $3 + 25M x $15 = $600. Bedrock's regional endpoints -- which
guarantee routing through a named region -- carry a 10% premium for Claude 4.5 and
later, $660. One Provisioned Throughput unit at the lesson's $21/hour floor
is $15,330 a month, and this workload would fill 11.4% of it. Cheapest of
all: Sonnet 5 on the direct API, $2/$10, $400.

**FINDING: Provisioned Throughput cannot break even at any utilization in
the reference's numbers.** One unit delivers 876M tokens a month; at full
load those tokens are worth $5,256 on-demand at 3:1, and $13,140 even if
every one is billed at the $15 output rate. The unit costs $15,330. There is
no volume at which it wins; the case for it is capacity and latency, not price.

**FINDING: the pages do not carry the numbers the exercise asks for.**
Bedrock's page says "For Provisioned Throughput pricing, please reach out to
your account team" -- the lesson's "$21-$50/hr" is not on it -- and its
rows for current Claude models did not render in the fetch, so Bedrock
on-demand here is the reference's $3/$15. The Azure
OpenAI page lists no Claude model at all (Claude on Microsoft Foundry bills
at Anthropic's rates). And Claude 3.7 Sonnet, the model the lesson opens
with, is gone from Anthropic's price list; the legacy Claude 3.5 Sonnet
Bedrock still serves costs $6/$30, double its launch price, and Bedrock's
lifecycle page says no new Provisioned Throughput can be created for a
Legacy model.

Structure: `PAGES` holds the figures read off the pages; `monthly()` prices
one path; `solve()` compares the three and tests Provisioned Throughput at
full load.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "01-managed-llm-platforms"
TOKENS, IN_SHARE, HOURS = 100e6, 0.75, 730
PAGES = {  # read 2026-09-26
    "anthropic": {"Claude Sonnet 5": (2.0, 10.0), "Claude Sonnet 4.6": (3.0, 15.0)},
    "anthropic_lists_3_7": False,
    "bedrock_regional_premium": 0.10,  # Claude 4.5+, per Anthropic's pricing page
    "bedrock_pt": "For Provisioned Throughput pricing, please reach out to your account team.",
    "bedrock_legacy_3_5_sonnet": (6.0, 30.0),  # public extended access, from 1 Dec 2025
    "azure_openai_lists_claude": False,
}


def monthly(rate_in, rate_out, tokens=TOKENS, share=IN_SHARE):
    return tokens * share / 1e6 * rate_in + tokens * (1 - share) / 1e6 * rate_out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    bedrock = ref.PLATFORMS[0]
    capacity = bedrock.ptu_tokens_per_hour * HOURS
    on_demand = monthly(bedrock.per_mtok_input, bedrock.per_mtok_output)
    pt = bedrock.ptu_hourly * HOURS
    return {
        "direct": {m: monthly(*r) for m, r in PAGES["anthropic"].items()},
        "bedrock": (
            on_demand,
            round(on_demand * (1 + PAGES["bedrock_regional_premium"]), 2),
        ),
        "bedrock_rate": (bedrock.per_mtok_input, bedrock.per_mtok_output),
        "pt": pt,
        "util": TOKENS / capacity,
        "capacity": capacity,
        "full_load": (
            monthly(bedrock.per_mtok_input, bedrock.per_mtok_output, capacity),
            monthly(0, bedrock.per_mtok_output, capacity, 0),
        ),
        "doc_range": "$21-$50/hr" in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    d, (od, regional), full = result["direct"], result["bedrock"], result["full_load"]
    return [
        practice.Check(
            "ANSWER: on-demand, by 25x -- direct and Bedrock on-demand tie at $600, PT costs $15,330",
            all(
                [
                    d["Claude Sonnet 4.6"] == od == 600.0,
                    regional == 660.0,
                    result["pt"] == 15330.0,
                    round(result["util"], 3) == 0.114,
                    d["Claude Sonnet 5"] == 400.0,
                    result["bedrock_rate"] == PAGES["anthropic"]["Claude Sonnet 4.6"],
                ]
            ),
            f"direct {d}; Bedrock on-demand ${od:,.0f} (${regional:,.0f} regional); one PT unit "
            f"${result['pt']:,.0f}, {result['pt'] / od:.1f}x, at {result['util']:.1%} utilization",
        ),
        practice.Check(
            "FINDING: Provisioned Throughput cannot break even at any utilization",
            max(full) < result["pt"],
            f"a unit's {result['capacity'] / 1e6:,.0f}M tokens/month are worth ${full[0]:,.0f} "
            f"on-demand at 3:1 and ${full[1]:,.0f} all at the output rate, against "
            f"${result['pt']:,.0f}",
        ),
        practice.Check(
            "FINDING: the pages do not carry the numbers the exercise asks for",
            all(
                [
                    result["doc_range"],
                    "account team" in PAGES["bedrock_pt"],
                    not PAGES["azure_openai_lists_claude"],
                    not PAGES["anthropic_lists_3_7"],
                ]
            ),
            "Bedrock withholds PT prices (the lesson's $21-$50/hr is not on the page); Azure "
            "OpenAI lists no Claude; Claude 3.7 Sonnet is off Anthropic's price list; legacy "
            f"Claude 3.5 Sonnet on Bedrock is ${PAGES['bedrock_legacy_3_5_sonnet'][0]:.0f}/"
            f"${PAGES['bedrock_legacy_3_5_sonnet'][1]:.0f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
