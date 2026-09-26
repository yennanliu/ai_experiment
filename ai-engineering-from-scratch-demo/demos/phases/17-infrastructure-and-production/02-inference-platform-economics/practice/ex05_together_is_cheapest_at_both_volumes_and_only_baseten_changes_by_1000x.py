"""Exercise 5 — Together is cheapest at both volumes, and only Baseten changes, by 1000x.

    Compare cost per 1,000 predictions for Llama 3.1 70B on Fireworks
    serverless, Together on-demand, Baseten dedicated, and Replicate API.
    Which is cheapest at 10 predictions/day? At 10,000?

Reading of the exercise: a prediction is one chat completion of 200 output
tokens, the lesson's Scenario A ratio (2M tokens / 10,000 predictions). Cost
per 1,000 is the reference `cost_per_day` divided by predictions/day, times
1,000. No API is called; the prices are the lesson's, plus Baseten's
published H100 rate for comparison.

**ANSWER: Together at both volumes, $0.176 per 1,000.** At 10 and at 10,000
predictions/day the costs per 1,000 are Together $0.176, Fireworks $0.18 and
Replicate $6.00. Baseten dedicated is $79,200 at 10/day and $79.20 at
10,000/day. Per-token and per-prediction prices are linear, so their cost per
1,000 does not move with volume. Only the reserved GPU amortizes, and it
falls 1000x across the range the exercise names. Baseten passes Together only
at 4.5M predictions/day on the code's $0.55/min, or 886k/day at its listed
H100 $0.10833/min, where 10/day costs $15,599.52 per 1,000.

**FINDING: the lesson's "50-70% cheaper than Replicate" holds only for
predictions of 2,045-3,409 tokens.** Replicate's flat $0.006 a prediction
against Together's $0.88/M puts the tie at 6,818 tokens. At the scenario's
200 tokens Together is 97.1% cheaper, and above 6,818 tokens Replicate is
the cheaper of the two. Whatever the answer is, it is set by prediction
length, which the exercise does not state.

**FINDING: nothing in the module is Llama 3.1, and Replicate's per-prediction
LLM price could not be confirmed.** The vendor models are "Llama 70B",
"Custom Llama 70B" and "Llama 70B RayTurbo". Replicate's pricing page
(fetched 2026-09-26) bills its example language model, deepseek-r1, per
input and output token. Its `meta/meta-llama-3-70b-instruct` page showed no
price, and no Llama 3.1 70B listing was found. The $0.006 is therefore the
lesson's figure, not a verified one.

Structure: `per_thousand()` wraps the reference `cost_per_day`;
`tie_tokens()` solves the per-token vs per-prediction crossover in closed
form.
"""

from __future__ import annotations

import dataclasses

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "02-inference-platform-economics"
TOKENS_PER_PREDICTION = 200  # Scenario A: 2,000,000 tokens / 10,000 predictions
BASETEN_H100 = 0.10833  # $/min, baseten.co/pricing, fetched 2026-09-26
NAMED = ("Fireworks", "Together", "Baseten", "Replicate")


def per_thousand(ref, vendor, per_day, tokens=TOKENS_PER_PREDICTION):
    return round(ref.cost_per_day(vendor, tokens * per_day, per_day) / per_day * 1000, 3)


def tie_tokens(per_prediction, per_mtok):
    return per_prediction / (per_mtok / 1e6)


def baseten_passes(vendor, rival):
    """Predictions/day at which a 24h-reserved GPU undercuts per-token."""
    daily = vendor.per_minute * vendor.min_reserved_minutes_per_day
    return round(daily / (TOKENS_PER_PREDICTION * rival.per_mtok_output / 1e6))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    v = {x.name: x for x in ref.VENDORS}
    table = {n: {d: per_thousand(ref, v[n], d) for d in (10, 10_000)} for n in NAMED}
    listed = dataclasses.replace(v["Baseten"], per_minute=BASETEN_H100)
    tie = tie_tokens(v["Replicate"].per_prediction, v["Together"].per_mtok_output)
    return {
        "table": table,
        "cheapest": {d: min(NAMED, key=lambda n: table[n][d]) for d in (10, 10_000)},
        "passes": (baseten_passes(v["Baseten"], v["Together"]), baseten_passes(listed, v["Together"])),
        "listed_10": per_thousand(ref, listed, 10),
        "tie": round(tie), "band": (round(0.3 * tie), round(0.5 * tie)),
        "saving_200": round(1 - table["Together"][10] / table["Replicate"][10], 3),
        "doc_claim": "50-70% cheaper than Replicate" in parity.doc_text(PHASE, LESSON),
        "models": sorted({x.model for x in ref.VENDORS}),
    }


def verify(result):
    t = result["table"]
    return [
        practice.Check(
            "ANSWER: Together at both volumes, $0.176 per 1,000",
            all([result["cheapest"] == {10: "Together", 10_000: "Together"},
                 t["Baseten"] == {10: 79200.0, 10_000: 79.2},
                 all(t[n][10] == t[n][10_000] for n in ("Fireworks", "Together", "Replicate")),
                 result["passes"] == (4_500_000, 886_336)]),
            f"$ per 1,000 predictions {t}; Baseten passes Together at "
            f"{result['passes'][0]:,}/day (code) or {result['passes'][1]:,}/day (listed H100, "
            f"${result['listed_10']:,} per 1,000 at 10/day)",
        ),
        practice.Check(
            "FINDING: '50-70% cheaper than Replicate' holds only for 2,045-3,409-token predictions",
            result["doc_claim"] and result["tie"] == 6818 and result["band"] == (2045, 3409)
            and result["saving_200"] == 0.971,
            f"Together ties Replicate at {result['tie']} tokens; 50-70% cheaper needs "
            f"{result['band']} tokens; at 200 tokens it is {result['saving_200']:.1%} cheaper",
        ),
        practice.Check(
            "FINDING: nothing in the module is Llama 3.1",
            not any("3.1" in m for m in result["models"]),
            f"vendor models {result['models']}; Replicate's per-prediction LLM price is the "
            "lesson's figure, not confirmed on replicate.com",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
