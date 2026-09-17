"""Exercise 3 — the simulator caches a prefix the endpoint would refuse.

    **Hard.** Build a layout optimizer: given a prompt and a list of fields
    marked `stable=True/False`, rewrite the prompt to put a single cache
    breakpoint at the maximum cache-friendly position without losing
    information. Verify on a real Anthropic endpoint.

Reading of the exercise: the prompt is eight labelled fields totalling 8,255
tokens, four stable and four not, ordered the way a prompt is usually written
-- instructions, tools, then whatever the request carries. "Maximum
cache-friendly position" is the end of the longest leading run of stable
fields, so the optimizer's only move is to bring every stable field in front of
every unstable one. The live verification is the one step this solution does
not run: no `ANTHROPIC_API_KEY` is set, so the layout is priced against the
lesson's own accountant and the endpoint check is written down, not asserted.

**ANSWER: the cacheable prefix goes from 4,000 to 6,100 tokens, +52.5%.** In
the original order the leading stable run is the role instructions and the tool
catalogue, and `user display name` truncates it. After the rewrite all four
stable fields precede all four unstable ones. Every field label survives and
the token total is unchanged at 8,255.

**MECHANISM: no number of breakpoints substitutes for the reorder.** A
breakpoint caches a *prefix*, and the first unstable field truncates every
prefix that follows it -- so the 2,100 stable tokens sitting after
`user display name` are unreachable, whatever Anthropic's four-breakpoint
allowance is spent on. Reordering is the only lever this exercise has.

**ANSWER: over 100 requests the bill goes $7.1205 to $4.3580** against a
$12.3825 no-cache baseline -- 42.5% saved becomes 64.8% saved. The 2,100 tokens
moved are worth $2.76 per hundred requests at Opus prices.

**FINDING: the simulator has no minimum cacheable prefix.** Anthropic will not
cache a prefix below 1,024 tokens, and `simulate_anthropic` will: a 900-token
stable block is modelled as two writes and 98 reads and reported as **9.6%
saved**, a saving the endpoint would refuse to deliver. An optimizer that trusts
the accountant recommends a breakpoint that does nothing, so it has to carry
the minimum itself and decline.

**FINDING: "without losing information" holds for the fields and not for the
order.** The rewrite moves `style guide` and `output schema` -- the latter
conventionally last, so the model reads it closest to generation -- in front of
the retrieved documents and the question. Labels and token counts survive;
position does not, and position is instruction in a prompt. That is the cost the
exercise does not mention, and it falls on 2 of the 4 stable fields.

Structure: `FIELDS` is the prompt, `stable_prefix` measures the leading stable
run, `optimise` is the rewrite, `price` runs one layout through the lesson's
accountant, and `MINIMUM` is Anthropic's published floor.
"""

from __future__ import annotations

import os

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "15-prompt-caching"
MODEL, MINIMUM, REQUESTS, TTL, GAP = "anthropic_claude_opus_4_7", 1024, 100, 300, 4
FIELDS = [("role instructions", 800, True), ("tool catalogue", 3200, True),
          ("user display name", 20, False), ("style guide", 1500, True),
          ("today's date", 15, False), ("retrieved documents", 2000, False),
          ("output schema", 600, True), ("the user's question", 120, False)]
BELOW_MINIMUM = 900


def stable_prefix(fields):
    """The maximum cache-friendly position: the end of the leading stable run."""
    total = 0
    for _, tokens, stable in fields:
        if not stable:
            break
        total += tokens
    return total


def optimise(fields):
    return [f for f in fields if f[2]] + [f for f in fields if not f[2]]


def stranded(fields):
    """Stable tokens sitting behind the first unstable field, unreachable by any breakpoint."""
    return sum(tokens for _, tokens, stable in fields if stable) - stable_prefix(fields)


def price(ref, prefix_tokens, total):
    log = [ref.Request(prefix_tokens=prefix_tokens, suffix_tokens=total - prefix_tokens,
                       prefix_key="stable") for _ in range(REQUESTS)]
    stats = ref.simulate_anthropic(log, ttl_seconds=TTL, seconds_between=GAP)
    baseline = ref.baseline_cost(log, MODEL)
    return {"cost": round(stats.total_cost, 4), "baseline": round(baseline, 4),
            "writes": stats.writes, "reads": stats.reads,
            "saves_pct": round((1 - stats.total_cost / baseline) * 100, 1)}


def moved(fields):
    """Stable fields that end up in front of content they originally followed."""
    order = [name for name, _, _ in fields]
    rewritten = [name for name, _, _ in optimise(fields)]
    return [name for name, _, stable in fields
            if stable and rewritten.index(name) < order.index(name)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    total = sum(tokens for _, tokens, _ in FIELDS)
    rewritten = optimise(FIELDS)
    before, after = stable_prefix(FIELDS), stable_prefix(rewritten)
    return {
        "fields": len(FIELDS), "stable": sum(1 for f in FIELDS if f[2]), "total": total,
        "before": before, "after": after, "gain_pct": round((after / before - 1) * 100, 1),
        "labels_kept": sorted(f[0] for f in FIELDS) == sorted(f[0] for f in rewritten),
        "tokens_kept": total == sum(tokens for _, tokens, _ in rewritten),
        "stranded": stranded(FIELDS), "moved": moved(FIELDS),
        "before_bill": price(ref, before, total), "after_bill": price(ref, after, total),
        "tiny_bill": price(ref, BELOW_MINIMUM, total), "minimum": MINIMUM,
        "live_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }


def verify(result):
    before, after, tiny = result["before_bill"], result["after_bill"], result["tiny_bill"]
    return [
        practice.Check(
            "ANSWER: the cacheable prefix goes from 4,000 to 6,100 tokens, +52.5%",
            all([result["before"] == 4000, result["after"] == 6100,
                 result["labels_kept"], result["tokens_kept"]]),
            f"{result['fields']} fields totalling {result['total']:,} tokens, "
            f"{result['stable']} of them stable: the leading stable run is "
            f"{result['before']:,} tokens and the rewrite reaches {result['after']:,}, "
            f"+{result['gain_pct']}%. Every label survives and the total is unchanged",
        ),
        practice.Check(
            "MECHANISM: no number of breakpoints substitutes for the reorder",
            all([result["stranded"] == 2100,
                 result["before"] + result["stranded"] == result["after"]]),
            f"a breakpoint caches a prefix, and the first unstable field truncates every "
            f"prefix after it, so the {result['stranded']:,} stable tokens behind "
            "`user display name` are unreachable however Anthropic's four breakpoints are "
            "spent. Reordering is the only lever",
        ),
        practice.Check(
            "ANSWER: over 100 requests the bill goes $7.1205 to $4.3580",
            all([before["cost"] == 7.1205, after["cost"] == 4.358,
                 before["baseline"] == after["baseline"]]),
            f"{REQUESTS} requests against a ${before['baseline']} no-cache baseline: "
            f"${before['cost']} before the rewrite and ${after['cost']} after, "
            f"{before['saves_pct']}% saved becoming {after['saves_pct']}%. The "
            f"{result['stranded']:,} tokens moved are worth "
            f"${before['cost'] - after['cost']:.2f} per hundred requests",
        ),
        practice.Check(
            "FINDING: the simulator has no minimum cacheable prefix",
            all([BELOW_MINIMUM < result["minimum"], tiny["writes"] == 2,
                 tiny["reads"] == REQUESTS - 2, tiny["saves_pct"] == 9.6]),
            f"Anthropic will not cache a prefix below {result['minimum']:,} tokens and "
            f"`simulate_anthropic` will: {BELOW_MINIMUM} tokens is modelled as "
            f"{tiny['writes']} writes and {tiny['reads']} reads and reported as "
            f"{tiny['saves_pct']}% saved. An optimizer that trusts the accountant "
            "recommends a breakpoint that does nothing",
        ),
        practice.Check(
            "FINDING: 'without losing information' holds for the fields, not for the order",
            all([result["moved"] == ["style guide", "output schema"],
                 not result["live_key"]]),
            f"the rewrite moves {result['moved']} in front of content they were written to "
            f"follow -- 2 of the {result['stable']} stable fields. Labels and token counts "
            "survive; position does not, and position is instruction in a prompt. The live "
            "endpoint check is documented and not run: ANTHROPIC_API_KEY is unset, so the "
            "layout is verified against the lesson's own accountant",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
