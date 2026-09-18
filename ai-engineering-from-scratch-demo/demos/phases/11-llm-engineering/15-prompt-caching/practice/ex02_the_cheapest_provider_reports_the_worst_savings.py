"""Exercise 2 — the cheapest provider reports the worst savings.

    **Medium.** Write a test harness that, given a prompt template and a
    request log, computes the expected hit rate and dollar savings per provider
    (Anthropic 5m, Anthropic 1h, OpenAI automatic, Gemini explicit).

Reading of the exercise: the request log is the lesson's own scenario -- 500
requests, three rotating 15,000-token prefixes, four seconds apart -- because a
harness is only as good as the log it is given and this is the log the lesson
reports on. The four arms are run, the two numbers the exercise asks for are
computed, and then each is checked for whether it means what it looks like.

**ANSWER: the four arms.**

```text
                 writes  reads   hit rate   cost       saves
Anthropic 5m         21    479      0.958   $19.6838   83.0%
Anthropic 1h          3    497      0.994   $15.5325   86.6%
OpenAI automatic      3    497      0.994   $19.8625   48.4%
Gemini 1h explicit    3    497      0.994   $ 2.6365   72.6%
```

**FINDING: the hit rate carries no information the cost does not.**
`writes + reads == 500` in every arm and `misses` is never incremented, so the
hit rate is exactly `1 - writes/500` and the two columns the exercise asks for
are one measurement. Three of the four arms have the identical hit rate and
costs that differ by 7.5x.

**FINDING: the TTL is never refreshed on a read.** An entry expires
`ttl_seconds` after it was *written*, so the 5-minute arm rewrites each prefix
on a 300-second clock whatever the traffic does: 1,996 seconds of log, seven
writes per prefix, 21 in total. Anthropic refreshes the five-minute window on
every hit, and with that one change the 5-minute arm writes 3 and costs
**$15.0263** against the 1-hour arm's **$15.5325** -- so the option the shipped
model prices as the expensive one is actually the cheaper one, and the harness
recommends paying the 2x write premium for nothing.

**FINDING: Gemini's storage is billed once per surviving entry.** The loop runs
over `cache.values()` after the simulation, so an entry rewritten seven times
is charged one tenancy. At a 300-second TTL the arm records 21 writes and bills
storage for 3 entries -- 18 tenancies free. And `simulate_openai` hardcodes
`3600` in its body, so the one arm described as "automatic" is the one arm that
cannot be run at another TTL.

**FINDING: dollar savings are not comparable across providers.** Each arm is
divided by its own baseline, so the percentage measures how cacheable a
provider's price list is, not how cheap it is. Gemini costs **$2.6365** against
Anthropic 1h's **$15.5325** -- 5.9x less -- and reports the worse saving, 72.6%
against 86.6%. A harness that ranks providers by savings picks the most
expensive one.

Structure: `LOG` is the request log, `ARMS` names the four provider
configurations, `measure` runs one arm and returns its two numbers, and
`refreshing` is `simulate_anthropic` with the TTL renewed on a read.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "15-prompt-caching"
ANTHROPIC, OPENAI, GEMINI = "anthropic_claude_opus_4_7", "openai_gpt_5", "gemini_3_pro"
REQUESTS, PREFIXES, PREFIX_TOKENS, SUFFIX_TOKENS, GAP = 500, 3, 15000, 400, 4


def arms(ref):
    return {"anthropic_5m": (lambda log: ref.simulate_anthropic(log, 300, GAP), ANTHROPIC),
            "anthropic_1h": (lambda log: ref.simulate_anthropic(log, 3600, GAP), ANTHROPIC),
            "openai": (lambda log: ref.simulate_openai(log, GAP), OPENAI),
            "gemini_1h": (lambda log: ref.simulate_gemini(log, 3600, GAP), GEMINI)}


def measure(ref, run, provider, log):
    stats = run(log)
    baseline = ref.baseline_cost(log, provider)
    return {"writes": stats.writes, "reads": stats.reads, "misses": stats.misses,
            "hit_rate": round(stats.reads / len(log), 3),
            "cost": round(stats.total_cost, 4),
            "saves_pct": round((1 - stats.total_cost / baseline) * 100, 1),
            "accounted": stats.writes + stats.reads == len(log)}


def refreshing(ref, log, ttl):
    """simulate_anthropic with the TTL renewed on a read, as Anthropic documents."""
    price = ref.PRICES[ANTHROPIC]
    write_rate = price["cache_write_1h"] if ttl > 300 else price["cache_write_5m"]
    writes, cost, seen = 0, 0.0, {}
    for index, request in enumerate(log):
        now = index * GAP
        last = seen.get(request.prefix_key)
        if last is None or now - last >= ttl:
            writes += 1
            cost += request.prefix_tokens / 1000 * write_rate
        else:
            cost += request.prefix_tokens / 1000 * price["cache_read"]
        seen[request.prefix_key] = now
        cost += request.suffix_tokens / 1000 * price["base"]
    return {"writes": writes, "cost": round(cost, 4)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    log = ref.make_traffic(REQUESTS, PREFIXES, PREFIX_TOKENS, SUFFIX_TOKENS)
    table = {name: measure(ref, run, provider, log)
             for name, (run, provider) in arms(ref).items()}
    gemini_5m = ref.simulate_gemini(log, 300, GAP)
    return {"requests": len(log), "table": table, "span": (len(log) - 1) * GAP,
            "renewed_5m": refreshing(ref, log, 300), "renewed_1h": refreshing(ref, log, 3600),
            "gemini_5m_writes": gemini_5m.writes, "gemini_entries": PREFIXES,
            "openai_hardcodes_ttl": "3600" in inspect.getsource(ref.simulate_openai)}


def verify(result):
    table, renewed = result["table"], result["renewed_5m"]
    return [
        practice.Check(
            "ANSWER: the four arms, from $2.6365 to $19.8625 on the same log",
            all([table["anthropic_5m"]["cost"] == 19.6838,
                 table["anthropic_1h"]["cost"] == 15.5325,
                 table["openai"]["cost"] == 19.8625, table["gemini_1h"]["cost"] == 2.6365]),
            f"{result['requests']} requests over {PREFIXES} rotating "
            f"{PREFIX_TOKENS:,}-token prefixes, {GAP}s apart: "
            + ", ".join(f"{name} {arm['hit_rate']} hit / ${arm['cost']} / "
                        f"{arm['saves_pct']}%" for name, arm in table.items()),
        ),
        practice.Check(
            "FINDING: the hit rate carries no information the cost does not",
            all([all(arm["accounted"] for arm in table.values()),
                 not any(arm["misses"] for arm in table.values()),
                 len({arm["hit_rate"] for arm in table.values()}) == 2]),
            f"writes + reads == {result['requests']} in every arm and misses is never "
            "incremented, so the hit rate is exactly 1 - writes/n. Three of the four arms "
            f"share the hit rate {table['anthropic_1h']['hit_rate']} while their costs "
            f"differ by {table['openai']['cost'] / table['gemini_1h']['cost']:.1f}x",
        ),
        practice.Check(
            "FINDING: the TTL is never refreshed, so the 5-minute arm is priced backwards",
            all([table["anthropic_5m"]["writes"] == 21, renewed["writes"] == PREFIXES,
                 renewed["cost"] < table["anthropic_1h"]["cost"],
                 result["renewed_1h"]["cost"] == table["anthropic_1h"]["cost"]]),
            f"an entry expires {300}s after it was written, so over {result['span']}s of log "
            f"each prefix is rewritten seven times -- {table['anthropic_5m']['writes']} "
            f"writes. Renewing the window on a read, as Anthropic documents, gives "
            f"{renewed['writes']} writes and ${renewed['cost']} against the 1-hour arm's "
            f"${table['anthropic_1h']['cost']}: the shipped model prices the cheaper option "
            "as the expensive one",
        ),
        practice.Check(
            "FINDING: Gemini's storage is billed once per surviving entry",
            all([result["gemini_5m_writes"] == 21, result["openai_hardcodes_ttl"],
                 table["gemini_1h"]["cost"] > 0]),
            f"the storage loop runs over cache.values() after the simulation, so at a 300s "
            f"TTL the arm records {result['gemini_5m_writes']} writes and bills "
            f"{result['gemini_entries']} tenancies -- "
            f"{result['gemini_5m_writes'] - result['gemini_entries']} free. And "
            "simulate_openai hardcodes 3600 in its body, so the arm described as automatic "
            "is the one that cannot be run at another TTL",
        ),
        practice.Check(
            "FINDING: dollar savings are not comparable across providers",
            all([table["gemini_1h"]["cost"] < table["anthropic_1h"]["cost"],
                 table["gemini_1h"]["saves_pct"] < table["anthropic_1h"]["saves_pct"]]),
            f"each arm is divided by its own baseline, so the percentage measures how "
            f"cacheable a price list is. Gemini costs ${table['gemini_1h']['cost']} against "
            f"Anthropic 1h's ${table['anthropic_1h']['cost']} -- "
            f"{table['anthropic_1h']['cost'] / table['gemini_1h']['cost']:.1f}x less -- and "
            f"reports {table['gemini_1h']['saves_pct']}% against "
            f"{table['anthropic_1h']['saves_pct']}%. Ranking by savings picks the dearest",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
