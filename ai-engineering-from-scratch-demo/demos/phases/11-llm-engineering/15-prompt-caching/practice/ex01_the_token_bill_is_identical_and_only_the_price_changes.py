"""Exercise 1 — the token bill is identical and only the price changes.

    **Easy.** Take a 10-turn conversation with a 5,000-token system prompt
    against Claude. Run it without `cache_control` and then with. Report the
    input-token bill for each.

Reading of the exercise: "the input-token bill" is two numbers, not one -- the
tokens billed and the dollars billed -- and they behave differently, so both
are reported. The conversation is modelled the way a conversation actually
grows, 400 tokens of transcript per turn on top of the 5,000-token system
prompt, and it is run twice with the cache breakpoint in the two places a real
client would put it.

**ANSWER: 72,000 input tokens either way.** Caching changes the *price* of a
token, never the count: `simulate_anthropic` bills every prefix token on every
request and only chooses between three rates. Any report phrased in tokens
shows no effect at all.

**ANSWER: in dollars, $1.0800 without `cache_control` and $0.4912 with** -- one
write and nine reads of the system prompt, **54.5%** saved. The prefix is
charged at 1.25x base once and 0.10x base nine times.

**FINDING: move the breakpoint with the conversation and caching costs 23.6%
more.** A client that caches the whole transcript so far -- the layout that
maximises the cached prefix -- changes the prefix on every turn, so every turn
is a cache write: 10 writes, 0 reads, **$1.3350** against a $1.0800 baseline.
The write premium is +25.0% and it is paid ten times for nothing.

**MECHANISM: the two rates are 1.25x and 0.10x base.** A cached prefix pays for
itself after a single read -- 1.25 + 0.10 < 2.00 -- so the break-even is two
requests against the same prefix inside the TTL. Everything in this exercise
follows from those two multipliers.

**FINDING: `ProviderStats.misses` is dead.** None of the four simulators ever
increments it; a miss is counted as a `write`, and `print_report` prints
`misses 0` on every line it has ever printed. The field exists so that a
reader can tell writes from misses, and it cannot.

Structure: `SYSTEM_TOKENS` and `GROWTH` size the conversation, `static_turns`
puts the breakpoint after the system prompt, `growing_turns` puts it after the
transcript, and `run` prices one layout against its own no-cache baseline.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "15-prompt-caching"
MODEL = "anthropic_claude_opus_4_7"
SYSTEM_TOKENS, GROWTH, TURNS = 5000, 400, 10
TTL, GAP = 300, 4


def static_turns(ref):
    """One breakpoint after the system prompt: the prefix never changes."""
    return [ref.Request(prefix_tokens=SYSTEM_TOKENS, suffix_tokens=GROWTH * (turn + 1),
                        prefix_key="system") for turn in range(TURNS)]


def growing_turns(ref):
    """One breakpoint after the transcript: the prefix changes every turn."""
    return [ref.Request(prefix_tokens=SYSTEM_TOKENS + GROWTH * turn, suffix_tokens=GROWTH,
                        prefix_key=f"turn_{turn}") for turn in range(TURNS)]


def run(ref, turns):
    stats = ref.simulate_anthropic(turns, ttl_seconds=TTL, seconds_between=GAP)
    baseline = ref.baseline_cost(turns, MODEL)
    return {"writes": stats.writes, "reads": stats.reads, "misses": stats.misses,
            "cached": round(stats.total_cost, 4), "baseline": round(baseline, 4),
            "delta_pct": round((stats.total_cost / baseline - 1) * 100, 1),
            "tokens": sum(r.prefix_tokens + r.suffix_tokens for r in turns)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    price = ref.PRICES[MODEL]
    return {"turns": TURNS, "system_tokens": SYSTEM_TOKENS,
            "static": run(ref, static_turns(ref)), "growing": run(ref, growing_turns(ref)),
            "write_multiple": round(price["cache_write_5m"] / price["base"], 4),
            "read_multiple": round(price["cache_read"] / price["base"], 4),
            "stats_fields": [f for f in ref.ProviderStats.__dataclass_fields__]}


def verify(result):
    static, growing = result["static"], result["growing"]
    return [
        practice.Check(
            "ANSWER: 72,000 input tokens either way",
            all([static["tokens"] == 72000, static["tokens"] == growing["tokens"]]),
            f"{result['turns']} turns on a {result['system_tokens']}-token system prompt bill "
            f"{static['tokens']:,} input tokens with the breakpoint after the system prompt "
            f"and {growing['tokens']:,} with it after the transcript. Caching changes the "
            "price of a token, never the count -- a report in tokens shows no effect",
        ),
        practice.Check(
            "ANSWER: $1.0800 without cache_control and $0.4912 with, 54.5% saved",
            all([static["baseline"] == 1.08, static["cached"] == 0.4912,
                 static["writes"] == 1, static["reads"] == result["turns"] - 1]),
            f"the static layout writes the prefix {static['writes']} time and reads it "
            f"{static['reads']} times: ${static['cached']} against ${static['baseline']}, "
            f"{-static['delta_pct']}% saved",
        ),
        practice.Check(
            "FINDING: move the breakpoint with the conversation and it costs 23.6% more",
            all([growing["writes"] == result["turns"], growing["reads"] == 0,
                 growing["delta_pct"] > 20]),
            f"caching the whole transcript maximises the cached prefix and changes it every "
            f"turn, so all {growing['writes']} turns are writes and {growing['reads']} are "
            f"reads: ${growing['cached']} against ${growing['baseline']}, "
            f"{growing['delta_pct']:+}%. The write premium is paid ten times for nothing",
        ),
        practice.Check(
            "MECHANISM: the two rates are 1.25x and 0.10x base",
            all([result["write_multiple"] == 1.25, result["read_multiple"] == 0.1]),
            f"a cached prefix costs {result['write_multiple']}x base to write and "
            f"{result['read_multiple']}x to read, so "
            f"{result['write_multiple']} + {result['read_multiple']} < 2 and the break-even "
            "is two requests against the same prefix inside the TTL. Everything else here "
            "follows from those two multipliers",
        ),
        practice.Check(
            "FINDING: ProviderStats.misses is dead",
            all(["misses" in result["stats_fields"], static["misses"] == 0,
                 growing["misses"] == 0]),
            f"`misses` is declared -- the fields are {result['stats_fields']} -- and none of "
            "the four simulators ever increments it. A miss is counted as a write, so "
            "print_report has printed `misses 0` on every line it has ever printed, and the "
            "field that exists to tell writes from misses cannot",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
