"""Exercise 5 — the degradation removes the event the exercise asks you to verify.

    **Implement a circuit breaker with degradation levels.** At 70% budget, log
    a warning. At 85%, automatically switch all routing to the cheapest model
    (gpt-4o-mini). At 95%, serve only cached responses and reject new queries.
    Test by simulating 1,000 requests against a $1.00 budget and verify each
    threshold triggers correctly.

Reading of the exercise: the breaker is built exactly as specified and run over
the lesson's own 10-query pipeline mix, cycled to 1,000 requests, against a
`CostTracker(monthly_budget=1.00)`. The same 1,000 requests are also run with
the breaker switched off, because "verify each threshold triggers" is a claim
about the run *without* the mitigation.

**ANSWER: all three thresholds trigger -- but only with the breaker off.** The
unmitigated run reaches $0.9516 of the $1.00 budget and records `warning` at
request 739, `throttle` at 895 and `stop` at 999. Switch the breaker on and the
85% degradation does its job: the bill lands at $0.8700, the run ends at 87% of
budget, and the `stop` level never fires. The exercise asks for a mitigation
and then asks to verify the event the mitigation prevents.

**FINDING: the 95% level has one request of headroom.** `stop` arrives at
request 999 of 1,000. The test is not a demonstration that the threshold works;
it is a coincidence of the query mix, and one cheaper query anywhere in the
cycle removes it.

**FINDING: the 85% switch touches 2 of the 10 queries.** Eight of them already
route to `gpt-4o-mini` at the "pro" tier, so "switch all routing to the cheapest
model" only moves the two `medium` queries off `claude-sonnet-4` -- worth 8.6%
of the bill.

**FINDING: `gpt-4o` is never selected, because "hi" hides inside
"architecture".** "Analyze the pros and cons of serverless architecture"
classifies `simple`: `classify_complexity` scans `SIMPLE_KEYWORDS` for
substrings before it reaches `COMPLEX_KEYWORDS`. The most expensive model in the
pro row is unreachable for this workload, so the breaker's whole purpose is
already served by a bug.

**FINDING: `_check_budget` uses `elif`, so a jump past several levels records
only the highest.** One `log_call("gpt-4o", 500000, 200000)` against the $1.00
budget totals $3.25 and records `['stop']` alone. A breaker that reads its level
off `tracker.alerts` never sees `warning` or `throttle` on any call large enough
to matter. The clean run only walks the levels in order because every request is
small.

**FINDING: the 95% "serve only cached" mode cannot be accounted for.** A call
logged with `cache_status="hit"` is still charged in full -- $0.0055 -- and
`cache_savings` reports the same $0.0055 as saved. And
`calculate_cost("gpt-4o", 100, 10, cached_input_tokens=500)` returns -0.000275,
because `non_cached = input_tokens - cached_input_tokens` goes negative. Logging
cache hits can lower `total_cost()` and un-trip the breaker that caused them.

Structure: `QUERIES` is the lesson's own pipeline mix, `run` cycles it through
the tracker with the breaker optional and records which request first raised
each alert, and `solve` runs it both ways.
"""

from __future__ import annotations

from collections import Counter

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "11-caching-cost"
BUDGET, CHEAPEST, TIER, REQUESTS = 1.00, "gpt-4o-mini", "pro", 1000
QUERIES = ["What is the return policy?", "How do I return something?",
           "What are your hours?", "When do you open?",
           "Explain the difference between TCP and UDP", "Compare TCP vs UDP protocols",
           "Hello", "What is your phone number?",
           "Write a Python function to sort a list",
           "Analyze the pros and cons of serverless architecture"]


def run(ref, requests, breaker=False):
    tracker, served, first = ref.CostTracker(monthly_budget=BUDGET), Counter(), {}
    for index in range(requests):
        query = QUERIES[index % len(QUERIES)]
        pct = tracker.total_cost() / BUDGET
        if breaker and pct >= 0.95:
            served["rejected"] += 1
            continue
        model = CHEAPEST if (breaker and pct >= 0.85) else ref.route_model(query, TIER)["model"]
        call = ref.simulate_llm_call(model, query)
        tracker.log_call(model, call["input_tokens"], call["output_tokens"],
                         latency_ms=call["latency_ms"])
        served[model] += 1
        for alert in tracker.alerts:
            first.setdefault(alert["level"], index + 1)
    return tracker, served, first


def jump(ref):
    """One call that clears every threshold at once, and one logged as a cache hit."""
    over = ref.CostTracker(monthly_budget=BUDGET)
    over.log_call("gpt-4o", 500_000, 200_000)
    hit = ref.CostTracker(monthly_budget=BUDGET)
    hit.log_call("gpt-4o", 1000, 300, cache_status="hit")
    return {"jump_total": over.total_cost(),
            "jump_alerts": [a["level"] for a in over.alerts],
            "hit_charged": hit.total_cost(), "hit_saved": hit.cache_savings()["saved"],
            "over_cached": ref.calculate_cost("gpt-4o", 100, 10,
                                              cached_input_tokens=500)["total_cost"]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "caching_cost")
    plain, plain_models, plain_at = run(ref, REQUESTS)
    broken, broken_models, broken_at = run(ref, REQUESTS, breaker=True)
    return {
        "requests": REQUESTS, "budget": BUDGET,
        "plain_total": plain.total_cost(), "plain_at": plain_at,
        "plain_alerts": [a["level"] for a in plain.alerts],
        "plain_models": dict(plain_models),
        "breaker_total": broken.total_cost(),
        "breaker_utilization": broken.summary()["budget_utilization"],
        "breaker_alerts": [a["level"] for a in broken.alerts],
        "breaker_models": dict(broken_models),
        "levels": dict(Counter(ref.classify_complexity(q) for q in QUERIES)),
        "routed": dict(Counter(ref.route_model(q, TIER)["model"] for q in QUERIES)),
        "architecture": ref.classify_complexity(QUERIES[-1]),
        **jump(ref),
    }


def verify(result):
    plain_at, saved = result["plain_at"], 1 - result["breaker_total"] / result["plain_total"]
    return [
        practice.Check(
            "ANSWER: all three thresholds trigger, and only with the breaker off",
            all([result["plain_alerts"] == ["warning", "throttle", "stop"],
                 result["breaker_alerts"] == ["warning", "throttle"]]),
            f"{result['requests']} requests against the ${result['budget']:.2f} budget reach "
            f"${result['plain_total']} unmitigated -- warning at request "
            f"{plain_at['warning']}, throttle at {plain_at['throttle']}, stop at "
            f"{plain_at['stop']}. With the breaker on the 85% switch works, the bill lands at "
            f"${result['breaker_total']} ({result['breaker_utilization']:.0%}) and stop never "
            "fires: the mitigation removes the event the exercise says to verify",
        ),
        practice.Check(
            "FINDING: the 95% level arrives with one request of headroom",
            plain_at["stop"] == result["requests"] - 1,
            f"stop is recorded at request {plain_at['stop']} of {result['requests']}. The "
            "test does not show the threshold works, it shows this query cycle happens to "
            "end just past 95% -- one cheaper query anywhere in it and the level is untested",
        ),
        practice.Check(
            "FINDING: 'switch all routing to the cheapest model' moves 2 of 10 queries",
            all([result["routed"][CHEAPEST] == 8, len(result["routed"]) == 2]),
            f"{result['routed'][CHEAPEST]} of the {len(QUERIES)} queries already route to "
            f"{CHEAPEST} at the '{TIER}' tier, so the 85% degradation only moves the "
            f"{result['levels']['medium']} medium queries off claude-sonnet-4 -- {saved:.1%} "
            f"of the bill. Routed models across the mix: {sorted(result['routed'])}",
        ),
        practice.Check(
            "FINDING: gpt-4o is never selected, because 'hi' hides inside 'architecture'",
            all([result["architecture"] == "simple", "gpt-4o" not in result["routed"],
                 "complex" not in result["levels"]]),
            f"'{QUERIES[-1]}' classifies {result['architecture']!r}: classify_complexity "
            f"scans SIMPLE_KEYWORDS for substrings first. {result['levels']} -- no query in "
            "the lesson's own mix is complex, so the most expensive model in the pro row is "
            "unreachable and the breaker's purpose is already served by a bug",
        ),
        practice.Check(
            "FINDING: elif hides levels, and a logged cache hit can un-trip the breaker",
            all([result["jump_alerts"] == ["stop"], result["over_cached"] < 0,
                 result["hit_charged"] == result["hit_saved"]]),
            f"one log_call('gpt-4o', 500000, 200000) totals ${result['jump_total']} and "
            f"records {result['jump_alerts']} alone, because _check_budget uses elif. A call "
            f"logged cache_status='hit' is charged ${result['hit_charged']} and reported as "
            f"${result['hit_saved']} saved. And calculate_cost with 500 cached tokens against "
            f"100 input tokens returns ${result['over_cached']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
