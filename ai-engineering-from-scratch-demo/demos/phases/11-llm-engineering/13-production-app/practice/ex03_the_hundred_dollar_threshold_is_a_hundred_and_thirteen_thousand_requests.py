"""Exercise 3 — the $100 threshold is 113,637 requests away.

    **Build a cost alerting system.** Track cost per user per day. When a user
    exceeds $0.50/day, switch them to `gpt-4o-mini`. When total daily cost
    exceeds $100, activate emergency mode: cache-only responses for repeated
    queries, `gpt-4o-mini` for everything else, reject requests over 2,000
    input tokens. Test with a simulated traffic spike.

Reading of the exercise: both thresholds are priced against what the service
actually charges for one request, because "test with a simulated traffic spike"
is only a test if the spike can reach them. The per-user switch is then priced
too, and the emergency mode's three rules are each checked against the code
they would have to be added to.

**ANSWER: a request costs $0.00088, so $0.50 is 569 requests from one user and
$100 is 113,637 from everybody.** 28 input tokens and 81 output tokens on
`gpt-4o`. At the lesson's own 0.2 s of simulated latency per call, the traffic
spike the exercise asks for is **6.3 hours** long.

**ANSWER: switching that user to `gpt-4o-mini` cuts 94.0% per request**, from
$0.00088 to $0.0000528 -- and moves their next crossing of $0.50 from 569
requests to 9,470. The mitigation works; it is the threshold that is in the
wrong units.

**FINDING: `CostTracker` has no day in it.** Its seven fields are four totals,
`total_cache_hits`, `cost_by_user` and `cost_by_model`, all keyed by name
alone. "Cost per user per day" cannot be read from the tracker at all: the only
place the day survives is `RequestLog.timestamp`, so the alerting system has to
be built on the log and not on the thing named for cost.

**FINDING: the emergency model switch is not a switch away from the configured
primary.** `PRIMARY_MODEL` comes from `LLM_MODEL` and defaults to
`claude-sonnet-5`, but `PromptTemplate.model` defaults to `gpt-4o` and only
`code_review` overrides it -- so 1 of the 3 templates uses the configured
primary and all 100 requests here bill `gpt-4o`. Routing `general_chat` to the
primary instead would have *raised* the bill by 47.6%.

**FINDING: emergency mode's cache-only rule leaks answers between users.**
`check_input_guardrails` redacts PII before the cache is consulted and the
cache key is the redacted text, so "My SSN is 111-45-6789" and "My SSN is
222-45-6789" are the same key. Four users asking with four different SSNs
produce 1 cache entry and 3 hits at similarity 1.0, and users two to four get
user one's answer. "Cache-only responses for repeated queries" is the rule that
makes this the normal path.

**FINDING: there is no length check to extend.** `check_input_guardrails` tests
the injection patterns and the PII patterns and nothing else, and
`estimate_tokens` is `words * 4 // 3`, so "reject over 2,000 input tokens"
means rejecting at 1,501 words -- a guardrail that has to be written, not
configured.

Structure: `run` drives one coroutine with the lesson's `asyncio.sleep` skipped
and its `random` pinned to a seed, `burst` is the traffic the pricing is
measured on, `pricing` converts one request's tokens into the two thresholds,
and `leak` replays four users' PII-bearing queries through the real pipeline.
"""

from __future__ import annotations

import asyncio
import dataclasses
import math

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "13-production-app"
REQUESTS, USER_LIMIT, DAILY_LIMIT, TOKEN_LIMIT = 100, 0.50, 100.0, 2000
SLEEP_PER_CALL = 0.2
SSNS = (111, 222, 333, 444)


def run(ref, make, seed=7):
    real, state = ref.asyncio.sleep, ref.random.getstate()

    async def instant(*_, **__):
        return None

    ref.asyncio.sleep = instant
    ref.random.seed(seed)
    try:
        return asyncio.run(make())
    finally:
        ref.asyncio.sleep, _ = real, ref.random.setstate(state)


def burst(ref):
    """The lesson's own pipeline, 100 requests over 5 users."""
    svc = ref.ProductionLLMService()

    async def send():
        for i in range(REQUESTS):
            await svc.handle_request(f"u{i % 5}", f"question number {i} about the system")

    run(ref, send)
    return svc


def pricing(ref, log):
    """One request's tokens, priced under each model and turned into thresholds."""
    each = {m: ref.calculate_cost(m, log.input_tokens, log.output_tokens) for m in ref.ModelName}
    full, mini = each[ref.ModelName.GPT_4O], each[ref.ModelName.GPT_4O_MINI]
    return {"input_tokens": log.input_tokens, "output_tokens": log.output_tokens,
            "per_request": full, "per_request_mini": mini,
            "mini_saving_pct": round((1 - mini / full) * 100, 1),
            "primary_delta_pct": round((each[ref.PRIMARY_MODEL] / full - 1) * 100, 1),
            "to_user_limit": math.ceil(USER_LIMIT / full),
            "to_user_limit_mini": math.ceil(USER_LIMIT / mini),
            "to_daily_limit": math.ceil(DAILY_LIMIT / full),
            "spike_hours": round(math.ceil(DAILY_LIMIT / full) * SLEEP_PER_CALL / 3600, 1)}


def leak(ref):
    """Four users, four different SSNs, one cache entry."""
    svc = ref.ProductionLLMService()
    asks = [f"My SSN is {n:03d}-45-6789, can you help me?" for n in SSNS]

    async def send():
        return [await svc.handle_request(f"user{i}", q) for i, q in enumerate(asks)]

    answers = run(ref, send)
    return {"askers": len(asks), "entries": len(svc.cache.entries),
            "redacted": len({ref.check_input_guardrails(q).modified_text for q in asks}),
            "hits": sum(1 for a in answers if a.get("cache_hit")),
            "similarity": sorted({a["similarity"] for a in answers if a.get("cache_hit")}),
            "one_answer": len({a["response"] for a in answers}) == 1}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "production_app")
    svc = burst(ref)
    words = next(w for w in range(1, 4000) if ref.estimate_tokens(" ".join(["word"] * w))
                 > TOKEN_LIMIT)
    billed = svc.cost_tracker.summary()["cost_by_model"]
    return {"requests": REQUESTS, "primary": ref.PRIMARY_MODEL.value, "reject_at_words": words,
            "billed": {m: round(c, 6) for m, c in billed.items()},
            "tracker_fields": [f.name for f in dataclasses.fields(ref.CostTracker)],
            "log_fields": [f.name for f in dataclasses.fields(ref.RequestLog)],
            "template_models": {n: v["v1"].model.value for n, v in ref.PROMPT_TEMPLATES.items()},
            **pricing(ref, svc.request_logs[0]), **leak(ref)}


def verify(result):
    return [
        practice.Check(
            "ANSWER: $0.50 is 569 requests from one user and $100 is 113,637 from everybody",
            all([result["per_request"] == 0.00088, result["to_user_limit"] == 569,
                 result["to_daily_limit"] == 113637]),
            f"one request is {result['input_tokens']} input and {result['output_tokens']} "
            f"output tokens on gpt-4o, ${result['per_request']}. ${USER_LIMIT:.2f} is "
            f"{result['to_user_limit']} requests from one user and ${DAILY_LIMIT:.0f} is "
            f"{result['to_daily_limit']:,} from everybody -- {result['spike_hours']} hours "
            f"of the lesson's own {SLEEP_PER_CALL} s per call",
        ),
        practice.Check(
            "ANSWER: switching that user to gpt-4o-mini cuts 94.0% per request",
            all([result["mini_saving_pct"] == 94.0,
                 result["to_user_limit_mini"] == 9470]),
            f"${result['per_request']} becomes ${result['per_request_mini']:.7f}, "
            f"-{result['mini_saving_pct']}%, which moves that user's next crossing of "
            f"${USER_LIMIT:.2f} from {result['to_user_limit']} requests to "
            f"{result['to_user_limit_mini']:,}. The mitigation works; the threshold is in "
            "the wrong units",
        ),
        practice.Check(
            "FINDING: CostTracker has no day in it",
            all([not [f for f in result["tracker_fields"] if "da" in f or "time" in f],
                 "timestamp" in result["log_fields"]]),
            f"its fields are {result['tracker_fields']}, keyed by name alone. 'Cost per user "
            "per day' cannot be read from the tracker at all -- the only place the day "
            "survives is RequestLog.timestamp, so the alerting has to be built on the log "
            "and not on the object named for cost",
        ),
        practice.Check(
            "FINDING: the configured primary model is used by 1 of the 3 templates",
            all([result["billed"] == {"gpt-4o": 0.088}, result["primary"] == "claude-sonnet-5",
                 sum(1 for m in result["template_models"].values()
                     if m == result["primary"]) == 1]),
            f"PRIMARY_MODEL is {result['primary']} but PromptTemplate.model defaults to "
            f"gpt-4o, so the templates resolve to {result['template_models']} and all "
            f"{result['requests']} requests bill {result['billed']}. Routing general_chat to "
            f"the primary would have raised the bill {result['primary_delta_pct']:+}%",
        ),
        practice.Check(
            "FINDING: emergency mode's cache-only rule leaks answers between users",
            all([result["redacted"] == 1, result["entries"] == 1,
                 result["hits"] == result["askers"] - 1, result["similarity"] == [1.0],
                 result["one_answer"]]),
            f"{result['askers']} users with {result['askers']} different SSNs redact to "
            f"{result['redacted']} string, so the cache holds {result['entries']} entry and "
            f"serves {result['hits']} hits at similarity {result['similarity'][0]}: users "
            "two to four get user one's answer. And there is no length check to extend -- "
            f"rejecting over {TOKEN_LIMIT} tokens means rejecting at "
            f"{result['reject_at_words']:,} words, a guardrail that has to be written",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
