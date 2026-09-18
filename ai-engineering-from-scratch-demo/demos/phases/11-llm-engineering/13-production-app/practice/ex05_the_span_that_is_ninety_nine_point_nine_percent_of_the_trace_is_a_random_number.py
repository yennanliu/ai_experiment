"""Exercise 5 — the span that is 99.98% of the trace is a random number.

    **Add OpenTelemetry tracing.** Instrument every component (cache lookup,
    guardrail check, LLM call, cost calculation) as a separate span. Each span
    records its duration. Export traces to the console. Show the full trace for
    a single request, with each component's contribution to total latency
    visible.

Reading of the exercise: the four named components are timed as written, each
over 2,000 repetitions so the number is stable, and the LLM span is measured as
the simulated time it asks for rather than the time it is served -- because
`asyncio.sleep` is what the span would be recording in production and skipping
it is the only way to keep this exercise at T0. The trace is then shown for one
cache miss and one cache hit.

**ANSWER: the trace for one request is `llm_call` 130.2 ms and everything else
under a fortieth of a millisecond.** `input_guardrail` 0.003 ms,
`cache_lookup` 0.012 ms, `output_guardrail` 0.017 ms, `cost_calculation`
0.0003 ms -- 0.032 ms for all four together. The LLM span is **99.98%** of the
trace, which is the answer to "each component's contribution" and also the
reason the question is not worth asking of this service.

**MECHANISM: the LLM span's duration is `random.uniform(0.1, 0.3)`.** The span
does not measure a model, a network or a queue; it measures the simulator. Over
200 seeded calls it has a median of 207.0 ms, a mean of 417.3 ms, a 95th
percentile of 1781.9 ms and a maximum of 4569.7 ms, and **28 of the 200** take
over a second -- because `call_llm_with_retry` fails 15% of the time on the
first attempt and then sleeps `min(2 ** attempt + uniform(0, 1), 10)`. A trace
built on this shows a retry tail that belongs to a coin flip.

**FINDING: the guardrail span runs before the cache span, so a cache hit pays
for it.** `handle_request` calls `check_input_guardrails` first and consults the
cache afterwards, so the cheapest path in the service still does a pass over
seven injection patterns and four PII patterns. It is 0.003 ms, so this costs
nothing here -- but the ordering is what a trace is for, and the trace makes it
visible: the hit's `latency_ms` is 0.03 ms against the miss's 0.4 ms.

**FINDING: `latency_ms` stops before the streaming starts.**
`handle_streaming_request` calls `handle_request`, which stamps `latency_ms`,
and *then* streams the response one word at a time at `uniform(0.02, 0.08)`
each. For a 61-token response that is 2.87 s of simulated time -- 99.996% of
the request -- and it appears in neither `latency_ms` (0.05 ms) nor
`eval_results`. The number the exercise asks to decompose excludes the largest
component of a streamed request.

Structure: `run` drives one coroutine with the lesson's `asyncio.sleep`
recording its delays instead of serving them and its `random` pinned to a
seed, `span` times one component over 2,000 repetitions, `trace` assembles the
four named spans for one request, and `tail` measures the retry distribution
over 200 calls.
"""

from __future__ import annotations

import asyncio
import statistics
import time

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "13-production-app"
QUERY = "how does the pipeline handle retries"
REPEATS, CALLS = 2000, 200


def run(ref, make, seed=7):
    slept, real, state = [], ref.asyncio.sleep, ref.random.getstate()

    async def instant(seconds, *_, **__):
        slept.append(seconds)

    ref.asyncio.sleep = instant
    ref.random.seed(seed)
    try:
        return asyncio.run(make()), slept
    finally:
        ref.asyncio.sleep, _ = real, ref.random.setstate(state)


def span(work):
    """One component's duration in milliseconds, averaged over REPEATS."""
    start = time.perf_counter()
    for _ in range(REPEATS):
        work()
    return round((time.perf_counter() - start) / REPEATS * 1000, 5)


def trace(ref):
    svc = ref.ProductionLLMService()
    miss, slept = run(ref, lambda: svc.handle_request("u", QUERY))
    hit, _ = run(ref, lambda: svc.handle_request("u", QUERY))
    answer = ref.SIMULATED_RESPONSES["general"]
    spans = {"input_guardrail": span(lambda: ref.check_input_guardrails(QUERY)),
             "cache_lookup": span(lambda: svc.cache.get(QUERY)),
             "output_guardrail": span(lambda: ref.check_output_guardrails(answer)),
             "cost_calculation": span(lambda: ref.calculate_cost(ref.ModelName.GPT_4O, 28, 81))}
    llm_ms = round(sum(slept) * 1000, 1)
    return {"spans": spans, "span_total": round(sum(spans.values()), 5), "llm_ms": llm_ms,
            "llm_share": round(llm_ms / (llm_ms + sum(spans.values())) * 100, 2),
            "miss_latency": miss["latency_ms"], "hit_latency": hit["latency_ms"],
            "hit": hit.get("cache_hit")}


def tail(ref):
    """The LLM span over 200 seeded calls: it is a uniform draw plus a retry tail."""
    draws = []
    for index in range(CALLS):
        _, slept = run(ref, lambda: ref.call_with_fallback("prompt", ref.ModelName.GPT_4O),
                       seed=1000 + index)
        draws.append(sum(slept) * 1000)
    return {"calls": CALLS, "median": round(statistics.median(draws), 1),
            "mean": round(statistics.mean(draws), 1),
            "p95": round(sorted(draws)[int(CALLS * 0.95) - 1], 1),
            "max": round(max(draws), 1), "over_a_second": sum(1 for d in draws if d > 1000)}


def streaming(ref):
    svc = ref.ProductionLLMService()
    result, slept = run(ref, lambda: svc.handle_streaming_request("u", "tell me about ML"))
    streamed = round(sum(slept[1:]), 2)
    return {"tokens": result["stream_tokens"], "streamed_s": streamed,
            "stream_call_ms": round(slept[0] * 1000, 1),
            "logged_ms": result["latency_ms"], "eval_ms": svc.eval_results[0]["latency_ms"],
            "hidden_share": round(streamed * 1000 / (streamed * 1000 + result["latency_ms"])
                                  * 100, 3)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "production_app")
    return {"repeats": REPEATS, **trace(ref), **tail(ref), **streaming(ref)}


def verify(result):
    spans = result["spans"]
    return [
        practice.Check(
            "ANSWER: llm_call is 130.2 ms and the other four spans are 0.032 ms together",
            all([result["llm_ms"] == 130.2, result["span_total"] < 0.1,
                 result["llm_share"] > 99.9]),
            f"timed over {result['repeats']} repetitions each, the four named components are "
            f"{spans} -- {result['span_total']} ms together -- against {result['llm_ms']} ms "
            f"of asyncio.sleep for the model call. The LLM span is {result['llm_share']}% of "
            "the trace, which answers the exercise and retires the question",
        ),
        practice.Check(
            "MECHANISM: the LLM span's duration is random.uniform(0.1, 0.3)",
            all([result["median"] == 207.0, result["over_a_second"] == 28,
                 result["max"] > 4000]),
            f"over {result['calls']} seeded calls the span has median {result['median']} ms, "
            f"mean {result['mean']}, p95 {result['p95']} and maximum {result['max']}, and "
            f"{result['over_a_second']} of {result['calls']} take over a second -- "
            "call_llm_with_retry fails 15% of the time then sleeps 2**attempt + uniform(0,1)",
        ),
        practice.Check(
            "FINDING: the guardrail span runs before the cache span, so a hit pays for it",
            all([result["hit"], spans["input_guardrail"] < spans["cache_lookup"],
                 result["hit_latency"] < result["miss_latency"]]),
            "`handle_request` calls check_input_guardrails first and consults the cache "
            f"afterwards, so the cheapest path still scans 7 injection and 4 PII patterns at "
            f"{spans['input_guardrail']} ms. The hit's latency_ms is {result['hit_latency']} "
            f"against the miss's {result['miss_latency']} -- the ordering is what a trace "
            "makes visible",
        ),
        practice.Check(
            "FINDING: latency_ms stops before the streaming starts",
            all([result["tokens"] == 61, result["streamed_s"] > 2.5,
                 result["hidden_share"] > 99.9,
                 result["eval_ms"] == result["logged_ms"]]),
            f"handle_streaming_request stamps latency_ms in handle_request and then streams "
            f"{result['tokens']} words at uniform(0.02, 0.08) each -- "
            f"{result['streamed_s']} s, {result['hidden_share']}% of the request -- while "
            f"latency_ms says {result['logged_ms']} ms and eval_results agrees with it",
        ),
        practice.Check(
            "MEASUREMENT: the trace's real work is four orders of magnitude below its total",
            all([result["llm_ms"] / result["span_total"] > 1000,
                 max(spans.values()) < 0.05, result["stream_call_ms"] == result["llm_ms"]]),
            f"{result['span_total']} ms of measurable work against {result['llm_ms']} ms of "
            f"simulated waiting, a ratio of "
            f"{result['llm_ms'] / result['span_total']:.0f}. The most expensive real span is "
            f"{max(spans, key=spans.get)} at {max(spans.values())} ms, and the streamed "
            "request's model call draws the identical sleep, seed for seed",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
