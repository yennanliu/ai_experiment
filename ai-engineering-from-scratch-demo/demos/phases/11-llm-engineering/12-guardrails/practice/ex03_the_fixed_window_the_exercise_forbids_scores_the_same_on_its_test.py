"""Exercise 3 — the fixed window the exercise forbids scores the same on its test.

    **Add rate limiting with sliding window.** Implement a per-user rate limiter
    that allows 10 requests per minute using a sliding window (not fixed
    window). Track the timestamp of each request. Block requests that exceed
    the limit and return a retry-after header. Test with a burst of 15 requests
    in 30 seconds.

Reading of the exercise: the clock is injected rather than slept on, so the
burst is 15 timestamps two seconds apart. "Not fixed window" is a claim about
the difference between the two, so the fixed window is implemented too and
scored on the same burst -- that is the only way to know the test tests it.

**ANSWER: 10 allowed, 5 blocked, retry-after 40, 38, 36, 34, 32 seconds.** The
sliding window keeps the timestamps, drops those older than 60 seconds, and
prices the wait as `60 - (now - oldest)`. The exercise's stated behaviour is
exactly what it does.

**FINDING: the fixed window returns the identical answer on the identical
test.** Same 10 allowed, same 5 blocked, and the same five retry-after values
to the tenth of a second -- because a burst that starts at t=0 and ends at t=28
never crosses a minute boundary, and a boundary is the only place the two
disagree. The test names the distinction it cannot measure.

**CONTROL: move the same burst 45 seconds later and the fixed window lets all
15 through.** Ten land in the window ending at t=60 and five in the next one,
so 15 requests arrive inside 28 seconds against a 10-per-minute limit. Worse,
10 timestamps at t=59.9 and 10 at t=60.0 give 20 allowed inside 0.1 seconds --
twice the limit, the textbook failure. The sliding window allows 10 in both.

**FINDING: the pipeline has nowhere to put a per-user limiter.**
`GuardrailPipeline.process(self, user_input, model_fn=None)` and
`validate_input(self, user_input)` take no user id, and `_log_event` records a
sha256 of the input with no user field. "Per-user" cannot be added as another
entry in `validate_input`; it needs a new signature.

**FINDING: "return a retry-after header" has no field to return it in.**
`GuardrailResult` is `(passed, category, details, confidence, latency_ms)` and
`GuardrailReport` is `(input_results, output_results, blocked, block_reason,
total_latency_ms)`. Neither has a header, a status code or a number, so the
wait rides inside the `details` string and the caller parses prose to find it.

Structure: `BURST` and `CROSSING` are the two 15-request bursts, `sliding` and
`fixed` are the two limiters over an injected clock, `as_check` wraps a verdict
as the lesson's own `GuardrailResult`, and `retry_after` recovers the number a
caller would have to parse back out of `details`.
"""

from __future__ import annotations

import dataclasses
import inspect
import re

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "12-guardrails"
LIMIT, WINDOW = 10, 60.0
BURST = [i * 2.0 for i in range(15)]
CROSSING = [45.0 + i * 2.0 for i in range(15)]
EDGE = [59.9] * 10 + [60.0] * 10


def sliding(times, limit=LIMIT):
    """Keep the timestamps, drop those older than one window, price the wait."""
    history, out = [], []
    for now in times:
        history = [t for t in history if now - t < WINDOW]
        if len(history) < limit:
            history.append(now)
            out.append((True, 0.0))
        else:
            out.append((False, round(WINDOW - (now - history[0]), 1)))
    return out


def fixed(times, limit=LIMIT):
    """The one the exercise says not to build: a counter per calendar minute."""
    counts, out = {}, []
    for now in times:
        bucket = int(now // WINDOW)
        if counts.get(bucket, 0) < limit:
            counts[bucket] = counts.get(bucket, 0) + 1
            out.append((True, 0.0))
        else:
            out.append((False, round((bucket + 1) * WINDOW - now, 1)))
    return out


def as_check(ref, allowed, wait):
    return ref.GuardrailResult(passed=allowed, category="rate_limit",
                               details=f"retry-after={wait}s", confidence=0.0 if allowed
                               else 1.0, latency_ms=0.0)


def retry_after(result):
    found = re.search(r"retry-after=([\d.]+)s", result.details)
    return float(found.group(1)) if found else None


def allowed(verdicts):
    return sum(1 for ok, _ in verdicts if ok)


def waits(verdicts):
    return [wait for ok, wait in verdicts if not ok]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "guardrails")
    slid, fix = sliding(BURST), fixed(BURST)
    blocked = as_check(ref, *slid[LIMIT])
    return {
        "burst": len(BURST), "seconds": BURST[-1], "limit": LIMIT,
        "slide_allowed": allowed(slid), "slide_waits": waits(slid),
        "fixed_allowed": allowed(fix), "fixed_waits": waits(fix),
        "cross_slide": allowed(sliding(CROSSING)), "cross_fixed": allowed(fixed(CROSSING)),
        "edge_slide": allowed(sliding(EDGE)), "edge_fixed": allowed(fixed(EDGE)),
        "edge_span": round(EDGE[-1] - EDGE[0], 1),
        "process_args": list(inspect.signature(ref.GuardrailPipeline.process).parameters),
        "validate_args": [p for p in
                          inspect.signature(ref.GuardrailPipeline.validate_input).parameters],
        "log_fields": sorted(ref.GuardrailPipeline()._log_event.__code__.co_names),
        "result_fields": [f.name for f in dataclasses.fields(ref.GuardrailResult)],
        "report_fields": [f.name for f in dataclasses.fields(ref.GuardrailReport)],
        "recovered": retry_after(blocked), "details": blocked.details,
    }


def verify(result):
    fields = result["result_fields"] + result["report_fields"]
    return [
        practice.Check(
            "ANSWER: 10 allowed, 5 blocked, retry-after 40, 38, 36, 34, 32 seconds",
            all([result["slide_allowed"] == LIMIT,
                 result["slide_waits"] == [40.0, 38.0, 36.0, 34.0, 32.0]]),
            f"{result['burst']} requests over {result['seconds']:.0f} seconds against "
            f"{result['limit']}/minute: {result['slide_allowed']} allowed, "
            f"{result['burst'] - result['slide_allowed']} blocked, and the waits are "
            f"{result['slide_waits']} -- 60 - (now - oldest) as the oldest ages out",
        ),
        practice.Check(
            "FINDING: the fixed window returns the identical answer on the identical test",
            all([result["fixed_allowed"] == result["slide_allowed"],
                 result["fixed_waits"] == result["slide_waits"]]),
            f"the forbidden implementation allows {result['fixed_allowed']} and waits "
            f"{result['fixed_waits']} -- the same numbers to the tenth of a second, because "
            "a burst from t=0 to t=28 never crosses a minute boundary and a boundary is the "
            "only place the two disagree. The test names a distinction it cannot measure",
        ),
        practice.Check(
            "CONTROL: 45 seconds later the fixed window lets all 15 through, and 20 at a"
            " boundary",
            all([result["cross_slide"] == LIMIT, result["cross_fixed"] == result["burst"],
                 result["edge_slide"] == LIMIT, result["edge_fixed"] == 2 * LIMIT]),
            f"the same burst started at t=45 gives sliding {result['cross_slide']} and fixed "
            f"{result['cross_fixed']} of {result['burst']}; ten timestamps at 59.9 and ten "
            f"at 60.0 give sliding {result['edge_slide']} and fixed {result['edge_fixed']} "
            f"inside {result['edge_span']} seconds -- twice the limit",
        ),
        practice.Check(
            "FINDING: the pipeline has nowhere to put a per-user limiter",
            all(["user_id" not in result["process_args"],
                 "user_id" not in result["validate_args"],
                 "user_id" not in result["log_fields"]]),
            f"`process{tuple(result['process_args'])}` and "
            f"`validate_input{tuple(result['validate_args'])}` take no user id, and "
            "`_log_event` stores a sha256 of the input with no user field. 'Per-user' is not "
            "another entry in validate_input; it needs a new signature",
        ),
        practice.Check(
            "FINDING: 'return a retry-after header' has no field to return it in",
            all([not [f for f in fields if "header" in f or "retry" in f],
                 result["recovered"] == result["slide_waits"][0]]),
            f"GuardrailResult is {result['result_fields']} and GuardrailReport is "
            f"{result['report_fields']} -- no header, status code or number. The wait rides "
            f"in details as {result['details']!r}, and recovering "
            f"{result['recovered']} takes a regex over prose",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
