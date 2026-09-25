"""Exercise 1 — the breaker opens once and never closes, so it caps the storm by switching the service off.

    Run `code/main.py`. Confirm the circuit breaker caps the retry storm. Vary
    the failure threshold and observe the tradeoff.

Reading of the exercise: "caps" is checked in downstream calls, and "the
tradeoff" in what those calls buy -- successes -- so the threshold sweep
reports both, and each number is then traced to the mechanism that produced it.

**ANSWER: it caps calls, 328 -> 140, and costs successes, 189 -> 113.** On
200 requests the breaker short-circuits 87 of them. Sweeping the threshold,
calls and successes fall together: 26/25 at 0.05, 41/38 at 0.1, 110/97 at
0.2, 111/97 at 0.3, 140/113 at 0.5, 280/176 at 0.7, and 328/189 -- the unprotected
run -- at 0.9 and 1.0. Every threshold that sheds load also sheds successes.

**FINDING: the breaker opens once and never closes.** `open_cooldown_s` is
0.5 *wall-clock* seconds and the whole simulation runs in well under a
millisecond, so `allow()` never reaches HALF_OPEN: one state transition in the
run, at call 140, and every request after it is short-circuited. The "cap" is
the service switched off for the rest of the run.

**FINDING: the shipped service can never recover, so backing off cannot
help it.** `service.load = min(total_calls // 10, 50)` counts every call ever
made; it never decays. The unprotected run ends failing 74% of calls, and at
1000 requests a breaker given a per-request clock so it *can* half-open still
lands 204 successes against 241 unprotected. Make load a recent rate -- calls
made over the last 20 requests -- and the storm is real: 3870 calls and 71
successes unprotected, 176 with the breaker, 2.5x. Over the last 5 requests
there is no storm and the breaker costs 994 -> 843.

**FINDING: the storm is 1.64x, not 10x, and cannot reach 10x.** 328 calls for
200 requests; with at most 4 attempts per request the ceiling is 4x. And the
lesson's Build It names `FailureTaxonomy`, `RetryStormSimulator` and
`DetectionAgent`; of the four classes it lists only `CircuitBreaker` exists.

Structure: `storm()` reruns the shipped loop on the reference's own
CircuitBreaker and DownstreamService with a virtual clock and a choice of
load model -- `lifetime` is the shipped one.
"""

from __future__ import annotations

import collections
import random
import time
import types

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "23-failure-modes-mast-groupthink"
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)
NAMED = ("FailureTaxonomy", "CircuitBreaker", "RetryStormSimulator", "DetectionAgent")


def load_ref():
    return parity.load_reference(PHASE, LESSON, "main")


def serve(svc, br, rng, load, breaker):
    """One request's attempts: (calls made, succeeded, short-circuited)."""
    for made in range(4):
        if breaker and not br.allow():
            return made, False, True
        svc.load = load(made)
        ok = svc.handle(rng)
        br.record(ok) if breaker else None
        if ok:
            return made + 1, True, False
    return 4, False, False


def storm(ref, requests, breaker, window=None, seed=0, tick=0.01):
    """(calls, successes, short-circuits). window=None is the shipped lifetime load."""
    clock = [0.0]
    ref.time = types.SimpleNamespace(monotonic=lambda: clock[0])
    rng, svc, br = random.Random(seed), ref.DownstreamService(), ref.CircuitBreaker()
    recent, totals = collections.deque(maxlen=window or 1), [0, 0, 0]
    for _ in range(requests):
        clock[0] += tick
        base = totals[0]
        load = ((lambda m: min((base + m) // 10, 50)) if window is None
                else (lambda m: sum(recent) + m))
        made, ok, cut = serve(svc, br, rng, load, breaker)
        totals = [totals[0] + made, totals[1] + ok, totals[2] + cut]
        recent.append(made)
    ref.time = time
    return tuple(totals)


def transitions(ref):
    """State changes of the shipped breaker over the shipped 200-request run."""
    seen, original = [], ref.CircuitBreaker.record

    def record(self, success):
        before = self.state
        original(self, success)
        if self.state != before:
            seen.append((before.value, self.state.value, len(self.outcomes)))
    ref.CircuitBreaker.record = record
    try:
        start = time.perf_counter()
        ref.simulate_retry_storm(200, use_breaker=True, seed=0)
        return seen, time.perf_counter() - start
    finally:
        ref.CircuitBreaker.record = original


def sweep(ref):
    original, out = ref.CircuitBreaker, {}
    for threshold in THRESHOLDS:
        ref.CircuitBreaker = lambda t=threshold: original(failure_threshold=t)
        out[threshold] = ref.simulate_retry_storm(200, use_breaker=True, seed=0)[:2]
    ref.CircuitBreaker = original
    return out


def solve():
    ref = load_ref()
    seen, elapsed = transitions(ref)
    return {
        "off": ref.simulate_retry_storm(200, use_breaker=False, seed=0),
        "on": ref.simulate_retry_storm(200, use_breaker=True, seed=0),
        "sweep": sweep(ref), "transitions": seen, "elapsed": elapsed,
        "cooldown": ref.CircuitBreaker().open_cooldown_s,
        "long": {"off": storm(ref, 1000, False, tick=0.1), "on": storm(ref, 1000, True, tick=0.1)},
        "recent": {w: (storm(ref, 1000, False, w), storm(ref, 1000, True, w)) for w in (5, 20)},
        "exists": [name for name in NAMED if hasattr(ref, name)],
    }


def verify(result):
    off, on, sweep_ = result["off"], result["on"], result["sweep"]
    calls = [sweep_[t][0] for t in THRESHOLDS]
    wins = [sweep_[t][1] for t in THRESHOLDS]
    end_rate = 0.1 + 0.02 * min(off[0] // 10, 50)
    storm20, calm5 = result["recent"][20], result["recent"][5]
    return [
        practice.Check(
            "ANSWER: it caps calls, 328 -> 140, and costs successes, 189 -> 113",
            all([off[:2] == (328, 189), on == (140, 113, 87), calls == sorted(calls),
                 wins == sorted(wins), sweep_[0.9] == sweep_[1.0] == off[:2]]),
            f"calls/successes by threshold {sweep_}; every threshold that sheds load "
            "sheds successes, and 0.9 and above equal no breaker",
        ),
        practice.Check(
            "FINDING: the breaker opens once and never closes",
            all([result["transitions"] == [("closed", "open", 140)],
                 result["elapsed"] < result["cooldown"], on[1] + on[2] == 200]),
            f"transitions {result['transitions']}; the run takes "
            f"{result['elapsed'] * 1e3:.2f}ms against a {result['cooldown']}s wall-clock "
            f"cooldown, so all {on[2]} requests after it opens are short-circuited",
        ),
        practice.Check(
            "FINDING: the shipped service can never recover, so backing off cannot help it",
            all([round(end_rate, 2) == 0.74,
                 result["long"]["on"][1] < result["long"]["off"][1],
                 storm20[1][1] > 2 * storm20[0][1], calm5[1][1] < calm5[0][1]]),
            f"load counts every call ever made, ending at a {end_rate:.0%} failure rate; "
            f"at 1000 requests a recovering breaker gets {result['long']['on'][1]} "
            f"successes against {result['long']['off'][1]}; with load over the last 20 "
            f"requests the storm is {storm20[0][:2]} and the breaker {storm20[1][:2]}, "
            f"over the last 5 {calm5[0][1]} -> {calm5[1][1]}",
        ),
        practice.Check(
            "FINDING: the storm is 1.64x, not 10x, and cannot reach 10x",
            off[0] / 200 == 1.64 and result["exists"] == ["CircuitBreaker"],
            f"{off[0]} calls for 200 requests, against a 4-attempt ceiling of 4x; of "
            f"{list(NAMED)} only {result['exists']} exists in the module",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
