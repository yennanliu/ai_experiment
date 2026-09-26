"""Exercise 1 — at 5% the chain serves 99.9875%, and the gateway changes nothing but a constant.

    Run `code/main.py`. Configure fallback from OpenAI→Anthropic→self-hosted.
    What's the expected hit rate at 5% provider error rate?

Reading of the exercise: the shipped `PROVIDERS` list already is the
OpenAI -> Anthropic -> self-hosted chain, so "configure" means setting every
provider's error rate to 0.05. "Hit rate" is read both ways a gateway reports
it -- the share of requests served, and the share that hit a fallback -- and
each is computed exactly from the chain and then measured on the simulator.

**ANSWER: 99.9875% served, 5% of requests hit a fallback and 0.25% reach
self-hosted.** Independent 5% errors fail a request only when all three
fail, 0.05^3 = 0.000125, and cost 1.0525 calls per request. The simulator
agrees: at 200,000 requests it serves 0.9999 and makes 10,434 fallback calls,
against 25 failures and 10,500 fallback calls expected. At its own n = 1000 it
prints 100.0% and 60 fallbacks: a run that size expects 0.125 total failures,
so 1000 requests cannot show the number the exercise asks for.

**FINDING: the four gateways differ only by a constant.** On the shipped run
every gateway reports success 100.0%, 34 retries and 34 fallbacks, because each
replays seed 7, and mean latency is 183.048 ms plus the gateway's overhead
(10 / 30 / 5 / 2 ms). The gateway is not modelled: it cannot change who
is called, how often, or what fails.

**FINDING: every "retry" is a fallback, and there is no backoff.** The code
never calls the same provider twice and never waits, so the lesson's
"exponential backoff, bounded attempts" has no counterpart. `retries` counts
failed calls and `fallback_hits` counts calls after the first, so
retries - fallbacks is exactly the number of requests that failed every
provider: 10,454 - 10,434 = 20 = 200,000 x (1 - 0.9999).

**FINDING: a failed call costs 0.3 of the provider's latency, not "half".**
`call_provider` returns 54.0 ms for a failed 180 ms OpenAI call; its comment
says "half-done before error". And the lesson's "Six core features" heading
numbers seven items.

Structure: `exact()` walks the chain in closed form; `with_rate()` swaps in
error rates for one run and restores `PROVIDERS` afterwards.
"""

from __future__ import annotations

import dataclasses

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "19-ai-gateways"
RATE, BIG = 0.05, 200_000


class AlwaysFail:
    def random(self):
        return 0.0


def exact(providers, overhead=0.0):
    """(served, fallback calls per request, mean latency) for a chain."""
    reach, fallbacks, latency = 1.0, 0.0, overhead
    for k, p in enumerate(providers):
        fallbacks += reach if k else 0.0
        latency += reach * p.base_latency_ms * ((1 - p.error_rate) + 0.3 * p.error_rate)
        reach *= p.error_rate
    return round(1 - reach, 6), round(fallbacks, 6), round(latency, 6)


def features(doc):
    """Numbered items under the lesson's 'Six core features' heading."""
    block = doc.split("### Six core features", 1)[1].split("###", 1)[0]
    return sum(line[:1].isdigit() for line in block.splitlines())


def with_rate(ref, rate, n):
    original = list(ref.PROVIDERS)
    ref.PROVIDERS[:] = [dataclasses.replace(p, error_rate=rate) for p in original]
    try:
        return ref.simulate_fallback("Kong", n=n), exact(ref.PROVIDERS)
    finally:
        ref.PROVIDERS[:] = original


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    small, expected = with_rate(ref, RATE, 1000)
    big, _ = with_rate(ref, RATE, BIG)
    return {
        "chain": [p.name for p in ref.PROVIDERS], "expected": expected,
        "small": small, "big": big,
        "shipped": {g: ref.simulate_fallback(g) for g in ref.GATEWAY_OVERHEAD},
        "overhead": dict(ref.GATEWAY_OVERHEAD),
        "fail_ms": ref.call_provider(ref.PROVIDERS[0], AlwaysFail())[1],
        "six": features(parity.doc_text(PHASE, LESSON)),
    }


def verify(result):
    served, fallbacks, _ = result["expected"]
    big, small, shipped = result["big"], result["small"], result["shipped"]
    base = {g: round(r["mean_latency"] - result["overhead"][g], 6) for g, r in shipped.items()}
    same = {(r["success_rate"], r["retries"], r["fallback_hits"]) for r in shipped.values()}
    failed = round(BIG * (1 - big["success_rate"]))
    return [
        practice.Check(
            "ANSWER: 99.9875% served, 5% hit a fallback, 0.25% reach self-hosted",
            all([result["chain"] == ["OpenAI", "Anthropic", "Self-hosted"],
                 served == 0.999875, fallbacks == 0.0525,
                 abs(big["fallback_hits"] / BIG - fallbacks) < 0.001,
                 small["success_rate"] == 1.0]),
            f"exact {served} served and {fallbacks} fallback calls/request; simulated at "
            f"{BIG}: {big['success_rate']} and {big['fallback_hits']} fallbacks; at n=1000 "
            f"it prints {small['success_rate']:.1%} with {small['fallback_hits']} fallbacks",
        ),
        practice.Check(
            "FINDING: the four gateways differ only by a constant",
            same == {(1.0, 34, 34)} and len(set(base.values())) == 1,
            f"every gateway: (success, retries, fallbacks) {same}; mean latency minus "
            f"overhead {base}",
        ),
        practice.Check(
            "FINDING: every 'retry' is a fallback, and there is no backoff",
            big["retries"] - big["fallback_hits"] == failed == 20,
            f"retries {big['retries']} - fallbacks {big['fallback_hits']} = {failed} "
            f"requests that failed all three providers",
        ),
        practice.Check(
            "FINDING: a failed call costs 0.3 of the provider's latency, not half",
            result["fail_ms"] == 54.0 and result["six"] == 7,
            f"a failed 180 ms OpenAI call costs {result['fail_ms']} ms; 'Six core "
            f"features' numbers {result['six']} items",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
