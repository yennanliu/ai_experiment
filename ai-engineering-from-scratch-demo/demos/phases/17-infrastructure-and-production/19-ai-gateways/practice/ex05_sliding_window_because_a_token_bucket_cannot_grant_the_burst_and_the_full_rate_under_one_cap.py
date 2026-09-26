"""Exercise 5 — sliding window, because a token bucket cannot grant the burst and the full rate under one cap.

    Design a rate-limit policy for a multi-tenant SaaS: free tier, trial tier,
    paid tier. Token-bucket or sliding-window?

Reading of the exercise: each tier gets a per-minute quota -- free 20, trial
100, paid 1000 requests (the design's choice) -- and the question is which
algorithm enforces it. The lesson's `code/main.py` has no rate limiter, so
four are built here on a virtual clock: token bucket, fixed window,
sliding-window log and sliding-window counter. Each faces the pattern that
separates them: a tenant idle for 30 s, then flooding at 4x its quota for 90 s.
"Enforces" is measured as the most requests admitted in any 60 s span.

**ANSWER: sliding window -- the log for every tier.** It is the only one that
never admits more than the quota in any 60 s (20 / 100 / 1000) while still
letting a tenant spend its whole minute at once and sustain the full rate. The
natural token bucket -- capacity = quota, refill = quota per minute -- admits
2L - 1 in one minute (39 / 199 / 1999): the full bucket plus a minute of
refill. The fixed window admits 2L across a boundary.

**FINDING: a token bucket can have the burst or the full rate, not both.**
Its worst minute is capacity + refill x 60 s. Halve both and it stays under
the quota (999 for paid), but the 90 s flood then gets 1249 requests through
where the sliding log serves 2000: the sustained rate is halved to buy the
cap. The lesson's "sliding-window + burst allowance" is this trade-off: the
log's burst allowance is the whole quota, for free.

**FINDING: the cheap sliding-window counter overshoots by half.** Weighting
the previous window's count assumes its traffic was spread evenly; after an
idle start it was not, and the counter admits 1.5L (30 / 150 / 1500). Its
memory is two integers per tenant against up to L timestamps for the log --
1000 per paid tenant -- so at scale the counter is a deliberate 1.5x
tolerance, not a free approximation.

**FINDING: the lesson's simulator has no rate limit to configure.** The
module defines no limiter, bucket or window; its "rate limits" feature exists
only in the prose.

Structure: each limiter is a closure `allow(t) -> bool` (the fixed window and
the counter share one); `worst()` slides a 60 s span over the admitted
timestamps.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "19-ai-gateways"
W, QUOTAS = 60.0, (20, 100, 1000)  # free, trial, paid: requests per minute
WORDS = ("limit", "bucket", "window", "throttle")


def token_bucket(limit, scale=1.0):
    cap = limit * scale
    s = {"t": 0.0, "tok": cap}

    def allow(t):
        s["tok"], s["t"] = min(cap, s["tok"] + (t - s["t"]) * cap / W), t
        ok = s["tok"] >= 1
        s["tok"] -= ok
        return ok
    return allow


def sliding_log(limit):
    q = []

    def allow(t):
        q[:] = [x for x in q if x > t - W]
        ok = len(q) < limit
        q.extend([t] * ok)
        return ok
    return allow


def window_counter(limit, weighted):
    """Fixed window, or with `weighted` the sliding-window counter approximation."""
    s = {"w": 0, "cur": 0, "prev": 0}

    def allow(t):
        w = int(t // W)
        if w != s["w"]:
            s["prev"], s["cur"], s["w"] = s["cur"] if w == s["w"] + 1 else 0, 0, w
        ok = weighted * s["prev"] * (1 - (t % W) / W) + s["cur"] < limit
        s["cur"] += ok
        return ok
    return allow


LIMITERS = {"token_bucket": token_bucket, "half_bucket": lambda n: token_bucket(n, 0.5),
            "fixed_window": lambda n: window_counter(n, False), "sliding_log": sliding_log,
            "sliding_counter": lambda n: window_counter(n, True)}


def worst(times):
    best = j = 0
    for i, t in enumerate(times):
        while times[j] <= t - W:
            j += 1
        best = max(best, i - j + 1)
    return best


def attack(make):
    """Idle 30 s, then 4x the quota for 90 s: ([worst minute], [total admitted])."""
    out = []
    for limit in QUOTAS:
        allow = make(limit)
        times = [30 + k * W / (4 * limit) for k in range(int(360 * limit / W))]
        admitted = [t for t in times if allow(t)]
        out.append((worst(admitted), len(admitted)))
    return [w for w, _ in out], [n for _, n in out]


def scaled(k, c=0):
    return [int(k * q) + c for q in QUOTAS]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runs = {name: attack(make) for name, make in LIMITERS.items()}
    return {
        "worst": {name: r[0] for name, r in runs.items()},
        "total": {name: r[1] for name, r in runs.items()},
        "ref_limiters": [n for n in dir(ref) if any(w in n.lower() for w in WORDS)],
    }


def verify(result):
    worst_, total = result["worst"], result["total"]
    return [
        practice.Check(
            "ANSWER: sliding window -- the log for every tier",
            worst_["sliding_log"] == scaled(1) and worst_["token_bucket"] == scaled(2, -1)
            and worst_["fixed_window"] == scaled(2),
            f"worst 60 s admitted, quotas {list(QUOTAS)}: {worst_}",
        ),
        practice.Check(
            "FINDING: a token bucket can have the burst or the full rate, not both",
            worst_["half_bucket"] == scaled(1, -1) and total["half_bucket"] == scaled(1.25, -1)
            and total["sliding_log"] == scaled(2),
            f"a half-size bucket peaks at {worst_['half_bucket']} but serves "
            f"{total['half_bucket']} over the 90 s flood, the sliding log {total['sliding_log']}",
        ),
        practice.Check(
            "FINDING: the cheap sliding-window counter overshoots by half",
            worst_["sliding_counter"] == scaled(1.5),
            f"sliding counter worst minute {worst_['sliding_counter']}, quotas {list(QUOTAS)}",
        ),
        practice.Check(
            "FINDING: the lesson's simulator has no rate limit to configure",
            result["ref_limiters"] == [],
            f"names in code/main.py matching {', '.join(WORDS)}: {result['ref_limiters']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
