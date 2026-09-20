"""Exercise 3 — the bucket has to precede the work it is protecting.

    Add a rate-limit check to `register_client` that runs before the
    registrar accepts a request. Use a token-bucket per source IP held in a
    small dict keyed by IP.

Reading of the exercise: "before the registrar accepts" fixes the position,
and position is the whole design -- a limiter placed after validation still
does the validation. So the limiter is installed twice, once in front and
once behind, and the difference is counted in work performed rather than in
requests refused, because both placements refuse the same requests.

**ANSWER: a token bucket per IP, in front, refusing the 6th request.**
Capacity **5**: the first five registrations from one IP return **201** and
the sixth answers **429** with `Retry-After`. A second IP is unaffected --
**5** more successes -- because the dict is keyed on the source.

**FINDING: placement is measurable as work, not as verdicts.** In front, the
sixth request costs **0** calls to the shipped `register_client`; behind, it
costs **1** -- the redirect-URI parsing, the application-type branch and a
`secrets.token_hex` all run before the refusal. Both return 429, so a test
that only reads the status cannot tell the two apart.

**FINDING: refusing late also mints.** The behind-placement runs
`register_client` to completion, so the authorization server's client table
grows by **1** on a request the limiter then rejects -- a registration the
caller never learns it has. Refusing before the work is what keeps the
refusal from having a side effect.

**FINDING: the bucket refills, so the limit is a rate and not a quota.** With
the clock advanced by 3 seconds at 1 token per second, **3** further requests
succeed from the same exhausted IP. A quota would need a different structure;
the exercise's own words -- token bucket -- choose the one that forgives.

Structure: `Registrar` wraps the lesson's `register_client` with a bucket
either in front of it or behind it, and counts the inner calls.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "18-mcp-auth-production"
CAPACITY, REFILL = 5, 1.0
BODY = {"application_type": "web", "redirect_uris": ["https://app.example/cb"]}


class Bucket:
    def __init__(self, now):
        self.tokens, self.updated = float(CAPACITY), now

    def take(self, now):
        self.tokens = min(CAPACITY, self.tokens + max(0.0, now - self.updated) * REFILL)
        self.updated = now
        if self.tokens < 1:
            return False
        self.tokens -= 1
        return True


class Registrar:
    """The lesson's registrar with a per-IP bucket, in front or behind."""

    def __init__(self, ref, auth, *, ahead=True):
        self.ref, self.auth, self.ahead, self.buckets, self.inner = ref, auth, ahead, {}, 0

    def register(self, body, ip, now):
        bucket = self.buckets.setdefault(ip, Bucket(now))
        if self.ahead and not bucket.take(now):
            return {"status": 429, "body": {"error": "slow_down"}, "Retry-After": "1"}
        self.inner += 1
        response = self.auth.register_client(body)
        if not self.ahead and not bucket.take(now):
            return {"status": 429, "body": {"error": "slow_down"}, "Retry-After": "1"}
        return response


def run(registrar, ip, count, now=1_000.0):
    return [registrar.register(dict(BODY), ip, now)["status"] for _ in range(count)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ahead = Registrar(ref, ref.AuthorizationServer())
    first = run(ahead, "10.0.0.1", 6)
    ahead_inner, ahead_clients = ahead.inner, len(ahead.auth.clients)  # before later probes
    second = run(ahead, "10.0.0.2", 5)
    refill = run(ahead, "10.0.0.1", 3, now=1_003.0)

    behind = Registrar(ref, ref.AuthorizationServer(), ahead=False)
    behind_statuses = run(behind, "10.0.0.1", 6)
    return {
        "first": first, "second": second,
        "retry_after": ahead.register(dict(BODY), "10.0.0.1", 1_000.0).get("Retry-After"),
        "ahead_inner": ahead_inner, "behind_inner": behind.inner,
        "ahead_clients": ahead_clients, "behind_clients": len(behind.auth.clients),
        "behind_statuses": behind_statuses,
        "refill": refill, "buckets": sorted(ahead.buckets),
        "capacity": CAPACITY,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: a token bucket per IP, in front, refusing the sixth request",
            all([result["first"] == [201] * 5 + [429], result["second"] == [201] * 5,
                 result["retry_after"] == "1",
                 result["buckets"] == ["10.0.0.1", "10.0.0.2"]]),
            f"capacity {result['capacity']}: one IP's first five registrations answer 201 and "
            f"the sixth {result['first'][-1]} with Retry-After "
            f"{result['retry_after']!r}, while a second IP gets {result['second']} because "
            f"the dict is keyed on the source -- {result['buckets']}",
        ),
        practice.Check(
            "FINDING: placement is measurable as work, not as verdicts",
            all([result["ahead_inner"] == 5, result["behind_inner"] == 6,
                 result["first"] == result["behind_statuses"]]),
            f"in front, six requests reach the shipped register_client "
            f"{result['ahead_inner']} times; behind, {result['behind_inner']} -- the URI "
            f"parsing, the application-type branch and a token_hex all run before the "
            f"refusal. Both placements answer {result['behind_statuses']}, so a test reading "
            "only the status cannot tell them apart",
        ),
        practice.Check(
            "FINDING: refusing late also mints",
            all([result["ahead_clients"] == 5, result["behind_clients"] == 6]),
            f"the behind-placement runs register_client to completion, so the client table "
            f"grows to {result['behind_clients']} against {result['ahead_clients']} -- a "
            "registration the caller is then told it did not get. Refusing before the work "
            "is what keeps the refusal free of side effects",
        ),
        practice.Check(
            "FINDING: the bucket refills, so the limit is a rate and not a quota",
            result["refill"] == [201, 201, 201],
            f"three seconds later at {REFILL} token per second the same exhausted IP gets "
            f"{result['refill']}. A quota would need a different structure; the exercise's "
            "own words -- token bucket -- choose the one that forgives",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
