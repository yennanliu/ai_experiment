"""Exercise 2 — the allowlist is checked against the URL the agent asked for.

    Implement a "navigate" action with an allowlist of URLs. What breaks if
    the agent tries to follow a redirect?

Reading of the exercise: what breaks is a timing question, not a matching
one. `assess` runs once, before execution, on the arguments the agent
supplied -- and a redirect is a second URL that the server chooses after the
check has already passed. So the allowlist is implemented here as the lesson
would, then run against a real chain, and the gap is measured rather than
argued.

**ANSWER: the gate admits 6 of 10 URLs, and a redirect walks through it.**
A navigate to an allowed host that redirects twice lands on
`grabber.example`, a host the allowlist rejects, with the verdict recorded
as `allow=True` -- **1** check performed, **3** URLs visited, **2** of them
never screened. Two of the six are admitted over `http` and `ftp`: an
allowlist of hosts is not an allowlist of URLs, and the shipped `Action`
carries no scheme field to check.

**FINDING: re-checking every hop closes it, and costs one check per hop.**
Assessing each `Location` before following it blocks the same chain at
`grabber.example` and raises the check count from **1** to **3**. Over the
lesson's 200-click horizon that is the difference between **200** and
**600** classifier calls on this fixture -- the per-step safety service has
to be per-*hop*, not per-action.

**FINDING: substring matching is both too loose and too tight.**
`"shop.example" in url` accepts `evil.test/?next=shop.example` and
`shop.example.attacker.test`, which parsing the netloc rejects -- and
rejects `SHOP.example`, which parsing accepts, because the raw URL was never
lowercased. **3** of **10** fixtures change verdict on that one line, in
both directions.

**FINDING: the shipped classifier already fails closed, so adding the action
is the risk.** `assess` ends in `SafetyVerdict(False, "unknown action kind")`,
so a `navigate` is denied **1** of **1** times before any allowlist exists.
Every gap above is introduced by the feature, not by the absence of it.

Structure: `WEB` is the redirect graph; `navigate()` follows it under either
a once-only or a per-hop gate.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "21-computer-use-agents"
ALLOWED = ("shop.example", "docs.example")
WEB = {
    "https://shop.example/cart": "https://shop.example/checkout",
    "https://shop.example/checkout": "https://grabber.example/steal",
    "https://docs.example/api": None,
    "https://grabber.example/steal": None,
}
URLS = ("https://shop.example/cart", "https://shop.example/checkout",
        "https://docs.example/api", "https://grabber.example/steal",
        "https://evil.test/?next=shop.example",
        "https://shop.example.attacker.test/login",
        "http://shop.example/cart", "ftp://docs.example/api",
        "https://SHOP.example/cart", "https://other.example/x")


def host_of(url):
    return url.split("://")[-1].split("/")[0].lower()


def permitted(url, strict=True):
    return host_of(url) in ALLOWED if strict else any(a in url for a in ALLOWED)


def assess_navigate(ref, url, strict=True):
    """The allowlist the exercise asks for, in the shipped verdict type."""
    if permitted(url, strict):
        return ref.SafetyVerdict(True, "host on allowlist")
    return ref.SafetyVerdict(False, f"host {host_of(url)!r} not on allowlist")


def navigate(ref, url, per_hop, limit=5):
    """Follow the redirect chain, gating once or at every hop."""
    checks, visited = 1, [url]
    verdict = assess_navigate(ref, url)
    if not verdict.allow:
        return {"checks": checks, "visited": visited, "landed": None,
                "allowed": False}
    while WEB.get(visited[-1]) and len(visited) < limit:
        nxt = WEB[visited[-1]]
        if per_hop:
            checks += 1
            if not assess_navigate(ref, nxt).allow:
                return {"checks": checks, "visited": visited, "landed": visited[-1],
                        "allowed": True, "stopped_at": nxt}
        visited.append(nxt)
    return {"checks": checks, "visited": visited, "landed": visited[-1],
            "allowed": True}


def unknown_kind(ref):
    guard = ref.SafetyClassifier(allowed_labels=("search_button",))
    screen = ref.Screen(elements=[], dom_text="")
    return guard.assess(ref.Action("navigate", {"url": URLS[0]}), screen)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    once = navigate(ref, URLS[0], per_hop=False)
    hopped = navigate(ref, URLS[0], per_hop=True)
    shipped = unknown_kind(ref)
    loose = [url for url in URLS if permitted(url, strict=False)]
    tight = [url for url in URLS if permitted(url, strict=True)]
    return {
        "urls": len(URLS), "admitted": len(tight), "loose_admitted": len(loose),
        "flipped": [host_of(u) for u in loose if u not in tight],
        "missed": [host_of(u) for u in tight if u not in loose],
        "schemes": sorted({u.split("://")[0] for u in tight}),
        "once": once, "hopped": hopped,
        "landed_allowed": permitted(once["landed"]),
        "unscreened": len(once["visited"]) - once["checks"],
        "horizon": {"per_action": 200, "per_hop": 200 * hopped["checks"]},
        "shipped_allow": shipped.allow, "shipped_reason": shipped.reason,
    }


def verify(result):
    once, hopped = result["once"], result["hopped"]
    return [
        practice.Check(
            "ANSWER: the gate admits 6 of 10 URLs and a redirect walks through it",
            all([result["admitted"] == 6, result["urls"] == 10,
                 result["schemes"] == ["ftp", "http", "https"],
                 once["allowed"] is True, once["checks"] == 1,
                 len(once["visited"]) == 3, result["unscreened"] == 2,
                 result["landed_allowed"] is False]),
            f"the allowlist admits {result['admitted']} of {result['urls']} URLs. A "
            f"navigate to an allowed host redirects twice and lands on "
            f"{host_of(once['landed'])!r}, which the allowlist rejects, with "
            f"allow={once['allowed']}: {once['checks']} check, "
            f"{len(once['visited'])} URLs, {result['unscreened']} never screened. The "
            f"admitted set spans {result['schemes']} -- a host allowlist is not a URL one",
        ),
        practice.Check(
            "FINDING: re-checking every hop closes it, at one check per hop",
            all([hopped["checks"] == 3, hopped.get("stopped_at") is not None,
                 permitted(hopped["landed"]) is True,
                 result["horizon"]["per_hop"] == 600]),
            f"assessing each Location before following it stops the chain at "
            f"{host_of(hopped['stopped_at'])!r} and raises the count from "
            f"{once['checks']} to {hopped['checks']}. Across the lesson's 200-click "
            f"horizon that is {result['horizon']['per_action']} calls against "
            f"{result['horizon']['per_hop']}: safety has to be per-hop, not per-action",
        ),
        practice.Check(
            "FINDING: substring matching is both too loose and too tight",
            all([result["loose_admitted"] == 7, result["admitted"] == 6,
                 result["flipped"] == ["evil.test", "shop.example.attacker.test"],
                 result["missed"] == ["shop.example"]]),
            f"'shop.example in url' admits {result['loose_admitted']} URLs where parsing "
            f"the netloc admits {result['admitted']}. It lets in {result['flipped']}, "
            f"which contain the allowed host without being it, and rejects "
            f"{result['missed']} spelled in capitals, which parsing accepts -- three "
            "verdicts changed by one line, in both directions",
        ),
        practice.Check(
            "FINDING: the shipped classifier already fails closed",
            all([result["shipped_allow"] is False,
                 "unknown action kind" in result["shipped_reason"]]),
            f"before any allowlist exists, assess falls through to "
            f"{result['shipped_reason']!r} with allow={result['shipped_allow']}, so a "
            "navigate is denied every time. Every gap above is introduced by adding the "
            "feature, not by the absence of it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
