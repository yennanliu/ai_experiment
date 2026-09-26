"""Exercise 5 — no, and the reference GLOBAL router sends EU requests to the US that a zone filter keeps home.

    A Paris-origin request matches a prefix in us-east-1. Do you route it?
    Write the policy.

Reading of the exercise: "Paris-origin" is read as an EU-resident tenant
whose data is bound to the EU by contract or regulation. The policy is
written below, and its routing core is run: a residency filter applied
*before* the reference's own cost rule. The US zone is us-east-1 and
us-west-2, the EU zone eu-west-1, and the router is the reference GLOBAL
inside each zone. Caches are exercise 1's FIFO.

**ANSWER: no. Filter by residency first, then optimise inside the zone.**
Policy:

1. The zone comes from the tenant's data agreement, not the client IP.
2. Candidates are only the replicas in the request's zone. This is a hard
   filter, applied before scoring, and has no TTFT override.
3. Inside the zone, pick min(prefill + RTT) as the reference GLOBAL does. If
   no replica in the zone holds the prefix, prefill locally.
4. For a shared prefix with no personal data, such as a system prompt or a
   tool schema, copy the prefix *text* into the zone and warm it there.
   Never move the request.
5. Log the serving region on every request and alarm on any out-of-zone
   serve. On a regional outage, fail over inside the zone or fail closed.

The router would take the bait. Serving the Paris request from us-east-1
costs 80 + 75 = 155 ms against an 800 ms local miss, and the reference GLOBAL
does exactly that for 175 of its 312 EU-origin requests. It also serves 191
US requests from eu-west-1. The zone filter takes both to 0. EU mean TTFT
goes from 264.2 to 576.2 ms, because eu-west-1 on its own has one caching
replica (exercise 2).

**FINDING: residency costs little once the local tie-breaker works.** Route
each prefix to a fixed one of eu-west-1's 4 replicas (exercise 2's
affinity). The EU partition then hits 87.2% at a mean of 172.3 ms with no
request leaving the EU, below the 264.2 ms the violating router got.

**FINDING: the reference has nothing to filter on.** `Request` carries only
`origin_region` and `prefix_hash`, with no tenant, zone or data class. Every
one of the 40 prefixes is shared across all three regions. The lesson's
"Partition routers by residency boundary" has no hook in the code.

**FINDING: the lesson's legal claim is stronger than GDPR.** It says a
Paris-to-us-east-1 route has "violated GDPR regardless" and calls the data
"EU customer PHI". PHI is a HIPAA term. GDPR Chapter V permits transfers
under an adequacy decision, and the Commission adopted one for the EU-US
Data Privacy Framework on 10 July 2023. So the binding rule is usually the
tenant's residency contract. The policy is the same either way: the
contract is what the router enforces.

Structure: `zoned()` splits the workload by zone and runs the reference
GLOBAL on each zone's replicas. `served_out()` counts a zone's requests
served outside it.
"""

from __future__ import annotations

import pathlib

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "11-multi-region-kv-locality"
EX01 = practice.load_module(next(pathlib.Path(__file__).resolve().parent.glob("ex01_*.py")))
ZONES = {"EU": ("eu-west-1",), "US": ("us-east-1", "us-west-2")}


def zone(region):
    return next(z for z, regions in ZONES.items() if region in regions)


def served_out(served, home="EU"):
    """Requests from `home` served outside it."""
    return sum(zone(q.origin_region) == home != zone(q.served_by.region) for q in served)


def eu_mean(served):
    eu = [q.ttft_ms for q in served if zone(q.origin_region) == "EU"]
    return round(sum(eu) / len(eu), 1), len(eu)


def zoned(ref, reqs, per_region=None):
    served = []
    for name, regions in ZONES.items():
        sub = [r for r in reqs if zone(r.origin_region) == name]
        served += EX01.fleet(ref, "GLOBAL", sub, regions=regions, per_region=per_region)[2]
    return served


def eu_affinity(ref, reqs, k=4):
    """eu-west-1 with prefix i pinned to replica i mod k; returns (hit rate, mean TTFT)."""
    eu = [r for r in reqs if zone(r.origin_region) == "EU"]
    served, hits = [], 0.0
    for part in range(k):
        sub = [r for r in eu if int(r.prefix_hash.split("_")[1]) % k == part]
        stats, _, out = EX01.fleet(ref, "GLOBAL", sub, regions=ZONES["EU"], per_region=1)
        served, hits = served + out, hits + stats["hit_rate"] * len(sub)
    return round(hits / len(eu), 3), eu_mean(served)[0], served_out(served)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    reqs = ref.make_workload()
    _, _, free = EX01.fleet(ref, "GLOBAL", reqs)
    fenced = zoned(ref, reqs)
    doc = parity.doc_text(PHASE, LESSON)
    return {
        "paris": (ref.CACHE_HIT_MS + ref.rtt("eu-west-1", "us-east-1"), ref.CACHE_MISS_MS),
        "free_out": served_out(free), "free_in": served_out(free, "US"), "free_eu": eu_mean(free),
        "fenced_out": served_out(fenced), "fenced_eu": eu_mean(fenced),
        "affinity": eu_affinity(ref, reqs),
        "fields": sorted(ref.Request.__dataclass_fields__),
        "prefixes": len({r.prefix_hash for r in reqs}),
        "shared": all(len({r.origin_region for r in reqs if r.prefix_hash == p}) == 3
                      for p in {r.prefix_hash for r in reqs}),
        "doc": ("violated GDPR regardless" in doc, "EU customer PHI" in doc),
    }


def verify(result):
    free, fenced, aff = result["free_eu"], result["fenced_eu"], result["affinity"]
    return [
        practice.Check(
            "ANSWER: no -- filter by residency first, then optimise inside the zone",
            result["paris"] == (155, 800) and result["free_out"] == 175 and result["free_in"] == 191
            and result["fenced_out"] == 0 and fenced[0] > free[0],
            f"the Paris request costs {result['paris'][0]} ms remote vs {result['paris'][1]} "
            f"ms local miss; GLOBAL serves {result['free_out']} of {free[1]} EU requests in "
            f"the US and {result['free_in']} US requests in the EU; the zone filter 0; EU mean TTFT {free[0]} -> {fenced[0]} ms",
        ),
        practice.Check(
            "FINDING: residency costs little once the local tie-breaker works",
            aff[2] == 0 and aff[1] < free[0],
            f"eu-west-1 with prefix affinity hits {aff[0]:.1%} at {aff[1]} ms, 0 out of "
            f"zone, against {free[0]} ms for the router that crossed the border",
        ),
        practice.Check(
            "FINDING: the reference has nothing to filter on",
            result["fields"] == ["crossregion", "origin_region", "prefix_hash", "served_by",
                                 "ttft_ms"] and result["shared"],
            f"Request fields {result['fields']}; all {result['prefixes']} prefixes are "
            "requested from all three regions",
        ),
        practice.Check(
            "FINDING: the lesson's legal claim is stronger than GDPR",
            all(result["doc"]),
            "the lesson says 'violated GDPR regardless' of 'EU customer PHI'; PHI is a HIPAA "
            "term, and GDPR Chapter V allows transfers under the 10 July 2023 EU-US Data "
            "Privacy Framework adequacy decision",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
