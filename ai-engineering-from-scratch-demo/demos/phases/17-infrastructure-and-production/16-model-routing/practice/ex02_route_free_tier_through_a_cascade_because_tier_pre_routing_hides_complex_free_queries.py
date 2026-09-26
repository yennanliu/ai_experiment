"""Exercise 2 — route the free tier through a cascade, because tier pre-routing hides complex free queries.

    Your user base is 30% enterprise (complex queries), 70% free tier
    (simple). Design the routing split. What online metric gates it?

Reading of the exercise: 1000 requests, 30% from enterprise users. Token
ranges are the reference workload's: "simple" is its simple bucket and
"complex" its medium/hard buckets in its own 3:1 ratio. Costs and quality come
from the reference's `cost_of` and `quality`, and the cascade uses the
reference's confidence rule. The tier label is free at request time, so four
splits are priced: all frontier; free -> cheap and enterprise -> frontier;
cascade for everyone; and free -> cascade with enterprise -> frontier. The
exercise's premise (free = simple) is then relaxed by a leak knob: the share
of free-tier queries that are actually complex.

**ANSWER: free tier -> cascade, enterprise -> frontier, gated on the free
tier's escalation rate.** With the premise exact it costs the same as the
pure tier split -- $4.28 against $6.61, a 35.2% saving at 99.31% quality --
because a simple query never escalates, so the cascade's extra call is
never paid. When 10% of free queries are complex, the pure tier split
drops to 98.43% quality and emits no signal at all: nothing escalates. The
cascade split holds 99.16% and its escalation rate climbs from 0 to 7.0%
(13.7% at a 20% leak); the insurance costs $5.24 against $4.35. That rate is the online metric, with a 5% judged
sample on the cheap route to catch what escalation misses.

**FINDING: the tier split saves 35.2%, not the Problem section's ~65%.**
The lesson prices this exact 70/30 scenario at "~65%". That holds only if
a simple query costs as much as a complex one. Here free-tier traffic is
70% of requests but only 38.0% of frontier spend, and on that traffic the
cheap model costs 7.5% of frontier, not the "3%" claimed. The saving is
0.380 x (1 - 0.075) = 35.2%.

**FINDING: a cascade for everyone is cheaper and worse.** $3.45 (47.8%
saved) at 98.34%: it keeps half the enterprise medium queries on the cheap
model, and its escalation rate (19.1%) sits inside the lesson's 30%
alarm.

Structure: `users()` draws the same uniforms for every leak value, so only
the leaked labels change; `run()` prices a tier -> route policy.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "16-model-routing"
RANGES = {
    "simple": ((200, 1000), (50, 200)),
    "medium": ((800, 3000), (100, 400)),
    "hard": ((2000, 8000), (200, 1500)),
}
POLICIES = {
    "all_frontier": lambda tier: "frontier",
    "tier": lambda tier: "cheap" if tier == "free" else "frontier",
    "cascade_all": lambda tier: "cascade",
    "free_cascade": lambda tier: "cascade" if tier == "free" else "frontier",
}


def users(ref, leak=0.0, n=1000, seed=7):
    rng, out = random.Random(seed), []
    for _ in range(n):
        u_tier, u_leak, u_hard = rng.random(), rng.random(), rng.random()
        tier = "ent" if u_tier < 0.3 else "free"
        complex_ = tier == "ent" or u_leak < leak
        kind = ("hard" if u_hard < 0.25 else "medium") if complex_ else "simple"
        (p0, p1), (o0, o1) = RANGES[kind]
        out.append((tier, ref.Query(kind, rng.randint(p0, p1), rng.randint(o0, o1))))
    return out


def run(ref, policy, reqs):
    """(cost, mean quality, cascade escalation rate) for one tier -> route policy."""
    rng, cost, qual, escalated, cascaded = random.Random(11), 0.0, 0.0, 0, 0
    for tier, q in reqs:
        route = policy(tier)
        if route != "cascade":
            cost, qual = cost + ref.cost_of(route, q), qual + ref.quality(route, q)
            continue
        cascaded += 1
        confident = q.difficulty == "simple" or (
            q.difficulty == "medium" and rng.random() < 0.5
        )
        cost += ref.cost_of("cheap", q) + (
            0 if confident else ref.cost_of("frontier", q)
        )
        qual += ref.quality("cheap", q) if confident else 1.0
        escalated += not confident
    return (
        round(cost, 2),
        round(qual / len(reqs), 4),
        round(escalated / max(cascaded, 1), 4),
    )


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    table = {
        leak: {k: run(ref, p, users(ref, leak)) for k, p in POLICIES.items()}
        for leak in (0.0, 0.1, 0.2)
    }
    reqs = users(ref)
    free = [q for tier, q in reqs if tier == "free"]
    spend = sum(ref.cost_of("frontier", q) for _, q in reqs)
    free_spend = sum(ref.cost_of("frontier", q) for q in free)
    return {
        "table": table,
        "free_share": round(free_spend / spend, 3),
        "cheap_ratio": round(
            sum(ref.cost_of("cheap", q) for q in free) / free_spend, 3
        ),
        "claim": "~65%" in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    t0, t1, t2 = (result["table"][leak] for leak in (0.0, 0.1, 0.2))
    save = round(1 - t0["tier"][0] / t0["all_frontier"][0], 3)
    return [
        practice.Check(
            "ANSWER: free tier -> cascade, enterprise -> frontier, gated on the free tier's "
            "escalation rate",
            all(
                [
                    t0["free_cascade"] == t0["tier"][:2] + (0.0,),
                    save == 0.352,
                    t1["free_cascade"][1] > t1["tier"][1],
                    t1["free_cascade"][2] == 0.0699,
                    t2["free_cascade"][2] == 0.1373,
                ]
            ),
            f"premise exact: {t0['free_cascade']} vs tier {t0['tier']} (saves {save:.1%}); "
            f"10% leak: tier {t1['tier'][:2]} with no signal, free_cascade {t1['free_cascade']}; "
            f"20% leak escalation {t2['free_cascade'][2]}",
        ),
        practice.Check(
            "FINDING: the tier split saves 35.2%, not the Problem section's ~65%",
            result["claim"]
            and result["free_share"] == 0.38
            and result["cheap_ratio"] == 0.075,
            f"free-tier traffic is {result['free_share']:.1%} of frontier spend and the cheap "
            f"model costs {result['cheap_ratio']:.1%} of frontier on it",
        ),
        practice.Check(
            "FINDING: a cascade for everyone is cheaper and worse",
            t0["cascade_all"] == (3.45, 0.9834, 0.191),
            f"cascade_all {t0['cascade_all']} (cost, quality, escalation) against the "
            "lesson's 30% over-routing alarm",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
