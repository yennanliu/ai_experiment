"""Exercise 5 — one consolidation event is 95s, which is more than a whole day of a 99.9% error budget.

    Compute the cost of the `WhenEmptyOrUnderutilized` consolidation trap on a
    24x7 production service that averages 60 request-dropping events/day at P99
    TTFT > 10s.

Reading of the exercise: an "event" is Karpenter consolidating a serving GPU
node: its pods are evicted and come back on a new node after the lesson's own
NODE_PROVISION_SEC + MODEL_LOAD_SEC. The cost is counted three ways: GPU money
billed while nothing serves, SLO time, and requests that miss the 10s TTFT.
Price: p5.48xlarge (8x H100) at $55.04/h on-demand, us-east-1, from Vantage on
2026-09-26. Dropped requests depend on node throughput, which the exercise
does not give, so they are shown at an illustrative 4 req/s per node.

**ANSWER: 95s per event, 5700s a day; $87/day ($31.8k/yr) of idle GPU, and
340 dropped requests per event at full load.** The replacement node bills
95s before it serves: $1.45 an event, $87.15 a day, $31,809 a year. At full
load on a 12-node pool, the evicted node's traffic cannot be absorbed. It
waits 95s, so every request arriving in the first 85s misses 10s TTFT: 340
per event, 20,400 a day, 7.4M a year. The survivors absorb it only when fleet
utilization is at most 11/12 = 91.7%; at 95% the loss is 136 per event.

**FINDING: one event is more than a whole day of a 99.9% error budget.** A
99.9% SLO allows 86.4s a day, and a single event is 95s. 60 events are 95
minutes a day, 6.6% of the day, so the node-level availability is 93.4%. A 99%
SLO allows 864s: 9 events fit (855s) and the 10th breaks it.

**FINDING: the lesson's simulator cannot price this.** It has no eviction
path, and its drop rule is a hard-coded 30s wait (`now - r.arrived_at > 30`),
not 10s TTFT. Under that rule the miss window is 65s and the count is 260 per
event. Karpenter's docs name two per-workload guards besides the policy.
`karpenter.sh/do-not-disrupt` on the pods, or a blocking PDB, stops graceful
consolidation. Neither stops Expiration (see exercise 2).

Structure: `per_event()` returns seconds, dollars and drops for one event;
`solve()` scales it to a day and a year and reads the reference constants.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "03-gpu-autoscaling-kubernetes"
EVENTS_PER_DAY, TTFT_SLO_S, DAY_S = 60, 10, 86_400
NODE_PRICE_H, NODES, NODE_RPS = 55.04, 12, 4.0


def per_event(outage_s, deadline_s=TTFT_SLO_S, utilization=1.0):
    """(outage seconds, idle-GPU dollars, requests missing the deadline)."""
    overflow = max(0.0, utilization * NODES - (NODES - 1))  # node-loads nobody absorbs
    drops = overflow * NODE_RPS * max(0.0, outage_s - deadline_s)
    return outage_s, outage_s / 3600 * NODE_PRICE_H, drops


def events_within(slo):
    budget = (1 - slo) * DAY_S
    return budget, int(budget // per_event(95)[0])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    outage = ref.NODE_PROVISION_SEC + ref.MODEL_LOAD_SEC
    seconds, dollars, drops = per_event(outage)
    source = parity.lesson_dir(PHASE, LESSON).joinpath("code", "main.py").read_text()
    return {
        "outage": outage,
        "dollars": dollars,
        "drops": drops,
        "day": {
            "s": EVENTS_PER_DAY * seconds,
            "usd": EVENTS_PER_DAY * dollars,
            "drops": EVENTS_PER_DAY * drops,
        },
        "year_usd": 365 * EVENTS_PER_DAY * dollars,
        "at95": per_event(outage, utilization=0.95)[2],
        "absorbed": per_event(outage, utilization=(NODES - 1) / NODES)[2],
        "slo": {slo: events_within(slo) for slo in (0.999, 0.99)},
        "sim_drops": per_event(outage, deadline_s=30)[2],
        "sim_rule": "now - r.arrived_at > 30" in source,
        "sim_evicts": "evict" in source.lower() or "consolidat" in source.lower(),
    }


def verify(result):
    day, slo = result["day"], result["slo"]
    avail = 1 - day["s"] / DAY_S
    return [
        practice.Check(
            "ANSWER: 95s per event, 5700s a day; $87/day ($31.8k/yr) of idle GPU, and "
            "340 dropped requests per event at full load",
            all(
                [
                    result["outage"] == 95,
                    day["s"] == 5700,
                    round(result["dollars"], 2) == 1.45,
                    round(day["usd"], 2) == 87.15,
                    round(result["year_usd"]) == 31_809,
                    result["drops"] == 340,
                    day["drops"] == 20_400,
                    round(result["at95"]) == 136,
                    round(result["absorbed"], 9) == 0,
                ]
            ),
            f"{result['outage']}s x {EVENTS_PER_DAY}/day = {day['s']}s; ${result['dollars']:.2f}"
            f"/event, ${day['usd']:.2f}/day, ${result['year_usd']:,.0f}/yr; drops/event "
            f"{result['drops']:.0f} at 100% load, {result['at95']:.0f} at 95%, 0 at 11/12",
        ),
        practice.Check(
            "FINDING: one event is more than a whole day of a 99.9% error budget",
            all(
                [
                    round(slo[0.999][0], 1) == 86.4,
                    slo[0.999][1] == 0,
                    slo[0.99][1] == 9,
                    round(avail, 3) == 0.934,
                ]
            ),
            f"99.9% allows {slo[0.999][0]:.1f}s/day ({slo[0.999][1]} events), 99% "
            f"{slo[0.99][0]:.0f}s ({slo[0.99][1]} events); 60 events leave {avail:.1%}",
        ),
        practice.Check(
            "FINDING: the lesson's simulator cannot price this",
            all([result["sim_rule"], not result["sim_evicts"], result["sim_drops"] == 260]),
            f"the reference drops after a hard-coded 30s wait and never evicts; under "
            f"that rule an event costs {result['sim_drops']:.0f} requests, not "
            f"{result['drops']:.0f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
