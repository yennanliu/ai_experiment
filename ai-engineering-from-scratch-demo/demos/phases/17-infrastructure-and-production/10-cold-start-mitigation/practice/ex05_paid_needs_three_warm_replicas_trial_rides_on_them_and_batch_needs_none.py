"""Exercise 5 — paid needs three warm replicas, trial rides on them, and batch needs none.

    Design a tiered warm-pool policy: how many warm replicas for paid users,
    trial users, and batch workloads? Show the math.

Reading of the exercise: `min_workers` must carry a tier's traffic for one
cold start -- the reference's raw 328 s, the time before an autoscaled replica
can help -- so each tier is sized on the peak rate it can reach inside that
window. Requests in flight are then Poisson with mean rate x service time
(M/G/inf, exact for any service-time distribution), and a request that finds
every warm slot busy waits for a cold start. Warm replicas N is the smallest N
with P(in flight > N * slots) <= the tier's miss budget. Assumed: 8 slots per
replica, 6 s per request, GPU at the reference's $4.50/hr; paid is 5 tenants
at 0.4 req/s with a P99 SLA (budget 0.01), trial 0.3 req/s at P95 (0.05),
batch one nightly 20,000-item job with no latency SLA.

**ANSWER: paid 3, trial 0 dedicated, batch 0.** Paid has 12 requests in
flight on average; 2 replicas (16 slots) are exceeded 10.13% of the time, 3
(24 slots) 0.07%. Trial alone would need 1 replica (1.8 in flight), but pooled
onto paid's 3 replicas the combined 13.8 in flight exceed 24 slots 0.42% of the
time -- inside paid's own 1% budget, so trial costs no extra replica. Batch
runs 15,000 s at 1.33 req/s per replica; the 328 s cold start is 2.2% of the
job, and keeping a replica warm the other 19.8 hours would cost $89.25 a day
to save $0.41 of boot. Warm pool: 3 replicas, $9,720/month.

**FINDING: the lesson's per-tenant premium floor costs 6 replicas where the
pooled policy needs 3.** "min_workers per tenant" gives each paid tenant its
own replica (2.4 in flight, 1 replica each) and trial its own: 6 replicas,
$19,440/month, twice the pooled cost for the same budgets. The lesson's
3,600 GPU-hours for 5 products is correct; at $4.50/hr that is $16,200/month.

**FINDING: the reference cannot express a tier.** `main.py` has no tier,
concurrency or replica count; `min_workers` appears only in text, and
`warm_pool_break_even` prices exactly one replica ($3,240/month).

Structure: `warm_replicas()` is the Poisson tail search; tiers are `TIERS`.
"""

from __future__ import annotations

import inspect
import math

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "10-cold-start-mitigation"
SLOTS, SERVICE_S, G, HOURS = 8, 6.0, 4.50, 24 * 30
TIERS = {"paid": (5 * 0.4, 0.01), "trial": (0.3, 0.05)}  # peak req/s, miss budget
BATCH_ITEMS = 20_000


def tail(mean, slots):
    """P(Poisson(mean) > slots)."""
    return 1 - sum(math.exp(-mean) * mean**i / math.factorial(i) for i in range(slots + 1))


def warm_replicas(rate, budget):
    n = 0
    while tail(rate * SERVICE_S, n * SLOTS) > budget:
        n += 1
    return n


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cold = ref.total_for_stack(set())
    (paid, paid_eps), (trial, trial_eps) = TIERS["paid"], TIERS["trial"]
    n_paid = warm_replicas(paid, paid_eps)
    job_s = BATCH_ITEMS / (SLOTS / SERVICE_S)
    per_tenant = 5 * warm_replicas(paid / 5, paid_eps) + warm_replicas(trial, trial_eps)
    src = inspect.getsource(ref)
    return {
        "cold": cold, "paid": n_paid, "paid_tail_2": tail(paid * SERVICE_S, 2 * SLOTS),
        "trial_alone": warm_replicas(trial, trial_eps),
        "pooled": warm_replicas(paid + trial, paid_eps),
        "pooled_tail": tail((paid + trial) * SERVICE_S, n_paid * SLOTS),
        "batch_share": cold / job_s, "batch_warm_day": (24 - job_s / 3600) * G,
        "batch_boot": cold / 3600 * G, "per_tenant": per_tenant,
        "monthly": n_paid * G * HOURS, "per_tenant_monthly": per_tenant * G * HOURS,
        "lesson_5": 5 * 24 * 30 * G,
        "ref_words": [w for w in ("tier", "concurrency", "replicas") if w in src],
    }


def verify(r):
    return [
        practice.Check(
            "ANSWER: paid 3, trial 0 dedicated, batch 0",
            all([r["paid"] == 3, round(r["paid_tail_2"], 4) == 0.1013, r["trial_alone"] == 1,
                 r["pooled"] == 3, r["pooled_tail"] < 0.01, round(r["batch_share"], 3) == 0.022,
                 round(r["batch_warm_day"], 2) == 89.25, r["monthly"] == 9720]),
            f"paid {r['paid']} (2 replicas miss {r['paid_tail_2']:.2%}); paid+trial pooled "
            f"{r['pooled']} at {r['pooled_tail']:.2%}; batch cold start "
            f"{r['batch_share']:.1%} of the job, warm ${r['batch_warm_day']:.2f}/day vs "
            f"${r['batch_boot']:.2f} boot; ${r['monthly']:,.0f}/month",
        ),
        practice.Check(
            "FINDING: the per-tenant premium floor costs 6 replicas where pooling needs 3",
            r["per_tenant"] == 6 and r["per_tenant_monthly"] == 2 * r["monthly"]
            and r["lesson_5"] == 16200,
            f"{r['per_tenant']} replicas, ${r['per_tenant_monthly']:,.0f}/month; the "
            f"lesson's 5 products x 24 x 30 GPU-hours is ${r['lesson_5']:,.0f}/month",
        ),
        practice.Check(
            "FINDING: the reference cannot express a tier",
            r["ref_words"] == [],
            f"words tier/concurrency/replicas in main.py: {r['ref_words']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
