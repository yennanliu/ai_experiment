"""Exercise 4 — recovery takes about three minutes, so a two-minute spike ends before the fleet catches up.

    Spike test from 10 RPS to 100 RPS. What's the expected recovery time if
    Karpenter + vLLM production-stack are in place (Phase 17 · 03 + 18)?

Reading of the exercise: the spike is driven through lesson 03's own
autoscaling simulator (Karpenter 50 s + 45 s model load, 15 s HPA tick, 30 s
queue timeout) with its queue-depth HPA. That simulator serves one request
per replica per tick, so both rates are divided by 150: 10 -> 100 RPS becomes
1/15 -> 2/3 req/s, which keeps the 10x ratio and needs 1 -> 10 replicas.
Recovery is the time from the spike's start to the last request that is
dropped or waits more than one tick. The spike is the lesson's own pattern --
"3-10x RPS for 2 min" -- and a sustained 10-minute version.

**ANSWER: about 3 minutes -- 193.5 s for a sustained 10x.** Queue-depth HPA
drops 55 requests and waits settle at +193.5 s. The chain is detection (15 s
tick) + Karpenter (50 s) + model load (45 s) = 110 s before the first new
replica serves, then the backlog drains. Lesson 03's own "2-5 minutes" for a
from-zero request brackets it. KAI's more aggressive rule drops 48;
Cluster Autoscaler's 110 s provisioning drops 69.

**FINDING: the lesson's 2-minute spike ends before the fleet recovers.** On
the 120 s spike the queue-depth fleet drops 55 of its 80 spike arrivals (69%)
and waits settle at +135 s -- after the spike is over. The 10-minute spike
drops the same 55, so the loss is set by the provisioning chain, not the
spike. A 2-minute spike test measures warm headroom, not autoscaling.

**FINDING: only headroom or a shorter chain moves the number.** Ten warm
replicas drop 0. Four drop 29. A zero-second model load (weights already on
the node) drops 28, and zero provisioning as well drops 2.

**FINDING: production-stack contributes no term.** Lesson 18's module has no
router, autoscaler or replica count -- `make_workload`, `simulate` and KV-block
constants only -- and its docs describe KV offload and cache-aware routing,
which change per-replica work, not how fast capacity arrives.

Structure: `spike()` builds evenly spaced arrivals; `run()` patches lesson 03's
constants for one run and restores them; `recovery()` reads the request
outcomes the reference simulator records.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "22-load-testing-llm-apis"
AUTOSCALER, STACK = "03-gpu-autoscaling-kubernetes", "18-vllm-production-stack-lmcache"
BASE_RPS, PEAK_RPS, SCALE, START, END = 10, 100, 150, 600, 1500
VARIANTS = {
    "karpenter": {}, "cluster autoscaler": {"NODE_PROVISION_SEC": 110},
    "warm 10": {"MIN_WARM_REPLICAS": 10}, "warm 4": {"MIN_WARM_REPLICAS": 4},
    "no model load": {"MODEL_LOAD_SEC": 0},
    "no provisioning": {"MODEL_LOAD_SEC": 0, "NODE_PROVISION_SEC": 0},
}


def spike(ref, seconds):
    reqs, t = [], 0.0
    while t < END:
        rps = PEAK_RPS if START <= t < START + seconds else BASE_RPS
        reqs.append(ref.Request(arrived_at=round(t, 3)))
        t += SCALE / rps
    return reqs


def recovery(reqs, tick):
    late = [r.arrived_at - START for r in reqs if r.arrived_at >= START and (
        r.dropped or (r.started_at is not None and r.started_at - r.arrived_at > tick))]
    return max(late) if late else 0.0


def run(ref, seconds=120, strategy="QUEUE_DEPTH", **overrides):
    saved = {k: getattr(ref, k) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(ref, k, v)
        reqs = spike(ref, seconds)
        out = ref.simulate(strategy, reqs)
    finally:
        for k, v in saved.items():
            setattr(ref, k, v)
    arrivals = sum(START <= r.arrived_at < START + seconds for r in reqs)
    return {"dropped": out["dropped"], "recovery": recovery(reqs, ref.HPA_TICK_SEC),
            "arrivals": arrivals}


def solve():
    ref = parity.load_reference(PHASE, AUTOSCALER, "main")
    stack = parity.load_reference(PHASE, STACK, "main")
    chain = ref.HPA_TICK_SEC + ref.NODE_PROVISION_SEC + ref.MODEL_LOAD_SEC
    return {
        "chain": chain, "sustained": run(ref, 600), "kai": run(ref, 600, "KAI_GANG"),
        "short": run(ref, 120), "variants": {k: run(ref, 120, **v) for k, v in VARIANTS.items()},
        "stack_names": sorted(n for n in vars(stack) if not n.startswith("_") and n.islower()),
        "doc_spike": "3-10x RPS for 2 min" in parity.doc_text(PHASE, LESSON),
        "doc_range": "2-5 minutes" in parity.doc_text(PHASE, AUTOSCALER),
    }


def verify(result):
    sus, short, v = result["sustained"], result["short"], result["variants"]
    return [
        practice.Check(
            "ANSWER: about 3 minutes -- 193.5 s for a sustained 10x",
            all([result["chain"] == 110, sus == {"dropped": 55, "recovery": 193.5, "arrivals": 400},
                 result["kai"]["dropped"] == 48, v["cluster autoscaler"]["dropped"] == 69,
                 result["doc_range"]]),
            f"first new replica after {result['chain']} s; sustained spike {sus}; KAI drops "
            f"{result['kai']['dropped']}, Cluster Autoscaler {v['cluster autoscaler']['dropped']}",
        ),
        practice.Check(
            "FINDING: the lesson's 2-minute spike ends before the fleet recovers",
            result["doc_spike"] and short == {"dropped": 55, "recovery": 135.0, "arrivals": 80},
            f"120 s spike: {short['dropped']} of {short['arrivals']} spike arrivals dropped, "
            f"waits settle at +{short['recovery']} s; the 600 s spike drops {sus['dropped']}",
        ),
        practice.Check(
            "FINDING: only headroom or a shorter chain moves the number",
            [v[k]["dropped"] for k in ("warm 10", "warm 4", "no model load", "no provisioning")]
            == [0, 29, 28, 2],
            f"dropped per variant {({k: r['dropped'] for k, r in v.items()})}",
        ),
        practice.Check(
            "FINDING: production-stack contributes no term",
            result["stack_names"] == ["annotations", "dataclass", "main", "make_workload",
                                      "random", "report", "simulate"],
            f"lesson 18's lowercase names: {result['stack_names']} -- no router or autoscaler",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
