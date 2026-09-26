"""Exercise 3 — "GPUs available" is a cluster sum no scheduler uses, so 392 of 609 available states cannot place an 8-GPU pod.

    Your team reports that deployments are stuck in Pending because "GPUs
    available but pod won't schedule." Diagnose — is this Karpenter,
    kube-scheduler, or KAI Scheduler? Which metrics confirm?

Reading of the exercise: "GPUs available" is taken literally -- total free
GPUs >= GPUs requested -- and every free-GPU state of a 3-node x 8-GPU pool is
enumerated to see how often that is true while the job still cannot place.
The diagnosis is then a decision procedure over a cluster snapshot that names
the layer and the metric that confirms it; metric names were checked on
2026-09-26 against the Kubernetes metrics reference, the Karpenter metrics
reference, and the KAI Scheduler quickstart (`schedulerName: kai-scheduler`).

**ANSWER: kube-scheduler reports it, Karpenter owns it, and KAI owns it only
when the pod names kai-scheduler.** kube-scheduler fits pods node by node, so
free GPUs spread over nodes read as "available" and the pod goes
unschedulable: `scheduler_pending_pods{queue="unschedulable"}` rises. Adding a
node is then Karpenter's job; `karpenter_nodepools_usage` at
`karpenter_nodepools_limit` means the NodePool cap blocks it, and
`karpenter_cloudprovider_instance_launch_failures_total` rising means capacity.
A missing toleration is kube-scheduler's own verdict, and Karpenter cannot fix
it either: `karpenter_scheduler_unschedulable_pods_count` counts it too.
A pod with `schedulerName: kai-scheduler` never enters kube-scheduler's queue,
so a flat `scheduler_pending_pods` together with a Pending pod means KAI is holding an
incomplete gang, and its Kubernetes Events say why.

**FINDING: "GPUs available" is a cluster sum no scheduler uses.** Of the 729
free-GPU states, 609 have 8 or more free GPUs. One 8-GPU pod fits in only 217
of them, so 392 -- 64% -- are "available but won't schedule". Two 4-GPU pods
fit in 473 of the 609; eight 1-GPU pods fit in all 609.

**FINDING: without gang scheduling, 119 states strand part of an 8x1 job and
192 strand half of a 2x4 job.** On the default scheduler those states start
some pods and not all, and the started pods hold GPUs and wait: 16% of all
states for eight 1-GPU pods, 26% for two 4-GPU pods. A single 8-GPU pod has no
partial state -- it fits or it does not. Gang scheduling turns every one of those into all-pending. The
lesson's KAI_GANG strategy gang-schedules nothing: it is QUEUE_DEPTH with
lower thresholds, `GPU_PER_REPLICA` is defined and never read, and setting it
to 8 leaves KAI_GANG at 61 drops, unchanged.

Structure: `placeable()` is the node-by-node fit; `census()` enumerates
states; `diagnose()` maps a snapshot to (layer, cause, metric).
"""

from __future__ import annotations

import itertools

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "03-gpu-autoscaling-kubernetes"
NODES, GPUS = 3, 8
SHAPES = ((1, 8), (2, 4), (8, 1))  # (pods, GPUs per pod) for an 8-GPU job
PENDING = 'scheduler_pending_pods{queue="unschedulable"}'


def placeable(free, per_pod):
    return sum(f // per_pod for f in free)


def census():
    states = list(itertools.product(range(GPUS + 1), repeat=NODES))
    avail = [s for s in states if sum(s) >= GPUS]
    out = {"states": len(states), "available": len(avail)}
    for pods, per in SHAPES:
        out[(pods, per)] = {
            "fits": sum(placeable(s, per) >= pods for s in avail),
            "partial": sum(0 < placeable(s, per) < pods for s in states),
        }
    return out


KARPENTER = {
    "limit": ("NodePool limit reached", "karpenter_nodepools_usage vs _limit"),
    "capacity": ("no capacity", "karpenter_cloudprovider_instance_launch_failures_total"),
    "coming": ("node provisioning in flight", "karpenter_nodeclaims_created_total"),
}


def karpenter_cause(snap):
    if snap["nodepool_room"] < snap["pods"] * snap["per_pod"]:
        return ("Karpenter", *KARPENTER["limit"])
    return ("Karpenter", *KARPENTER["capacity" if snap["launch_failures"] else "coming"])


def diagnose(snap):
    """(layer, cause, confirming metric) for a Pending GPU job."""
    if not snap["tolerates"]:
        metric = f"{PENDING} + karpenter_scheduler_unschedulable_pods_count"
        return "kube-scheduler", "taint not tolerated", metric
    fits = placeable(snap["free"], snap["per_pod"]) >= snap["pods"]
    if snap["scheduler"] == "kai-scheduler":
        cause = "gang incomplete" if not fits else "queue quota or priority"
        return "KAI Scheduler", cause, f"flat {PENDING} + KAI Events"
    if fits:
        return "none", "schedulable; look past scheduling", "pod Events"
    return karpenter_cause(snap)


def snapshot(**over):
    base = dict(free=(3, 3, 2), pods=1, per_pod=8, scheduler="default-scheduler")
    return {**base, "tolerates": True, "nodepool_room": 0, "launch_failures": False, **over}


CASES = {
    "fragmented, pool full": snapshot(),
    "fragmented, no capacity": snapshot(nodepool_room=64, launch_failures=True),
    "fragmented, node coming": snapshot(nodepool_room=64),
    "missing toleration": snapshot(free=(8, 0, 0), tolerates=False),
    "KAI gang of 8": snapshot(free=(3, 3, 1), pods=8, per_pod=1, scheduler="kai-scheduler"),
}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    saved, ref.GPU_PER_REPLICA = ref.GPU_PER_REPLICA, 8
    try:
        kai8 = ref.simulate("KAI_GANG", ref.make_workload())["dropped"]
    finally:
        ref.GPU_PER_REPLICA = saved
    source = parity.lesson_dir(PHASE, LESSON).joinpath("code", "main.py").read_text()
    kai1 = ref.simulate("KAI_GANG", ref.make_workload())["dropped"]
    diagnoses = {k: diagnose(v) for k, v in CASES.items()}
    reads = source.count("GPU_PER_REPLICA")
    return dict(census=census(), diagnoses=diagnoses, kai1=kai1, kai8=kai8, gpu_reads=reads)


def verify(result):
    c, d = result["census"], result["diagnoses"]
    layers = [d[k][0] for k in CASES]
    return [
        practice.Check(
            "ANSWER: kube-scheduler reports it, Karpenter owns it, and KAI owns it only "
            "when the pod names kai-scheduler",
            layers + [d["KAI gang of 8"][1]]
            == ["Karpenter"] * 3 + ["kube-scheduler", "KAI Scheduler", "gang incomplete"],
            "; ".join(f"{k}: {v[0]} / {v[1]} / {v[2]}" for k, v in d.items()),
        ),
        practice.Check(
            'FINDING: "GPUs available" is a cluster sum no scheduler uses',
            [c["states"], c["available"]] + [c[s]["fits"] for s in SHAPES]
            == [729, 609, 217, 473, 609],
            f"{c['available']} of {c['states']} states have >= 8 free; fits: 1x8 "
            f"{c[(1, 8)]['fits']}, 2x4 {c[(2, 4)]['fits']}, 8x1 {c[(8, 1)]['fits']}",
        ),
        practice.Check(
            "FINDING: without gang scheduling, 119 states strand part of an 8x1 job and "
            "192 strand half of a 2x4 job",
            [c[s]["partial"] for s in SHAPES] + [result[k] for k in ("kai1", "kai8", "gpu_reads")]
            == [0, 192, 119, 61, 61, 1],
            f"partial starts: 8x1 {c[(8, 1)]['partial']}, 2x4 {c[(2, 4)]['partial']}; "
            f"KAI_GANG drops {result['kai1']} at GPU_PER_REPLICA=1 and {result['kai8']} at 8; "
            f"the name appears {result['gpu_reads']} time in the module",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
