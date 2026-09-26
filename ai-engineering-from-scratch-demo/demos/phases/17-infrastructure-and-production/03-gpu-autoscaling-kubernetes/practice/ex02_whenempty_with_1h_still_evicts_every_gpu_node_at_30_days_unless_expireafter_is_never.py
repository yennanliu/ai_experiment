"""Exercise 2 — WhenEmpty with 1h still evicts every GPU node at 30 days, unless expireAfter is Never.

    Design a Karpenter NodePool for a cluster serving Llama 3.3 70B FP8 on H100
    SXM5. Specify `capacity-type`, `disruption.consolidationPolicy`,
    `consolidateAfter`, and a taint that keeps non-GPU workloads off these nodes.

Reading of the exercise: the deliverable is a NodePool manifest, so the
solution builds one in the `karpenter.sh/v1` schema, sizes the node for the
model, and checks every field against the Karpenter docs (fetched 2026-09-26,
karpenter.sh/docs/concepts/{nodepools,disruption}). Nothing is applied to a
cluster; T0.

**ANSWER: on-demand p5.48xlarge, WhenEmpty, consolidateAfter 1h, taint
nvidia.com/gpu=true:NoSchedule, expireAfter Never.** p5.48xlarge is 8x H100
80 GiB at $55.04/h on-demand (us-east-1, Vantage). FP8 weights of 70.6B
parameters are 70.6 GB; at `--gpu-memory-utilization 0.9` TP=1 leaves 6.7 GB,
about 20k tokens of FP16 KV cache (80 layers x 8 KV heads x 128 x 2 x 2 bytes
= 320 KiB per token), and TP=2 leaves 84 GB, about 256k tokens. So TP=2 and
4 replicas per node. `on-demand`, because a spot reclaim costs a 95s
node-plus-model reload in the lesson's own constants. The taint is repelled
by every pod that does not tolerate it; the vLLM pods tolerate it.

**FINDING: WhenEmpty with 1h still evicts every GPU node at 30 days.** The
lesson says this setting "never evict[s] a running job". Karpenter's
`expireAfter` defaults to 720h, and Expiration is a *forceful* method: it
cannot be rate-limited by disruption budgets, and `karpenter.sh/do-not-disrupt`
does not exclude nodes from it. That is 12.2 forced drains per node-year, 146
a year on a 12-node pool. The design sets `expireAfter: Never` and rotates
nodes in a maintenance window instead.

**FINDING: the lesson names two policies and Karpenter now has three.** The
disruption docs list `WhenEmpty`, `Balanced` and `WhenEmptyOrUnderutilized`,
and the defaults are `WhenEmptyOrUnderutilized` with `consolidateAfter: 0s`,
so leaving the block out gets the trap. The lesson's code models no NodePool at
all; the nearest knob, NODE_PROVISION_SEC = 0 (every node already warm, which
a 1h hold approximates), takes QUEUE_DEPTH drops 64 -> 44.

Structure: `NODEPOOL` is the manifest as data, `render()` prints it as YAML,
and `kv_tokens()` does the sizing.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "03-gpu-autoscaling-kubernetes"
# verified against the Karpenter docs on 2026-09-26
CAPACITY_TYPES = ("spot", "on-demand", "reserved")
POLICIES = ("WhenEmpty", "Balanced", "WhenEmptyOrUnderutilized")
DEFAULTS = {"consolidationPolicy": "WhenEmptyOrUnderutilized", "consolidateAfter": "0s"}
DEFAULT_EXPIRE_H = 720
GPU_BYTES, PARAMS, UTIL = 80 * 2**30, 70.6e9, 0.9  # H100 SXM5 80 GiB, Llama 3.3 70B
KV_BYTES_PER_TOKEN = 80 * 8 * 128 * 2 * 2  # layers x kv heads x head dim x (K,V) x fp16


def req(key, value):
    return {"key": key, "operator": "In", "values": [value]}


TEMPLATE = {
    "requirements": [req("karpenter.sh/capacity-type", "on-demand"),
                     req("node.kubernetes.io/instance-type", "p5.48xlarge")],
    "taints": [{"key": "nvidia.com/gpu", "value": "true", "effect": "NoSchedule"}],
    "expireAfter": "Never",
    "nodeClassRef": {"group": "karpenter.k8s.aws", "kind": "EC2NodeClass", "name": "gpu"},
}  # fmt: skip
DISRUPTION = {
    "consolidationPolicy": "WhenEmpty",
    "consolidateAfter": "1h",
    "budgets": [{"nodes": "1"}],
}
NODEPOOL = {"apiVersion": "karpenter.sh/v1", "kind": "NodePool",
            "metadata": {"name": "h100-llama33-70b-fp8"},
            "spec": {"template": {"spec": TEMPLATE}, "disruption": DISRUPTION,
                     "limits": {"nvidia.com/gpu": "96"}}}  # fmt: skip


def render(node, indent=0):
    """Minimal YAML emitter for the manifest (dicts, lists of dicts, scalars)."""
    pad, lines = "  " * indent, []
    for key, value in node.items():
        if isinstance(value, dict):
            lines += [f"{pad}{key}:", render(value, indent + 1)]
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            lines.append(f"{pad}{key}:")
            for item in value:
                lines.append("  " * (indent + 1) + "- " + render(item, indent + 2).lstrip())
        else:
            lines.append(f"{pad}{key}: {value!r}".replace("'", '"'))
    return "\n".join(lines)


def kv_tokens(tp):
    """KV-cache tokens left after FP8 weights on `tp` GPUs at 0.9 utilization."""
    return (tp * GPU_BYTES * UTIL - PARAMS) / KV_BYTES_PER_TOKEN


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    saved, ref.NODE_PROVISION_SEC = ref.NODE_PROVISION_SEC, 0
    try:
        warm = ref.simulate("QUEUE_DEPTH", ref.make_workload())["dropped"]
    finally:
        ref.NODE_PROVISION_SEC = saved
    names = " ".join(dir(ref)).lower()
    return {
        "yaml": render(NODEPOOL),
        "spec": NODEPOOL["spec"],
        "warm": warm,
        "kv": {tp: kv_tokens(tp) for tp in (1, 2)},
        "reload_s": ref.NODE_PROVISION_SEC + ref.MODEL_LOAD_SEC,
        "cold": ref.simulate("QUEUE_DEPTH", ref.make_workload())["dropped"],
        "models_nodepool": "nodepool" in names or "karpenter" in names,
    }


def verify(result):
    tmpl, dis, kv = result["spec"]["template"]["spec"], result["spec"]["disruption"], result["kv"]
    cap = tmpl["requirements"][0]["values"][0]
    drains = 365 * 24 / DEFAULT_EXPIRE_H
    design = (cap, dis["consolidationPolicy"], dis["consolidateAfter"], tmpl["taints"][0]["effect"])
    return [
        practice.Check(
            "ANSWER: on-demand p5.48xlarge, WhenEmpty, consolidateAfter 1h, taint "
            "nvidia.com/gpu=true:NoSchedule, expireAfter Never",
            design == ("on-demand", "WhenEmpty", "1h", "NoSchedule")
            and cap in CAPACITY_TYPES
            and design[1] in POLICIES
            and (round(kv[1] / 1e3), round(kv[2] / 1e3)) == (20, 256)
            and 'consolidationPolicy: "WhenEmpty"' in result["yaml"],
            f"KV tokens after weights: TP=1 {kv[1]:,.0f}, TP=2 {kv[2]:,.0f} -> TP=2, "
            f"4 replicas per 8-GPU node; a reclaim costs {result['reload_s']}s of reload",
        ),
        practice.Check(
            "FINDING: WhenEmpty with 1h still evicts every GPU node at 30 days",
            (tmpl["expireAfter"], round(drains, 1), round(12 * drains)) == ("Never", 12.2, 146),
            f"default expireAfter {DEFAULT_EXPIRE_H}h is forceful: {drains:.1f} drains per "
            f"node-year, {12 * drains:.0f} on 12 nodes; the design sets expireAfter Never",
        ),
        practice.Check(
            "FINDING: the lesson names two policies and Karpenter now has three",
            (len(POLICIES), DEFAULTS["consolidationPolicy"], result["models_nodepool"])
            == (3, POLICIES[2], False)
            and (result["cold"], result["warm"]) == (64, 44),
            f"policies {POLICIES}, defaults {DEFAULTS}; the lesson code has no NodePool; "
            f"warm nodes take QUEUE_DEPTH drops {result['cold']} -> {result['warm']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
