"""Exercise 1 -- on AWS's three EKS demo incidents the lesson's agent matches 1 remediation and 0 root causes.

    Run your agent on the same three incidents AWS's DevOps Agent is demo'd on.
    Publish the side-by-side. Report where the agent diverges.

Reading of the exercise: "the three incidents" are the three EKS investigations
AWS published for its DevOps Agent (read 2026-09-29): (A) API-server slowness
from a 50-replica controller exhausting API Priority and Fairness seats
(aws.amazon.com/blogs/containers, 2026-07-02); (B) an OOMKilled web-python pod
after an image change, caused by an unbounded in-memory list, remedied by a
rollback (aws.amazon.com/blogs/devops, 2026-08-31); (C) pods Running but DNS
failing because fault-injection iptables DROP rules sit on one node (blogs/devops,
2026-06-11). Each is rebuilt as the lesson's own `Graph` with the objects and
telemetry the post names, alerted where the alert would fire, and passed to the
lesson's `root_cause`. Its #1 hypothesis is scored against AWS's answer on two
axes: the root cause, and the remediation. Incident C is run twice, alerted at
the Deployment and at a Pod, because the answer changes.

**ANSWER: the side-by-side.**

| incident | AWS DevOps Agent | lesson agent #1 | cause | remedy |
|---|---|---|---|---|
| A apiserver 429s | controller floods APF | DNS flap in coredns, 0 citations | no | no |
| B OOMKilled | unbounded list, roll back | bad rollout, "fails /healthz" | no | yes |
| C DNS, Deployment | iptables DROP on node | bad rollout (deployed 3d ago) | no | no |
| C DNS, Pod | iptables DROP on node | node-level pressure, kernel | node only | no |

**FINDING: the graph cannot reach two of the three causes.** The lesson's edge
set has no client-to-API-server edge, so A's culprit Deployment is not a
neighbour of anything alerted; and `root_cause` only reads one hop, so a
Deployment alert never sees the Node its Pods run on. The node hypothesis fires
only when the alert names a Pod.

**FINDING: every hypothesis is a template.** "bad rollout" is emitted for
every Deployment alert, even one last deployed 3 days ago, and always says
"fails /healthz", including for an OOM. "DNS flap in kube-system/coredns" is
emitted on every incident with no citation, which the lesson's own skill file
lists as a hard reject, and it ranks last on C, the one DNS incident.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "06-devops-troubleshooting-agent"
AWS = {"A": ("controller Deployment floods API Priority and Fairness", "scale down controller"),
       "B": ("unbounded processed_records list leaks memory", "roll back"),
       "C": ("fault-injection iptables DROP rules on one node", "delete iptables rules")}


# incident -> (app, deployed_at, Prom series, Loki summary, also alert at the Pod)
WORKLOAD = {"B": ("web-python", "12m ago", "container_memory_working_set_bytes{pod=web-python}",
                  "OOMKilled x2", False),
            "C": ("shop-api", "3d ago", "coredns_errors", "SERVFAIL lookups", True)}


def apiserver_incident(ref):
    """A: the culprit is a client of the API server, and no lesson edge says so."""
    N = ref.Node
    objs = [N("Service", "kube-apiserver"), N("Deployment", "controller", {"replicas": 50}),
            N("Prom", "apiserver_request_duration_seconds", {"last_15m": "p99 multi-second"}),
            N("Loki", "eks-audit,code=429", {"last_15m": "429 from controller SA"})]
    links = [("Service/kube-apiserver", "OBSERVED_BY", o.key) for o in objs[2:]]
    return objs, links, ["Service/kube-apiserver"]


def workload_incident(ref, name):
    """B and C: a Deployment, one Pod on one Node, and two telemetry sources."""
    N = ref.Node
    app, age, series, logs, pod_alert = WORKLOAD[name]
    dep = N("Deployment", app, {"image": f"{app}:v2", "deployed_at": age})
    pod, node = N("Pod", f"{app}-645b"), N("Node", "ip-10-0-1-7", {"kernel": "6.1.109"})
    tel = [N("Prom", series), N("Loki", f"app={app}", {"last_15m": logs})]
    links = [(dep.key, "OWNS", pod.key), (pod.key, "SCHEDULED_ON", node.key)]
    links += [(dep.key, "OBSERVED_BY", t.key) for t in tel]
    return [dep, pod, node, *tel], links, [dep.key, pod.key][: 1 + pod_alert]


def incident(ref, name):
    return apiserver_incident(ref) if name == "A" else workload_incident(ref, name)


def run(ref, name):
    objs, links, alerted = incident(ref, name)
    g = ref.Graph()
    for o in objs:
        g.add(o)
    for s, rel, d in links:
        g.link(s, rel, d)
    rows = []
    for key in alerted:
        hyps = ref.root_cause(g, key)
        rows.append({"incident": name, "alerted": key, "top": hyps[0].title, "n": len(hyps),
                     "titles": [h.title for h in hyps], "citations": [len(h.citations) for h in hyps],
                     "scores": [round(h.score(), 3) for h in hyps],
                     "has_node": any(h.title.startswith("node-level") for h in hyps)})
    return rows


def judge(row):
    top = row["top"]
    cause = {"A": "controller" in top, "B": "memory" in top or "leak" in top,
             "C": "iptables" in top}[row["incident"]]
    implied = "roll back" if top.startswith("bad rollout") else None
    return cause, implied == AWS[row["incident"]][1]


def summary(rows):
    dns = "DNS flap in kube-system/coredns"
    return {
        "tops": [r["top"] for r in rows], "n": [r["n"] for r in rows],
        "remedy_ok": sum(r["remedy_ok"] for r in rows), "cause_ok": sum(r["cause_ok"] for r in rows),
        "node_runs": [i for i, r in enumerate(rows) if r["has_node"]],
        "dns_rank": [r["titles"].index(dns) + 1 for r in rows],
        "dns_citations": [r["citations"][r["titles"].index(dns)] for r in rows],
        "stale_rollout": rows[2]["scores"][0],
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = [r for name in AWS for r in run(ref, name)]
    for r in rows:
        r["cause_ok"], r["remedy_ok"] = judge(r)
    return summary(rows)


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: side-by-side on AWS's 3 EKS incidents -- 1 remediation matches, 0 root causes",
            (r["remedy_ok"], r["cause_ok"], r["tops"]) == (1, 0, [
                "DNS flap in kube-system/coredns", "bad rollout: image web-python:v2 fails /healthz",
                "bad rollout: image shop-api:v2 fails /healthz",
                "node-level pressure on ip-10-0-1-7 (kernel=6.1.109)"]),
            f"#1 per run (A, B, C at Deployment, C at Pod): {r['tops']}",
        ),
        practice.Check(
            "FINDING: the graph cannot reach A's culprit, and a Deployment alert never sees its Node",
            (r["n"][0], r["node_runs"]) == (1, [3]),
            f"A emits {r['n'][0]} hypothesis; the node hypothesis appears only on run {r['node_runs']} "
            f"(the Pod-alerted C)",
        ),
        practice.Check(
            "FINDING: hypotheses are templates; the uncited DNS guess ranks last on the DNS incident",
            (r["dns_citations"], r["dns_rank"], r["stale_rollout"]) == ([0, 0, 0, 0], [1, 2, 2, 2], 0.467),
            f"DNS rank per run {r['dns_rank']}, its citations {r['dns_citations']}; a 3-days-old "
            f"rollout is still #1 at {r['stale_rollout']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
