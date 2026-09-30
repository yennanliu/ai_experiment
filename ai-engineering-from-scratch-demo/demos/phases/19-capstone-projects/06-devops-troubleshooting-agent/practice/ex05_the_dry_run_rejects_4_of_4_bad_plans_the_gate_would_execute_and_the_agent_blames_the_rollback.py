"""Exercise 5 -- the rollback dry-run rejects 4 of 4 bad plans the lesson's gate would execute, and the agent still blames the rollback.

    Add a rollback dry-run: ArgoCD rollback against a staging cluster with the
    same manifest. Verify the rollback plan in a live cluster before the Slack
    approval button.

Reading of the exercise: there is no live cluster, so "staging with the same
manifest" is a second `build_sample_cluster()`, and the plan is the one
`main()` proposes, `argocd_rollback {"app": "checkout-api", "to_revision":
41}`. The dry-run runs before any approval. It checks that the prod and
staging manifests match, that the target revision exists in the app's history
and is not the current one, and that applying it to staging changes only the
Deployment. Then it re-runs the lesson's `root_cause` on the rolled-back
staging graph, the check a live verification would make. The app history
(rev 41 = v2.40, rev 42 = v2.41) comes from the lesson's "Use It" transcript;
the graph itself stores only the current revision.

**ANSWER: the plan verifies: the manifests match, rev 41 exists, and the diff
is 2 fields (image v2.41 -> v2.40, revision 42 -> 41) on 1 object.** Four
malformed plans are rejected before the button: rev 999, rev 42 (the current
one), rev -1, and "latest". With approval, the lesson's gate executes all 4.

**FINDING: the lesson's graph cannot verify any rollback on its own.** It holds
revision 42 and nothing else. Without the history from the prose, the dry-run's
verdict on rev 41 is "unverifiable".

**FINDING: the lesson's agent blames the rollback it just verified.** After the
rollback on staging, `root_cause` still ranks "bad rollout: image
checkout-api:v2.40 fails /healthz" #1. Its score rises from 0.735 to 0.817,
because the telemetry nodes are unchanged and the fresh rollout counts as more
recent evidence. It cannot say a rollback worked.

**FINDING: the gate keys on tool name, so no dry-run can run before approval.**
`Agent.call("argocd_rollback", {..., "dry_run": True})` is "blocked: no slack
approval", and a read-only `argocd_app_diff` is "blocked: unknown tool": 0 of 2
pre-approval paths run. ArgoCD's `app rollback` has no `--dry-run` flag; the
preview is `argocd app sync --revision R --dry-run` ("Preview apply without
affecting cluster"; argo-cd.readthedocs.io, read 2026-09-29).
"""

from __future__ import annotations

import copy

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "06-devops-troubleshooting-agent"
PLAN = {"app": "checkout-api", "to_revision": 41}
HISTORY = {41: "checkout-api:v2.40", 42: "checkout-api:v2.41"}  # the lesson's Use It transcript
BAD = [999, 42, -1, "latest"]


def dry_run(prod, staging, plan, history):
    """Verify a rollback plan against staging; returns (verdict, diff, rolled-back graph)."""
    key = f"Deployment/{plan['app']}"
    if prod.nodes[key].attrs != staging.nodes[key].attrs or set(prod.nodes) != set(staging.nodes):
        return "manifest drift", {}, None
    current, target = staging.nodes[key].attrs.get("revision"), plan["to_revision"]
    if target not in history:
        return "unverifiable: revision not in history", {}, None
    if target == current:
        return "no-op: target is the current revision", {}, None
    trial = copy.deepcopy(staging)
    trial.nodes[key].attrs.update(revision=target, image=history[target], deployed_at="0m ago")
    before, after = staging.nodes[key].attrs, trial.nodes[key].attrs
    diff = {k: (before[k], after[k]) for k in ("revision", "image") if before[k] != after[k]}
    touched = [k for k in trial.nodes if trial.nodes[k].attrs != staging.nodes[k].attrs]
    return ("ok" if touched == [key] else "touches more than the Deployment"), diff, trial


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    prod, staging = ref.build_sample_cluster(), ref.build_sample_cluster()
    verdict, diff, trial = dry_run(prod, staging, PLAN, HISTORY)
    alerted = "Deployment/checkout-api"
    before = ref.root_cause(staging, alerted)[0]
    after = ref.root_cause(trial, alerted)[0]
    bad = {str(t): dry_run(prod, staging, {**PLAN, "to_revision": t}, HISTORY)[0] for t in BAD}
    gate = ref.Agent(graph=prod)
    executed = [gate.call("argocd_rollback", {**PLAN, "to_revision": t}, approver="alice@sre").executed
                for t in BAD]
    pre = [gate.call("argocd_rollback", {**PLAN, "dry_run": True}).result,
           gate.call("argocd_app_diff", PLAN).result]
    return {
        "verdict": verdict, "diff": diff, "bad": bad, "gate_executes_bad": sum(executed),
        "graph_only": dry_run(prod, staging, PLAN, {42: "checkout-api:v2.41"})[0],
        "graph_revisions": [n.attrs.get("revision") for n in prod.nodes.values() if "revision" in n.attrs],
        "before": (before.title, round(before.score(), 3)), "after": (after.title, round(after.score(), 3)),
        "pre_approval": pre,
    }


def verify(result):
    r = result
    rejected = [v for v in r["bad"].values() if v != "ok"]
    return [
        practice.Check(
            "ANSWER: rev 41 verifies on staging as a 2-field Deployment-only diff; 4/4 bad plans rejected",
            r["verdict"] == "ok" and r["diff"] == {"revision": (42, 41),
                                                   "image": ("checkout-api:v2.41", "checkout-api:v2.40")}
            and len(rejected) == 4 and r["gate_executes_bad"] == 4,
            f"verdict {r['verdict']}, diff {r['diff']}; bad plans {r['bad']}; lesson gate executes "
            f"{r['gate_executes_bad']}/4 of them when approved",
        ),
        practice.Check(
            "FINDING: the lesson's graph holds one revision, so alone it cannot verify a rollback",
            r["graph_revisions"] == [42] and r["graph_only"].startswith("unverifiable"),
            f"revisions in the graph {r['graph_revisions']}; verdict from the graph alone: {r['graph_only']}",
        ),
        practice.Check(
            "FINDING: after the rollback the agent blames the rollback, with a higher score",
            r["before"] == ("bad rollout: image checkout-api:v2.41 fails /healthz", 0.735)
            and r["after"] == ("bad rollout: image checkout-api:v2.40 fails /healthz", 0.817),
            f"#1 before {r['before']}, after {r['after']}",
        ),
        practice.Check(
            "FINDING: the gate keys on tool name, so 0 of 2 pre-approval dry-run paths run",
            r["pre_approval"] == ["blocked: no slack approval", "blocked: unknown tool"],
            f"dry_run=True rollback and argocd_app_diff: {r['pre_approval']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
