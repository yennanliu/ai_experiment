"""Exercise 2 -- over a simulated week, 32.4% of considered commands are near-misses and the lesson's list sees 32 of 46.

    Add a "near-miss" audit that flags any command the agent *considered* that
    would have been destructive without approval. Measure the near-miss rate
    over one week.

Reading of the exercise: no cluster exists, so "one week" is a seeded
simulation: 7 days of 3-9 alerts a day, each on the lesson's sample cluster,
alerted at the Deployment (70%) or a Pod. Per alert the agent runs the
lesson's own flow from `main()`: two read-only queries, `root_cause`, then it
proposes the remediation its #1 hypothesis implies (rollout -> argocd_rollback,
node -> kubectl_drain) through the lesson's `Agent.call` without an approver;
a Pod alert also considers `kubectl_exec` to inspect the container. A human
approves 40% of proposals, and the agent re-calls with that approver. The
audit reads the lesson's audit log and flags an event when the command is
destructive *by effect* (verb in delete/scale/rollback/drain/cordon/exec/
port-forward/patch) and the event carries no approval. Rate = near-misses /
considered events.

**ANSWER: 39 alerts, 142 considered commands, 46 near-misses: 32.4%, or 1.18
per alert.** Every alert ends in one unapproved destructive proposal, because
`root_cause`'s #1 always implies a remediation, and Pod alerts add an exec.
None of the 46 executed. The log itself shows 49; 3 of those were approved by
a human (next finding).

**FINDING: the lesson's own destructive list sees 32 of the 46.** Its
`destructive_tools` names scale/rollback/delete/argocd_rollback;
`kubectl_drain` and `kubectl_exec` are logged as "blocked: unknown tool",
though the skill file calls exec "destructive in effect". An audit keyed on
that list reports 22.5% instead of 32.4%.

**FINDING: the log loses human approvals.** `Agent.call` records the
approver only for listed tools, so 3 of the 18 human approvals (all
`kubectl_drain`) are logged as approved=False, approver=None, and read as
near-misses. The skill file requires the log to say "who approved".

**FINDING: the approval gate is a string.** `Agent.call(tool, args,
approver="devops-agent")` executes all 4 destructive tools, and the row looks
like a human's approval. `considered` is True on all 142 events, so the field
carries no information.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "06-devops-troubleshooting-agent"
VERBS = ("delete", "scale", "rollback", "drain", "cordon", "exec", "port_forward", "patch")
REMEDY = {"bad rollout": "argocd_rollback", "node-level": "kubectl_drain", "DNS flap": "kubectl_delete"}


def remedy_for(title):
    return next(tool for prefix, tool in REMEDY.items() if title.startswith(prefix))


def handle(ref, agent, alerted, rng):
    """One alert, the lesson's main() flow; the audit log grows as a side effect."""
    agent.call("promql", {"query": "rate(http_requests_total{status=~'5..'}[5m])"})
    agent.call("logql", {"query": '{app="checkout-api"} |~ "stack"'})
    if alerted.startswith("Pod/"):
        agent.call("kubectl_exec", {"pod": alerted, "cmd": "cat /proc/meminfo"})
    tool = remedy_for(ref.root_cause(agent.graph, alerted)[0].title)
    agent.call(tool, {"target": alerted})
    if rng.random() < 0.4:
        return agent.call(tool, {"target": alerted}, approver="oncall@sre")
    return None


def week(ref, seed=19062):
    rng = random.Random(seed)
    agent = ref.Agent(graph=ref.build_sample_cluster())
    alerts, approved = 0, []
    for _day in range(7):
        for _ in range(rng.randint(3, 9)):
            alerted = "Deployment/checkout-api" if rng.random() < 0.7 else "Pod/checkout-api-abc-0"
            approved.append(handle(ref, agent, alerted, rng))
            alerts += 1
    return agent, alerts, [ev for ev in approved if ev is not None]


def near_miss(ev):
    """Destructive by effect, and not approved: the audit this exercise adds."""
    return any(v in ev.tool for v in VERBS) and not ev.approved


def rates(log, flagged, dropped, alerts):
    true = len(flagged) - len(dropped)
    return {"considered": len(log), "flagged": len(flagged), "true": true,
            "rate": round(100 * true / len(log), 1), "per_alert": round(true / alerts, 2),
            "flagged_executed": sum(ev.executed for ev in flagged)}


def list_view(log, flagged, listed):
    """What an audit keyed on the lesson's destructive_tools would report."""
    by_list = [ev for ev in log if ev.tool in listed and not ev.approved]
    missed = [ev for ev in flagged if ev.tool not in listed]
    return {"by_list": len(by_list), "list_rate": round(100 * len(by_list) / len(log), 1),
            "unlisted": sorted({ev.tool for ev in missed}), "unlisted_result": sorted({ev.result for ev in missed})}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    agent, alerts, human_ok = week(ref)
    log = agent.audit
    flagged = [ev for ev in log if near_miss(ev)]
    dropped = [ev for ev in human_ok if near_miss(ev)]  # a human approved it; the log says no
    probe = ref.Agent(graph=agent.graph)
    self_ok = [probe.call(t, {}, approver="devops-agent").executed for t in probe.destructive_tools]
    return {
        "alerts": alerts, **rates(log, flagged, dropped, alerts), **list_view(log, flagged, agent.destructive_tools),
        "human_ok": len(human_ok), "dropped": len(dropped), "dropped_tools": sorted({ev.tool for ev in dropped}),
        "dropped_approver": [ev.approver for ev in dropped],
        "self_approved": sum(self_ok), "considered_true": sum(ev.considered for ev in log),
    }


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: 46 of 142 considered commands over the week are near-misses (32.4%)",
            (r["alerts"], r["considered"], r["flagged"], r["true"], r["rate"], r["per_alert"])
            == (39, 142, 49, 46, 32.4, 1.18) and r["flagged_executed"] == 0,
            f"{r['alerts']} alerts, {r['considered']} considered, {r['flagged']} flagged by the log, "
            f"{r['true']} true near-misses ({r['rate']}%, {r['per_alert']}/alert)",
        ),
        practice.Check(
            "FINDING: the lesson's destructive_tools list sees 32 of the 46 near-misses",
            (r["by_list"], r["list_rate"], r["unlisted"], r["unlisted_result"])
            == (32, 22.5, ["kubectl_drain", "kubectl_exec"], ["blocked: unknown tool"]),
            f"list-based {r['by_list']} ({r['list_rate']}%); missed tools {r['unlisted']} "
            f"logged as {r['unlisted_result']}",
        ),
        practice.Check(
            "FINDING: a human-approved drain is logged as unapproved with no approver",
            (r["human_ok"], r["dropped"], r["dropped_tools"], r["dropped_approver"])
            == (18, 3, ["kubectl_drain"], [None, None, None]),
            f"{r['dropped']} of {r['human_ok']} human approvals lost ({r['dropped_tools']}), "
            f"approver recorded as {r['dropped_approver']}",
        ),
        practice.Check(
            "FINDING: any approver string executes, and 'considered' is always True",
            (r["self_approved"], r["considered_true"]) == (4, r["considered"]),
            f"approver='devops-agent' executes {r['self_approved']}/4 destructive tools; "
            f"considered=True on {r['considered_true']}/{r['considered']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
