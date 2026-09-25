"""Exercise 4 — the three-round cap is never reached.

    **Protocol negotiation.** Implement ANP's meta-protocol concept. Two agents
    exchange `protocolNegotiation` messages with candidate formats (e.g., "I
    can speak JSON-RPC" vs "I prefer REST"). After max 3 rounds, they agree on
    a format or timeout. The agreed format determines which `TaskManager` or
    `AuditableRunner` they use.

Reading of the exercise: implement it, then sweep every preference pair it
could be run on -- because the specification's one free parameter is the round
cap, and over the whole space that cap never binds.

**ANSWER: implemented, and the cap decides nothing.** Over all **4096**
ordered pairs of preference lists across four formats, **3964** pairs share at
least one format and **every one of them agrees**, in **1** round for 3796 and
**2** for 168. The maximum is **2**. The other **132** share no format and
would still fail after 3 rounds, or 30. What separates agreement from timeout
is whether the capability sets intersect -- which both agents could tell each
other in one message.

**FINDING: the cap is one more than the worst case that exists.** No pair
takes **3** rounds, so raising or lowering the limit to 3 changes the outcome
for **0** of the 4096 pairs. A one-shot exchange of whole capability sets
settles all **4096** in **1** round and reports impossibility immediately
rather than after three silences, which is strictly better on both axes.

**FINDING: nothing in the module consumes a negotiated format.** The agreed
format is supposed to select a `TaskManager` or an `AuditableRunner`.
`AgentCard.capabilities.streaming` is declared **3** times and read **0**
times, and `delegateTask` contains **0** branches on any capability -- there
is no switch for a negotiation to set.

**FINDING: the gateway already runs both, on every delegation.**
`delegateTask` calls `auditRunner.run(targetAgent, [message], sessionId)` and
then `taskManager.sendMessage(targetAgent, message)`. Both execute the target
agent. The port counts **2** executions per delegation, through two registries
that share no state, and returns a `task` and an `audit` describing different
runs of the same request. A negotiated format cannot pick one path when the
code takes both.

Structure: `negotiate()` is the round-by-round protocol; `sweep()` runs it
over every preference pair; `Gateway` ports `delegateTask`.
"""

from __future__ import annotations

import collections
import itertools
import re

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "03-communication-protocols"
FORMATS = ("json-rpc", "rest", "grpc", "sse")
CAP = 3


def orderings():
    """Every non-empty capability set paired with every order of preference."""
    return [order for size in range(1, len(FORMATS) + 1)
            for subset in itertools.combinations(FORMATS, size)
            for order in itertools.permutations(subset)]


def negotiate(mine, theirs, cap=CAP):
    """One candidate each per round; agree when either offer is speakable."""
    ours, yours = set(mine), set(theirs)
    for index in range(cap):
        if index < len(mine) and mine[index] in yours:
            return index + 1, mine[index]
        if index < len(theirs) and theirs[index] in ours:
            return index + 1, theirs[index]
    return None, None


def one_shot(mine, theirs):
    """The alternative: exchange whole sets once, intersect, order by the caller."""
    shared = [name for name in mine if name in set(theirs)]
    return (1, shared[0]) if shared else (1, None)


def sweep(pairs):
    """Rounds to agreement for every preference pair that shares a format."""
    rounds, disjoint, shot = collections.Counter(), 0, collections.Counter()
    for mine, theirs in pairs:
        taken, _ = negotiate(mine, theirs)
        shot[one_shot(mine, theirs)[0]] += 1
        if set(mine) & set(theirs):
            rounds[taken] += 1
        else:
            disjoint += 1
    return rounds, disjoint, shot


class Gateway:
    """delegateTask, ported: the audit runner and the task manager both run the agent."""

    def __init__(self, handler):
        self.handler, self.runs = handler, []

    def _audit_run(self, agent, message):
        self.runs.append(("audit", agent))
        return self.handler(message)

    def _send_message(self, agent, message):
        self.runs.append(("task", agent))
        return self.handler(message)

    def delegate(self, agent, message):
        audit = self._audit_run(agent, message)
        task = self._send_message(agent, message)
        return {"audit": audit, "task": task}


def doubled():
    """How many times one delegation executes the target agent."""
    executions = []
    gateway = Gateway(lambda message: executions.append(message) or len(executions))
    result = gateway.delegate("researcher", "summarise the filing")
    return {"executions": len(executions), "paths": len(set(gateway.runs)),
            "audit_is_task": result["audit"] == result["task"]}


def solve():
    src = (parity.lesson_dir(PHASE, LESSON) / "code" / "main.ts").read_text("utf-8")
    start = src.index("  async delegateTask(")
    delegate = src[start:src.index("  discoverAndDelegate", start)]
    orders = orderings()
    rounds, disjoint, shot = sweep(itertools.product(orders, orders))
    return {
        **doubled(),
        "orders": len(orders), "pairs": len(orders) ** 2, "disjoint": disjoint,
        "overlapping": sum(rounds.values()), "distribution": dict(sorted(rounds.items())),
        "worst": max(rounds), "cap": CAP, "one_shot": dict(shot),
        "at_cap": rounds.get(CAP, 0), "timed_out": rounds.get(None, 0),
        "streaming_declared": src.count("streaming:"),
        "streaming_read": len(re.findall(r"\.streaming\b", src)),
        "capability_branches": len(re.findall(r"if \(.*capabilit", delegate)),
        "calls_audit": "auditRunner.run" in delegate,
        "calls_task": "taskManager.sendMessage" in delegate,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: implemented, and the cap decides nothing",
            all([result["pairs"] == 4096, result["overlapping"] == 3964,
                 result["disjoint"] == 132, result["timed_out"] == 0,
                 result["distribution"] == {1: 3796, 2: 168}]),
            f"over all {result['pairs']} ordered preference pairs "
            f"{result['overlapping']} share a format and every one agrees -- "
            f"{result['distribution'][1]} in one round, {result['distribution'][2]} in two "
            f"-- while {result['disjoint']} share none and fail after any number",
        ),
        practice.Check(
            "FINDING: the cap is one more than the worst case that exists",
            all([result["worst"] == 2, result["at_cap"] == 0,
                 result["one_shot"] == {1: result["pairs"]}]),
            f"no pair takes {result['cap']} rounds, so the limit changes the outcome for "
            f"{result['at_cap']} of {result['pairs']} pairs; exchanging whole capability "
            f"sets settles all {result['pairs']} in one round and reports impossibility "
            "immediately rather than after three silences",
        ),
        practice.Check(
            "FINDING: nothing in the module consumes a negotiated format",
            all([result["streaming_declared"] == 3, result["streaming_read"] == 0,
                 result["capability_branches"] == 0]),
            f"capabilities.streaming is declared {result['streaming_declared']} times and "
            f"read {result['streaming_read']}, and delegateTask has "
            f"{result['capability_branches']} branches on any capability -- there is no "
            "switch for a negotiation to set",
        ),
        practice.Check(
            "FINDING: the gateway already runs both, on every delegation",
            all([result["calls_audit"], result["calls_task"],
                 result["executions"] == 2, result["paths"] == 2]),
            f"delegateTask calls auditRunner.run then taskManager.sendMessage, both of "
            f"which execute the target agent: the port counts {result['executions']} "
            f"executions over {result['paths']} registries per delegation, so task and "
            "audit describe different runs of the same request",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
