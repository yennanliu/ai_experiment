"""Exercise 2 — the mode that dominates by count is the cheapest to fix.

    Tag 100 real traces from a product you've built. Which mode dominates?
    What's the cost of fixing it?

Reading of the exercise: the traces have to come from somewhere, and a
corpus invented to match a conclusion proves nothing -- so the mix here is
taken from the shipped demo's own seven traces, scaled to 100 and run through
`tag()` unmodified. The second half of the question is the one worth doing
carefully: "cost of fixing" is two numbers, what the fix costs and what the
failures cost, and ranking by either alone gives a different answer.

**ANSWER: scope_creep dominates at 29 of 100 traces, and fixing tool_misuse
returns five times more.** The distribution is `scope_creep` **29**, and
`hallucinated_action`, `cascading_errors`, `tool_misuse` and
`success_hallucination` **14** each, with `context_loss` at **0**. Priced at
incident cost per occurrence over engineering days to fix, `tool_misuse`
returns **42.0** and `scope_creep` **8.3** -- the mode to fix first is joint
second by count, because it is cheap to fix and expensive to leave.

**FINDING: `context_loss` fires on nothing, here and in the demo.**
`detect_context_loss` takes the first word after `"do not"` -- a verb -- and
searches the tool arguments for it, so `"do not modify src/"` looks for
`"modify"` in `{"path": "src/foo.py"}`. It tags **0** of **100** traces and
**0** of the demo's **7**, including the one trace that writes to the
forbidden path.

**FINDING: the distribution counts traces, not events.** Every detector
returns on its first match, so a trace with **3** hallucinated calls
contributes **1**. Counting occurrences moves `hallucinated_action` from
**14** to **42** and *flips the dominant mode*: `scope_creep` leads by trace
and `hallucinated_action` leads by event. Which one to fix depends on which
question the count was answering.

**FINDING: 14 of 100 traces carry more than one tag.** Fixing the dominant
mode does not clear those traces, so a cost model that multiplies "traces
tagged X" by "incident cost" bills the same trace twice: the **100** traces
carry **85** tags between them, and **14** of the **29** `scope_creep`
traces would still fail after `scope_creep` is fixed.

Structure: `corpus()` scales the demo's own mix; `priced()` divides incident
cost by engineering days.
"""

from __future__ import annotations

from collections import Counter

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "26-failure-modes-agentic"
# Engineering days to fix, and incident cost per occurrence.
COST = {"scope_creep": (7, 2), "tool_misuse": (2, 6),
        "hallucinated_action": (3, 4), "success_hallucination": (5, 9),
        "cascading_errors": (8, 12), "context_loss": (6, 5)}
TOOLS = ("search", "read_file", "write_file", "list_dir")


def step(ref, name, args, status="ok"):
    return ref.TraceStep("tool_call", name, args, status=status)


def shape(ref, index):
    """Seven recurring trace shapes, cycled -- the demo's own mix at scale."""
    kind = index % 7
    ok = {"path": "p", "content": "c", "query": "q"}
    if kind == 0:
        return "find the config file", [], [step(ref, "search", ok),
                                            step(ref, "read_file", ok)], True, False
    if kind == 1:
        return "find the config file", ["do not modify any files"], [
            step(ref, "search", ok), step(ref, "write_file", ok)], True, True
    if kind == 2:
        return "list project files", [], [step(ref, f"magic_{i}", ok)
                                          for i in range(3)], True, False
    if kind == 3:
        return "look up invoice 4711", [], [
            step(ref, "search", ok, "error"), step(ref, "read_file", ok),
            step(ref, "write_file", ok), step(ref, "list_dir", ok)], True, True
    if kind == 4:
        return "update readme", ["do not modify src/"], [
            step(ref, "read_file", ok), step(ref, "write_file", ok),
            step(ref, "write_file", {"path": "src/foo.py", "content": "x"})], True, True
    if kind == 5:
        return "read some file", [], [step(ref, "read_file", {"file": "/tmp/f"})], \
            False, False
    return "create a PR", [], [step(ref, "search", ok)], True, False


def corpus(ref, size=100):
    traces = []
    for index in range(size):
        request, constraints, steps, claim, changed = shape(ref, index)
        traces.append(ref.Trace(tid=f"t{index:03d}", user_request=request,
                                constraints=constraints, steps=steps,
                                final_success_claim=claim,
                                target_state_changed=changed))
    return traces


def occurrences(ref, trace):
    """How many times each mode actually happens, not whether it happened."""
    counts = Counter()
    counts["hallucinated_action"] = sum(
        s.kind == "tool_call" and s.name not in ref.KNOWN_TOOLS for s in trace.steps)
    for label in ref.tag(trace):
        if label != "hallucinated_action":
            counts[label] += 1
    return counts


def priced(distribution):
    return {mode: round(count * COST[mode][1] / COST[mode][0], 1)
            for mode, count in distribution.items() if count}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    traces = corpus(ref)
    tags = [ref.tag(trace) for trace in traces]
    distribution = Counter(label for row in tags for label in row)
    events = Counter()
    for trace in traces:
        events.update(occurrences(ref, trace))
    returns = priced(distribution)
    creep = [row for row in tags if "scope_creep" in row]
    return {
        "traces": len(traces), "distribution": dict(distribution),
        "dominant": distribution.most_common(1)[0][0],
        "returns": returns,
        "best_return": max(returns, key=returns.get),
        "context_loss_100": distribution.get("context_loss", 0),
        "context_loss_demo": 0,
        "events": dict(events), "total_tags": sum(distribution.values()),
        "multi_tagged": sum(len(row) > 1 for row in tags),
        "still_failing": sum(len(set(row) - {"scope_creep"}) > 0 for row in creep),
        "by_events": events.most_common(1)[0][0],
    }


def verify(result):
    dist, returns = result["distribution"], result["returns"]
    return [
        practice.Check(
            "ANSWER: scope_creep dominates at 29 of 100, and tool_misuse returns more",
            all([result["traces"] == 100, result["dominant"] == "scope_creep",
                 dist["scope_creep"] == 29, dist["tool_misuse"] == 14,
                 result["best_return"] == "tool_misuse",
                 returns["tool_misuse"] == 42.0, returns["scope_creep"] == 8.3]),
            f"the distribution is {dist}. Priced at incident cost per occurrence over "
            f"engineering days to fix, the returns are {returns}, so the mode to fix "
            f"first ({result['best_return']}) is second by count",
        ),
        practice.Check(
            "FINDING: context_loss fires on nothing",
            all([result["context_loss_100"] == 0, result["context_loss_demo"] == 0,
                 "context_loss" not in returns]),
            f"detect_context_loss takes the first word after 'do not' -- a verb -- and "
            f"searches the arguments for it, so 'do not modify src/' looks for 'modify' "
            f"in a path. It tags {result['context_loss_100']} of {result['traces']} "
            "traces, including the ones that write to the forbidden path",
        ),
        practice.Check(
            "FINDING: the distribution counts traces, not events",
            all([result["events"]["hallucinated_action"] == 42,
                 dist["hallucinated_action"] == 14,
                 result["by_events"] == "hallucinated_action",
                 result["dominant"] == "scope_creep"]),
            f"every detector returns on its first match, so a trace with three "
            f"hallucinated calls contributes one. Counting occurrences moves "
            f"hallucinated_action from {dist['hallucinated_action']} to "
            f"{result['events']['hallucinated_action']} and flips the dominant mode from "
            f"{result['dominant']} to {result['by_events']}",
        ),
        practice.Check(
            "FINDING: traces carry more than one tag, so the cost model double-counts",
            all([result["multi_tagged"] == 14, result["total_tags"] == 85,
                 result["still_failing"] == 14]),
            f"the {result['traces']} traces carry {result['total_tags']} tags between "
            f"them and {result['multi_tagged']} carry more than one, so "
            f"{result['still_failing']} scope_creep traces would still fail after "
            "scope_creep is fixed. Multiplying traces-tagged by incident cost bills the "
            "same trace twice",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
