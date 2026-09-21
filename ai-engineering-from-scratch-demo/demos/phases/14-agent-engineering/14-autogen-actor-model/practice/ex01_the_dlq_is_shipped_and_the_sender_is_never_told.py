"""Exercise 1 — the DLQ is shipped, and the sender is never told.

    Add a dead-letter queue: when a handler raises, park the failing message
    for human inspection. How often does DLQ get hit in your toy?

Reading of the exercise: `Runtime` already has `dead_letters`, already parks
a failing message, and already keeps the other actors running -- that is the
fault-isolation property the demo closes on. So the exercise is answered by
measuring the queue rather than building it, and the measurement is what
exposes what parking does *not* do.

**ANSWER: the demo hits the DLQ once in 8 messages, 12%.** The user sends
**2** and the actors generate the other **6** -- three `review` requests and
three `review_result` replies -- so the denominator is the whole message
count, not the caller's. One `crash_me` raises inside
`ReviewerAgent.receive` and is parked while **7** messages are handled.
Adding one message to an unregistered actor takes it to **2** of **9**,
**22%**, with the two entries distinguishable only by their reason strings.

**FINDING: the queue mixes two different failures.** `dead_letters` holds
`(message, reason)` pairs, and the reason is free text: `no actor 'ghost'`
for an addressing mistake and `RuntimeError: simulated handler failure` for a
handler fault. One is a routing bug and the other is a code bug; telling them
apart means parsing a string.

**FINDING: nobody is told.** Parking a message notifies neither the sender
nor a supervisor. A request that would have been answered by
`review_result` is simply never answered, and `ChecklistAgent` waits with
**2** of the **3** results it expects and `consensus` left at `None` -- a
hang that looks exactly like work still in flight.

**FINDING: `consensus` is set before the results are in.** On the demo's own
inputs `ChecklistAgent` assigns `consensus = True` as soon as every result
*so far* is ok, so after the first clean snippet it reads **True** with
**1** of **3** results. The final line corrects it to **False**, but any
reader between those two messages sees a verdict that has not been earned.

Structure: `run()` drives the lesson's own `Runtime` and returns what the DLQ
caught; nothing here reimplements the queue.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "14-autogen-actor-model"
SNIPPETS = ("def add(a, b): return a + b",
            "def hazard(): eval('1+1')",
            "def silent():\n    try:\n        f()\n    except:\n        pass")


def build(ref):
    runtime = ref.Runtime()
    runtime.register(ref.ReviewerAgent("reviewer"))
    checklist = ref.ChecklistAgent("checklist", partner="reviewer")
    runtime.register(checklist)
    return runtime, checklist


def run(ref, extra=()):
    runtime, checklist = build(ref)
    runtime.send("__user__", "checklist", "start", list(SNIPPETS))
    runtime.send("__user__", "reviewer", "crash_me", {})
    for recipient, topic in extra:
        runtime.send("__user__", recipient, topic, {})
    runtime.run_until_idle()
    return runtime, checklist


def early_consensus(ref):
    """The state of `consensus` after each review_result, one message at a time."""
    runtime, checklist = build(ref)
    runtime.send("__user__", "checklist", "start", list(SNIPPETS))
    seen = []
    while runtime.queue:
        message = runtime.queue.popleft()
        runtime.actors[message.recipient].receive(message, runtime)
        if message.topic == "review_result":
            seen.append((len(checklist.results), checklist.consensus))
    return seen


def hung(ref):
    """A review whose reply never comes back, because the handler raised."""
    runtime, checklist = build(ref)
    runtime.send("__user__", "checklist", "start", list(SNIPPETS[:2]))
    runtime.send("__user__", "reviewer", "crash_me", {})
    runtime.send("checklist", "reviewer", "crash_me", {})
    runtime.run_until_idle()
    return len(checklist.results), checklist.consensus


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runtime, checklist = run(ref)
    wider, _ = run(ref, extra=(("ghost", "review"),))
    results, consensus = hung(ref)
    return {
        "sent": runtime.counter, "dlq": len(runtime.dead_letters),
        "rate": round(len(runtime.dead_letters) / runtime.counter, 2),
        "handled": sum(1 for line in runtime.trace if line.startswith("[recv")),
        "wider_sent": wider.counter, "wider_dlq": len(wider.dead_letters),
        "wider_rate": round(len(wider.dead_letters) / wider.counter, 2),
        "reasons": [reason for _, reason in wider.dead_letters],
        "hung_results": results, "hung_consensus": consensus,
        "early": early_consensus(ref),
        "final_consensus": checklist.consensus,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 1 of 8 in the demo, 2 of 9 once a bad address is added",
            all([result["sent"] == 8, result["dlq"] == 1, result["rate"] == 0.12,
                 result["wider_sent"] == 9, result["wider_dlq"] == 2,
                 result["wider_rate"] == 0.22, result["handled"] == 7]),
            f"the demo sends {result['sent']} messages and parks {result['dlq']} -- "
            f"{result['rate']:.0%} -- while {result['handled']} are handled. Adding one "
            f"message to an unregistered actor takes it to {result['wider_dlq']} of "
            f"{result['wider_sent']}, {result['wider_rate']:.0%}",
        ),
        practice.Check(
            "FINDING: the queue mixes two different failures",
            all([len(result["reasons"]) == 2,
                 any(reason.startswith("no actor") for reason in result["reasons"]),
                 any("RuntimeError" in reason for reason in result["reasons"])]),
            f"dead_letters holds (message, reason) pairs and the reason is free text: "
            f"{result['reasons']}. One is a routing mistake and the other a handler "
            "fault -- one belongs to whoever addressed the message and the other to "
            "whoever wrote the actor, and telling them apart means parsing a string",
        ),
        practice.Check(
            "FINDING: nobody is told",
            all([result["hung_results"] == 2, result["hung_consensus"] is True,
                 result["hung_results"] < 3]),
            f"parking a message notifies neither the sender nor a supervisor, so a "
            f"review that raised is never answered: the checklist waits with "
            f"{result['hung_results']} of 3 results and consensus "
            f"{result['hung_consensus']} -- a hang that looks exactly like work still "
            "in flight",
        ),
        practice.Check(
            "FINDING: consensus is set before the results are in",
            all([result["early"][0] == (1, True), result["early"][-1][0] == 3,
                 result["final_consensus"] is False]),
            f"ChecklistAgent assigns consensus = True as soon as every result so far is "
            f"ok, so after the first clean snippet it reads {result['early'][0][1]} with "
            f"{result['early'][0][0]} of 3 results. The last message corrects it to "
            f"{result['final_consensus']}, and any reader in between sees a verdict that "
            "has not been earned",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
