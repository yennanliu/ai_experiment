"""Exercise 3 — the summary must keep what the next agent's tools take.

    Read the OpenAI Agents SDK docs on handoff filters. Implement a
    "summarize-on-handoff" version: the outgoing agent compresses context to a
    bullet summary before the incoming agent takes over.

Reading of the exercise: a summary is only right or wrong relative to what the
incoming agent needs next, so the test conversation puts the order number in a
turn *before* the handoff -- "hi, the blender I got last week arrived damaged,
order 77", then "I want my money back" -- and each context policy is scored on
which order the refund agent refunds.

**ANSWER: derive the bullets from the incoming agent's tool signatures.** The
SDK's hook is `handoff(agent, input_filter=...)`, a function over
`HandoffInputData` whose default is to pass "the entire previous conversation
history". A signature-driven filter reads `process_refund(order_id)` from the
target, extracts `order_id` from the history and hands over one bullet,
`- order_id: 77`: the refund agent refunds 77, from 14 characters of context
against 57 for the full history. A generic bullet summary -- each user turn cut
to its first 5 words -- keeps "hi, the blender I got" and refunds 42.

**FINDING: on the shipped code every filter is a no-op, because no agent reads
context.** `scripted_router(current, user_msg)` takes the current message and
nothing else, and `run_swarm`'s `history` is written and never passed in. The
reference run of the same two turns refunds order 42: 0 bytes of the
conversation reach the incoming agent, whatever policy is chosen.

**FINDING: the loss lands on the tool argument, silently.** Both lossy runs
still end in "Refund processed for order 42." -- a confident, successful tool
call. `process_refund` has a default to fall back on, so a dropped identifier
does not fail; it becomes a different identifier.

Structure: three context policies -- `full`, `bullets`, `signature` -- feed a
refund step that reads context plus the current message; the signature policy
uses `inspect.signature` on the reference agent's own functions.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "11-handoffs-and-routines"
TURNS = ["hi, the blender I got last week arrived damaged, order 77", "I want my money back"]


def full(history, target):
    return "\n".join(history)


def bullets(history, target, words=5):
    return "\n".join("- " + " ".join(turn.split()[:words]) for turn in history)


def signature(history, target):
    """One bullet per parameter of the target's tools, filled from the history."""
    params = [p for f in target.functions for p in inspect.signature(f).parameters]
    digits = [w.strip(",.") for turn in history for w in turn.split() if w.strip(",.").isdigit()]
    return "\n".join(f"- {p}: {digits[-1]}" for p in params if p.endswith("_id") and digits)


def refund_step(ref, context, message):
    """The reference router, shown the transferred context before the message."""
    return ref.scripted_router(ref.refund_agent, " ".join((context + " " + message).split()))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shipped = ref.run_swarm(ref.triage_agent, TURNS)
    runs = {}
    for policy in (full, bullets, signature):
        context = policy(TURNS[:1], ref.refund_agent)
        runs[policy.__name__] = (len(context), refund_step(ref, context, TURNS[1]))
    return {
        "runs": runs, "shipped": shipped[-1].content,
        "handoff": [m.content for m in shipped if m.content.startswith("(handoff")],
        "router_params": list(inspect.signature(ref.scripted_router).parameters),
        "history_passed": "scripted_router(active, user)" in inspect.getsource(ref.run_swarm),
    }


def verify(result):
    runs = result["runs"]
    return [
        practice.Check(
            "ANSWER: derive the bullets from the incoming agent's tool signatures",
            all([runs["signature"][1].endswith("order 77."), runs["full"][1].endswith("order 77."),
                 runs["bullets"][1].endswith("order 42."),
                 runs["signature"][0] < runs["full"][0]]),
            f"signature bullets ({runs['signature'][0]} chars): {runs['signature'][1]!r}; "
            f"full history ({runs['full'][0]} chars): {runs['full'][1]!r}; 5-word bullets "
            f"({runs['bullets'][0]} chars): {runs['bullets'][1]!r}",
        ),
        practice.Check(
            "FINDING: on the shipped code every filter is a no-op",
            all([result["router_params"] == ["current", "user_msg"], result["history_passed"],
                 result["handoff"] == ["(handoff to refund)"],
                 result["shipped"] == "Refund processed for order 42."]),
            f"scripted_router takes {result['router_params']}; the reference run hands off "
            f"to refund and answers {result['shipped']!r} -- history never reaches it",
        ),
        practice.Check(
            "FINDING: the loss lands on the tool argument, silently",
            result["shipped"] == runs["bullets"][1] == "Refund processed for order 42.",
            "both lossy runs end in a successful refund of order 42: process_refund has a "
            "default, so a dropped identifier becomes a different identifier",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
