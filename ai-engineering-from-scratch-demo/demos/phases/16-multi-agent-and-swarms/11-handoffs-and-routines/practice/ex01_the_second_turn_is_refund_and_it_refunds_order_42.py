"""Exercise 1 — the second turn is refund, and it refunds order 42.

    Run `code/main.py`, triage to the refund agent. Confirm the second turn's
    active agent is refund.

Reading of the exercise: the demo's four scenarios are one message each, so
none has a second turn to confirm anything on; the refund flow is re-run with
a second (and third) user message, and the check is on who answers *and* what
they do.

**ANSWER: yes -- the second turn's active agent is refund.** Turn 1, "I need a
refund on order 77", produces `(handoff to refund)` from triage and then
"Refund processed for order 77." from refund. Turn 2, "thanks!", is answered
by `sender='refund'` with no handoff in between: `run_swarm` carries `active`
across the loop.

**FINDING: the second turn issues a second refund, on an order nobody named.**
The refund router has one output -- it calls `process_refund` on every turn --
and when the message has no bare-digit word it falls back to `order = "42"`.
So "thanks!" prints "Refund processed for order 42." And "order #77." is not
a digit word either: the first-turn refund goes to 42 as well.

**FINDING: a handoff is permanent.** 3 of the 4 agents have no handoff tool,
so nothing can leave a specialist. Turn 3, "actually I want to buy the pro
plan", is still answered by refund, with a third refund on order 42 -- the
sales keywords the triage agent would have routed are never read again.

**FINDING: the instructions are read by nothing.** `scripted_router` branches
on `current.name`; 0 of the 4 agents' `instructions` strings appear in any
code path. The demo's closing line, "The agent prompts ARE the routing
logic", describes an LLM this module does not have -- here the routing is the
router's `if` chain, and the prompts are comments.

Structure: `turns()` runs the reference `run_swarm` and pairs each user
message with the agent that answered it and what it said.
"""

from __future__ import annotations

import inspect
import re

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "11-handoffs-and-routines"
SESSION = ["I need a refund on order 77", "thanks!", "actually I want to buy the pro plan"]


def turns(ref, messages):
    """[(user message, answering agent, answer)] for one run_swarm session."""
    history, out, user = ref.run_swarm(ref.triage_agent, messages), [], None
    for msg in history:
        if msg.role == "user":
            user = msg.content
        elif not msg.content.startswith("(handoff"):
            out.append((user, msg.sender, msg.content))
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    agents = [ref.triage_agent, ref.refund_agent, ref.sales_agent, ref.support_agent]
    router = inspect.getsource(ref.scripted_router)
    main = inspect.getsource(ref.main)
    return {
        "session": turns(ref, SESSION),
        "hashed": turns(ref, ["I need a refund on order #77."])[0][2],
        "handoffs": [sum(f.__name__.startswith("transfer_to_") for f in a.functions)
                     for a in agents],
        "reads_instructions": "instructions" in router,
        "branches_on_name": router.count("current.name =="),
        "scenario_lengths": [inner.count('", "') + 1
                             for inner in re.findall(r'\("[^"]+", \[([^\]]*)\]\)', main)],
    }


def verify(result):
    s = result["session"]
    return [
        practice.Check(
            "ANSWER: yes -- the second turn's active agent is refund",
            s[0][1] == "refund" and s[1][1] == "refund"
            and result["scenario_lengths"] == [1, 1, 1, 1],
            f"the demo's scenarios have {result['scenario_lengths']} messages, so none "
            f"reaches a second turn; re-run, turn 1 {s[0][0]!r} -> {s[0][1]}: {s[0][2]!r}; turn 2 {s[1][0]!r} -> "
            f"{s[1][1]} with no handoff between, since run_swarm carries active across turns",
        ),
        practice.Check(
            "FINDING: the second turn issues a second refund, on an order nobody named",
            s[1][2] == "Refund processed for order 42." and "order 42" in result["hashed"],
            f"'thanks!' gets {s[1][2]!r} -- process_refund runs every turn with order "
            f"defaulting to '42' -- and 'order #77.' gets {result['hashed']!r}",
        ),
        practice.Check(
            "FINDING: a handoff is permanent",
            result["handoffs"] == [3, 0, 0, 0] and s[2][1] == "refund" and "42" in s[2][2],
            f"handoff tools per agent {result['handoffs']}; turn 3 {s[2][0]!r} is still "
            f"answered by {s[2][1]}: {s[2][2]!r}",
        ),
        practice.Check(
            "FINDING: the instructions are read by nothing",
            not result["reads_instructions"] and result["branches_on_name"] == 4,
            f"scripted_router has {result['branches_on_name']} branches on current.name "
            "and never reads instructions -- the prompts the demo calls the routing "
            "logic are comments",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
