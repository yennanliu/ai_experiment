"""Exercise 2 — the fallback cannot be triage, because triage is half the loop.

    Add a loop-detection rule: if the same two agents have handed off 3 times
    in a row, force an exit. Design the fallback.

Reading of the exercise: the shipped agents cannot loop -- no specialist has a
handoff -- so the refund agent is given the one it most plausibly needs, "hand
back to triage when there is no order number". Then "in a row" is read two
ways, within one user turn and across turns, and each fallback is kept only
if the run actually terminates.

**ANSWER: count handoffs between one unordered pair within a single user turn;
on the third, exit to an agent outside the pair that has no handoffs and asks
for what was missing.** "I want a refund" loops: triage routes on "refund",
refund has no order number and hands back, triage reads the same message and
routes again. With no rule the turn hits the 50-hop cap. With the rule it
stops after 3 handoffs and the fallback asks for the order number -- the one
thing that would have ended the loop, since the loop exists because no new
information arrives.

**FINDING: the natural fallback re-enters the loop.** The lesson's checklist
says "fall back to a safe default", and the safe default of a triage topology
is triage. Triage is one of the pair: falling back to it resumes the same
ping-pong, the rule fires again on every subsequent hop, and the turn still
hits the 50-hop cap.

**FINDING: across turns, the rule fires on a customer, not a loop.** "refund
on order 77", then "hmm, something else", then "refund on order 78" is three
user-driven handoffs between the same two agents. A ring kept across turns
trips on it; reset per user turn, it does not. A user message is new
information, which is exactly what a ping-pong lacks.

**FINDING: the shipped run loop hides a second handoff by printing it.**
`run_swarm` calls the router at most twice per turn and stringifies the second
result. When the refund agent hands back, the reply shown to the user is
`Agent(name='triage', ...)` and `active` stays refund -- where the cookbook's
loop runs "while True" until there are no tool calls.

Structure: `route()` is the looping router; `run_turn()` is a cookbook-shaped
turn loop with an optional `pair_ring` detector and fallback; the shipped
`run_swarm` is re-run with `route` patched in as its router.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "11-handoffs-and-routines"
CAP, LIMIT = 50, 3


def build(ref):
    agents = {n: ref.Agent(n, f"{n} routine") for n in ("triage", "refund", "human")}

    def route(current, text):
        digits = [w for w in text.split() if w.isdigit()]
        if current.name == "triage":
            return agents["refund"] if "refund" in text else "What do you need help with?"
        if current.name == "refund":
            return f"Refund processed for order {digits[0]}." if digits else agents["triage"]
        return "Escalated: which order number should be refunded?"
    return agents, route


def pair_ring(handoffs, limit=LIMIT):
    """True when the last `limit` handoffs are all between one unordered pair."""
    tail = [frozenset(h) for h in handoffs[-limit:]]
    return len(tail) == limit and len(set(tail)) == 1


def run_turn(active, text, route, fallback=None, ring=None):
    """(final agent, reply, hops, fallbacks) for one user turn."""
    ring, fired = [] if ring is None else ring, 0
    for hop in range(CAP):
        out = route(active, text)
        if isinstance(out, str):
            return active.name, out, hop, fired
        ring.append((active.name, out.name))
        active = out
        if fallback is not None and pair_ring(ring):
            active, fired = fallback, fired + 1
    return active.name, None, CAP, fired


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    agents, route = build(ref)
    t, msg = agents["triage"], "I want a refund"
    session = ["refund on order 77", "hmm, something else", "refund on order 78"]
    shared, fresh, active = [], [], t
    for text in session:
        active_name, _, _, fired = run_turn(active, text, route, agents["human"], shared)
        fresh.append(run_turn(active, text, route, agents["human"])[3])
        shared_fired, active = fired, agents[active_name]
    original = ref.scripted_router
    ref.scripted_router = route
    try:
        shipped = ref.run_swarm(t, [msg])
    finally:
        ref.scripted_router = original
    return {
        "none": run_turn(t, msg, route), "human": run_turn(t, msg, route, agents["human"]),
        "triage": run_turn(t, msg, route, t),
        "shared_fired": shared_fired, "fresh_fired": sum(fresh),
        "shipped_reply": shipped[-1].content[:30], "shipped_sender": shipped[-1].sender,
    }


def verify(result):
    none, human, triage = result["none"], result["human"], result["triage"]
    return [
        practice.Check(
            "ANSWER: per-turn pair ring, fallback outside the pair that asks",
            none[2] == CAP and human[:3] == ("human", human[1], LIMIT) and "order" in human[1],
            f"'I want a refund' runs {none[2]} hops with no rule; with it the turn stops "
            f"after {human[2]} handoffs at {human[0]}: {human[1]!r}",
        ),
        practice.Check(
            "FINDING: the natural fallback re-enters the loop",
            triage[1] is None and triage[2] == CAP and triage[3] > 10,
            f"falling back to triage, one of the pair, the rule fires {triage[3]} times "
            f"and the turn still hits the {CAP}-hop cap",
        ),
        practice.Check(
            "FINDING: across turns, the rule fires on a customer, not a loop",
            result["shared_fired"] == 1 and result["fresh_fired"] == 0,
            f"three user-driven triage/refund handoffs over three turns: a ring kept "
            f"across turns fires {result['shared_fired']}x, one reset per turn "
            f"{result['fresh_fired']}x",
        ),
        practice.Check(
            "FINDING: the shipped run loop hides a second handoff by printing it",
            result["shipped_reply"].startswith("Agent(name='triage'")
            and result["shipped_sender"] == "refund",
            f"run_swarm stringifies the second router result: the user sees "
            f"{result['shipped_reply']!r}... from {result['shipped_sender']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
