"""Exercise 1 — the invitations are scripted by index, so agents five to nine never hear.

    Run `code/main.py`. Confirm 3+ agents converge at the party. Increase agents
    to 10 — does the emergence still happen?

Reading of the exercise: "increase agents to 10" is `run_simulation(10)`, the
function's own parameter, and "does the emergence still happen" is asked of
the share that converges, not only the count -- then of an agent-driven
relay, to see what the architecture would do if the agents did the inviting.

**ANSWER: 5 of 5 converge at n=5; at n=10 it is 5 of 10 -- the same five.**
Agents 5-9 receive **0** observations. At n=25 it is 5 of 25, and n=3 or
n=4 raises `IndexError`. Nothing spreads, because the invitations are three
`if tick == ...` branches inside `run_simulation` naming agents 1, 2, 3 and 4
by index.

**FINDING: the simulation loop is the orchestrator.** `Agent` has 4 public
methods -- observe, reflect, update_plan, act -- and none takes another
agent, so no agent can invite anyone. "No orchestrator. One seed." is printed
by the only code that decides who hears what. Given a relay in which every
agent that holds the belief invites one uninvited agent per tick, through the
same `observe`, the reference `Agent` converges 10 of 10 and 25 of 25.

**FINDING: the party time is a literal, and it caps the reach at 64.**
`update_plan` appends `Plan(tick=5, ...)` whatever the invitation says: an
invitation "at tick 3" still plans tick 5. `act` fires only when
`p.tick == tick`, so `run_simulation(5, ticks=5)` converges 0 of 5. Under the
relay the knowers double each tick from the one seed -- 2, 4, 8, 16, 32, 64
by the end of tick 5 -- and anyone invited later can never attend: n=100
converges 64.

**FINDING: retrieval is never called.** `retrieve_top_k` -- one of the three
components the lesson builds -- appears in neither `run_simulation` nor
`Agent`. Its only caller is `demo_retrieval`, on a separate 4-memory stream.

Structure: `drive()` replays run_simulation's loop over reference `Agent`s
with a pluggable invitation schedule, and matches the reference's printed
final locations on the shipped schedule.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import re

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "17-generative-agents-simulation"
TEXT = "agent-{} invited me to a party at HobbsCafe at tick {}"
SHIPPED = {0: [(0, 1, 8), (0, 2, 8)], 1: [(1, 3, 7)], 2: [(2, 4, 7)]}


def reference(ref, n, ticks=6):
    """(converged, n) from the reference's own printout, or the exception name."""
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            ref.run_simulation(n, ticks)
    except Exception as exc:  # noqa: BLE001 - the exception is the finding
        return type(exc).__name__
    found = re.search(r"(\d+)/(\d+) agents converged", out.getvalue())
    return int(found.group(1)), int(found.group(2))


def relay(tick, agents):
    """Every agent holding the belief invites the next uninvited agent."""
    knowers = [i for i, a in enumerate(agents) if a.beliefs]
    fresh = iter(i for i, a in enumerate(agents) if not a.beliefs)
    return [(src, dst, 7) for src, dst in zip(knowers, fresh)]


def drive(ref, n, schedule, ticks=6, when=5):
    agents = [ref.Agent(f"agent-{i}", location="home") for i in range(n)]
    agents[0].plans.append(ref.Plan(tick=5, where="HobbsCafe", note="host the party"))
    agents[0].beliefs.append("there is a party I was invited to")
    for tick in range(ticks):
        for src, dst, importance in schedule(tick, agents):
            agents[dst].observe(tick, TEXT.format(src, when), importance)
        for agent in agents:
            agent.reflect(tick)
            agent.update_plan(tick)
            agent.act(tick)
    return agents


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    at = lambda agents: sum(a.location == "HobbsCafe" for a in agents)  # noqa: E731
    shipped = drive(ref, 10, lambda t, _: SHIPPED.get(t, []))
    early = drive(ref, 5, lambda t, _: SHIPPED.get(t, []), when=3)
    source = inspect.getsource(ref.run_simulation) + inspect.getsource(ref.Agent)
    return {
        "ref": {n: reference(ref, n) for n in (3, 4, 5, 10, 25)},
        "short": reference(ref, 5, ticks=5),
        "parity": at(shipped) == reference(ref, 10)[0],
        "unheard": sum(not a.stream for a in shipped[5:]),
        "methods": sorted(m for m in vars(ref.Agent) if not m.startswith("_")),
        "relay": {n: at(drive(ref, n, relay)) for n in (10, 25, 100)},
        "early_plan": sorted({p.tick for a in early for p in a.plans}),
        "retrieval_used": "retrieve_top_k" in source,
    }


def verify(result):
    ref, relayed = result["ref"], result["relay"]
    return [
        practice.Check(
            "ANSWER: 5 of 5 at n=5; at n=10 it is 5 of 10 -- the same five",
            all([ref[5] == (5, 5), ref[10] == (5, 10), ref[25] == (5, 25),
                 ref[3] == ref[4] == "IndexError", result["unheard"] == 5,
                 result["parity"]]),
            f"run_simulation converges {ref[5]}, {ref[10]}, {ref[25]} at n=5, 10, 25 and "
            f"raises {ref[3]} at n=3 and 4; agents 5-9 hold 0 memories "
            f"({result['unheard']} of 5 empty streams)",
        ),
        practice.Check(
            "FINDING: the simulation loop is the orchestrator",
            result["methods"] == ["act", "observe", "reflect", "update_plan"]
            and relayed[10] == 10 and relayed[25] == 25,
            f"Agent's methods are {result['methods']}, none taking another agent; with "
            f"an agent-driven relay through the same observe, {relayed[10]}/10 and "
            f"{relayed[25]}/25 converge",
        ),
        practice.Check(
            "FINDING: the party time is a literal, and it caps the reach at 64",
            result["early_plan"] == [5] and result["short"] == (0, 5)
            and relayed[100] == 64,
            f"an invitation 'at tick 3' plans ticks {result['early_plan']}; ticks=5 "
            f"converges {result['short']}; the doubling relay reaches {relayed[100]} of "
            "100 by the tick-5 deadline",
        ),
        practice.Check(
            "FINDING: retrieval is never called",
            not result["retrieval_used"],
            "retrieve_top_k appears in neither run_simulation nor Agent -- only "
            "demo_retrieval calls it, on a separate 4-memory stream",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
