"""Exercise 4 — the fifth guest is turned away and never tries again.

    Add spatial constraints: Hobbs Cafe holds at most 4 agents. Does the
    simulation handle overflow gracefully, or does it hit the "single-person
    bathroom" failure pattern?

Reading of the exercise: first ask the unmodified reference, then add the
smallest capacity rule -- refuse entry when 4 are inside, and tell the agent
-- and check whether anything in the architecture lets a refused agent
recover, which is what "gracefully" would mean.

**ANSWER: unmodified, it hits the bathroom pattern exactly: 5 agents enter a
4-agent cafe at tick 5, with no error and no observation.** A location is a
bare string; the word "capacity" appears nowhere in the module. Park et al.
describe the same failure -- agents entering a one-person dorm bathroom while
someone was inside, because the norm "did not percolate to the agents" --
and suggest writing the norm into the location's state.

**FINDING: with the norm enforced, overflow is not graceful -- it is
permanent.** Admit 4, refuse agent-4 and record "HobbsCafe was full" in its
memory: it is still at home at tick 11. `act` fires a plan only when
`p.tick == tick`, and `update_plan` adds a cafe plan only if none exists, so
a missed appointment can never be rescheduled.

**FINDING: who is refused is loop order, and it can be the host.** Admission
goes to whoever the `for a in agents` loop visits first. Visit the agents in
reverse and agent-0 -- the one who seeded the party -- is the one turned
away from its own party.

**FINDING: "wait outside and retry" never succeeds, because nobody leaves.**
Give the refused agent a fresh plan for the next tick after each refusal and
it tries 7 times in 7 ticks and gets in 0 times. `act` only ever moves an
agent *to* a plan's place; a `Plan` is an instant, not an interval, so no
guest can leave and free a seat. Handling overflow needs departure times,
which the plan model cannot express.

Structure: `run()` replays run_simulation's invitations over reference
`Agent`s with a capacity-checking act; `order` and `retry` are the two knobs.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import re

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "17-generative-agents-simulation"
CAFE, CAPACITY, TICKS = "HobbsCafe", 4, 12
TEXT = "agent-{} invited me to a party at HobbsCafe at tick 5"
SHIPPED = {0: [(0, 1, 8), (0, 2, 8)], 1: [(1, 3, 7)], 2: [(2, 4, 7)]}


def reference_crowd(ref):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ref.run_simulation()
    return len(re.findall(r"agent-\d+\s+at HobbsCafe", out.getvalue()))


def gated_act(ref, agent, tick, inside, log, retry):
    for plan in agent.plans:
        if plan.tick != tick:
            continue
        if plan.where == CAFE and len(inside) >= CAPACITY:
            log.append((tick, agent.name))
            agent.observe(tick, "HobbsCafe was full", importance=6)
            if retry:
                agent.plans.append(ref.Plan(tick=tick + 1, where=CAFE, note="retry"))
            return
        agent.location = plan.where
        inside.add(agent.name) if plan.where == CAFE else inside.discard(agent.name)
        return


def seeded(ref):
    agents = [ref.Agent(f"agent-{i}", location="home") for i in range(5)]
    agents[0].plans.append(ref.Plan(tick=5, where=CAFE, note="host the party"))
    agents[0].beliefs.append("there is a party I was invited to")
    return agents


def run(ref, order=range(5), retry=False):
    agents, inside, log = seeded(ref), set(), []
    for tick in range(TICKS):
        for src, dst, importance in SHIPPED.get(tick, []):
            agents[dst].observe(tick, TEXT.format(src), importance)
        for agent in (agents[i] for i in order):
            agent.reflect(tick)
            agent.update_plan(tick)
            gated_act(ref, agent, tick, inside, log, retry)
    return {"inside": sorted(inside), "refused": log, "home": at_home(agents),
            "remembers": sum(m.content == "HobbsCafe was full" for a in agents for m in a.stream)}


def at_home(agents):
    return sorted(a.name for a in agents if a.location == "home")


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    source = inspect.getsource(ref)
    return {
        "crowd": reference_crowd(ref), "capacity_named": "capacity" in source.lower(),
        "gated": run(ref), "reversed": run(ref, order=range(4, -1, -1)),
        "retry": run(ref, retry=True),
    }


def verify(result):
    gated, rev, retry = result["gated"], result["reversed"], result["retry"]
    return [
        practice.Check(
            "ANSWER: unmodified, 5 agents enter a 4-agent cafe with no error",
            result["crowd"] == 5 and not result["capacity_named"],
            f"the reference ends with {result['crowd']} agents at HobbsCafe; 'capacity' "
            "appears nowhere in the module -- a location is a bare string",
        ),
        practice.Check(
            "FINDING: with the norm enforced, overflow is permanent",
            gated["refused"] == [(5, "agent-4")] and gated["home"] == ["agent-4"]
            and gated["remembers"] == 1,
            f"refusals {gated['refused']}; at tick {TICKS - 1} {gated['home']} is still "
            "home with the refusal in memory -- act needs p.tick == tick and update_plan "
            "never adds a second cafe plan",
        ),
        practice.Check(
            "FINDING: who is refused is loop order, and it can be the host",
            rev["refused"] == [(5, "agent-0")],
            f"visiting agents in reverse, the refusal is {rev['refused']} -- the host, "
            "turned away from its own party",
        ),
        practice.Check(
            "FINDING: wait-and-retry never succeeds, because nobody leaves",
            len(retry["refused"]) == TICKS - 5 and retry["home"] == ["agent-4"],
            f"retrying every tick, agent-4 is refused {len(retry['refused'])} times in "
            f"ticks 5-{TICKS - 1} and admitted 0 times; act only moves agents to a plan's "
            "place, so no seat is ever freed",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
