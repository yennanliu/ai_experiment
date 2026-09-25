"""Exercise 3 — the talk cannot spread, and when it can, the first invitation wins.

    Introduce a competing seeded goal ("Klaus wants to give a research talk at
    5pm"). Do agents split, or does one goal dominate? What determines it?

Reading of the exercise: Klaus is agent-4 -- the last guest the shipped
script invites to the party -- seeded with his own goal and a plan for the
Library at tick 5, inviting agent-3 at tick 0 and agent-1 at tick 1. The
question is run twice: on the reference `Agent`, and on one whose reflection
can hold more than one event, so that what decides the outcome is visible.

**ANSWER: on the reference, the party dominates completely -- and not because
of anything the agents weigh.** Party 4, talk 1: Klaus alone. His invitations
say "research talk at Library", and `reflect` only forms a belief from a
memory containing "party at", so the talk can never become anyone's belief.
One goal dominates because of a substring in `reflect`.

**FINDING: with events generalised, agents split, and invitation order
decides.** Let reflection form one belief per invited place. Then agent-3,
invited to the talk at tick 0 and the party at tick 1, goes to the Library;
agent-1, invited the other way round, goes to the cafe -- party 3, talk 2.
`act` moves an agent to the *first* plan whose tick matches, and plans are
appended in the order beliefs formed. Deliver each talk invitation one tick
after that guest's party invitation and the talk keeps only Klaus.

**FINDING: importance, the one signal the memory records, changes nothing.**
Talk invitations at importance 10 against party invitations at 6 give the
same 3-2 split: importance only gates reflection at 6, and retrieval, which
would weigh it, is never called. Klaus himself holds two tick-5 plans --
he was invited to the party at tick 2 -- and goes to his talk only because
his seeded plan sits first in the list.

Structure: `drive()` replays run_simulation's loop over any Agent class with
a list of invitations; `EventAgent` is the reference Agent with reflection
keyed by place.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "17-generative-agents-simulation"
PARTY = "agent-{} invited me to a party at HobbsCafe at tick 5"
TALK = "agent-{} invited me to a research talk at Library at tick 5"
KLAUS = 4


def invitations(talk_ticks, party=7, talk=9):
    party_invites = [(0, 0, 1, 8), (0, 0, 2, 8), (1, 1, 3, party), (2, 2, 4, party)]
    rows = [(t, s, d, PARTY, i) for t, s, d, i in party_invites]
    return rows + [(tick, KLAUS, dst, TALK, talk) for dst, tick in talk_ticks.items()]


def drive(cls, ref, invites):
    agents = [cls(f"agent-{i}", location="home") for i in range(5)]
    agents[0].plans.append(ref.Plan(tick=5, where="HobbsCafe", note="host the party"))
    agents[0].beliefs.append("there is a party I was invited to")
    agents[KLAUS].stream.append(ref.Memory(0, "goal", "give a research talk at Library", 10))
    agents[KLAUS].plans.append(ref.Plan(tick=5, where="Library", note="give the talk"))
    for tick in range(6):
        for _, src, dst, text, importance in (r for r in invites if r[0] == tick):
            agents[dst].observe(tick, text.format(src), importance)
        for agent in agents:
            agent.reflect(tick)
            agent.update_plan(tick)
            agent.act(tick)
    return {place: [i for i, a in enumerate(agents) if a.location == place]
            for place in ("HobbsCafe", "Library")}


def event_belief(memory, tick):
    """reflect's gate (importance 6+, last 5 ticks), keyed by the invited place."""
    place = re.search(r"invited me to .* at (\w+) at tick", memory.content)
    fresh = memory.importance >= 6 and tick - memory.ts <= 5
    return f"invited to an event at {place.group(1)}" if place and fresh else None


def event_agent(ref):
    class EventAgent(ref.Agent):
        def reflect(self, tick):
            for belief in filter(None, (event_belief(m, tick) for m in self.stream)):
                if belief not in self.beliefs:
                    self.beliefs.append(belief)

        def update_plan(self, tick):
            for belief in self.beliefs:
                where = belief.rsplit(" ", 1)[-1]
                if belief.startswith("invited") and all(p.where != where for p in self.plans):
                    self.plans.append(ref.Plan(tick=5, where=where, note="attend"))
            super().update_plan(tick)
    return EventAgent


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    events = event_agent(ref)
    base = invitations({3: 0, 1: 1})
    return {
        "reference": drive(ref.Agent, ref, base),
        "events": drive(events, ref, base),
        "late": drive(events, ref, invitations({3: 2, 1: 1})),
        "weighted": drive(events, ref, invitations({3: 0, 1: 1}, party=6, talk=10)),
    }


def verify(result):
    ref, ev = result["reference"], result["events"]
    return [
        practice.Check(
            "ANSWER: on the reference the party dominates completely",
            ref == {"HobbsCafe": [0, 1, 2, 3], "Library": [KLAUS]},
            f"reference Agents end {ref}: the talk invitations never contain 'party at', "
            "so reflect never turns them into a belief",
        ),
        practice.Check(
            "FINDING: with events generalised, agents split, and invitation order decides",
            ev == {"HobbsCafe": [0, 1, 2], "Library": [3, KLAUS]}
            and result["late"]["Library"] == [KLAUS],
            f"reflection keyed by place ends {ev}: agent-3 heard of the talk first, "
            f"agent-1 of the party; with every talk invitation a tick later, the Library "
            f"holds {result['late']['Library']}",
        ),
        practice.Check(
            "FINDING: importance changes nothing",
            result["weighted"] == ev,
            f"talk at importance 10 against party at 6 ends {result['weighted']} -- "
            "importance only gates reflection at 6, and act takes the first matching plan",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
