"""Exercise 2 — without reflection nobody comes, because reflection is the only wire.

    Remove the reflection step. What does behavior look like? Map to the
    ablation finding in Park 2023.

Reading of the exercise: "remove the reflection step" is `Agent.reflect`
replaced by a no-op on the reference, everything else untouched; the mapping
to Park 2023 is to the controlled evaluation's no-reflection condition, and
asks whether the miniature degrades the way the paper's agents did.

**ANSWER: only the host comes -- 1 of 5, against 5 of 5.** Agents 1-4 still
receive their invitations, at importance 7 and 8, and nothing reads them:
`update_plan` looks only at `beliefs`, and only `reflect` writes beliefs.
Agent 0 attends because its belief is seeded directly by `run_simulation`.
Park et al. report the opposite shape: removing reflection lowered the
TrueSkill believability rating from 29.89 to 26.88 -- still above the human
crowdworker condition at 22.95. Their agents got worse; these stop.

**FINDING: reflection here is a relay, not a synthesis.** Delete `reflect`
and put its test -- "invited" and "party at" in an importance-6+ memory from
the last 5 ticks -- straight into `update_plan`, and the final locations and
every plan are identical to the shipped run. The belief it writes is one
fixed string, "there is a party I was invited to", with no host, place or
time in it; the Klaus-and-Maria kind of generalisation the paper uses to
motivate reflection has nothing to generalise from.

**FINDING: the trigger is two case-sensitive substrings.** Of 6 phrasings of
the same invitation, 3 become a belief. "asked me to come to her party",
"a Valentine's party, at HobbsCafe" and "Invited me to a Party" do not --
`reflect` never lowercases -- so with reflection *on*, those guests stay home
exactly as if it were off.

Structure: `drive()` replays run_simulation's loop over reference `Agent`s;
`DirectAgent` is the reference Agent with reflect's test moved into
update_plan.
"""

from __future__ import annotations

import contextlib
import io
import re

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "17-generative-agents-simulation"
TEXT = "agent-{} invited me to a party at HobbsCafe at tick 5"
SHIPPED = {0: [(0, 1, 8), (0, 2, 8)], 1: [(1, 3, 7)], 2: [(2, 4, 7)]}
PHRASINGS = ["agent-0 invited me to a party at HobbsCafe at tick 5",
             "agent-0 invited me to the party at HobbsCafe",
             "agent-0 says there is a party at HobbsCafe and I am invited",
             "agent-0 asked me to come to her party at HobbsCafe at tick 5",
             "agent-0 invited me to a Valentine's party, at HobbsCafe",
             "agent-0 Invited me to a Party at HobbsCafe at tick 5"]
PARK = {"full": 29.89, "no reflection": 26.88, "human crowdworker": 22.95}


def converged(ref):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ref.run_simulation()
    return int(re.search(r"(\d+)/\d+ agents converged", out.getvalue()).group(1))


def drive(cls, ref):
    agents = [cls(f"agent-{i}", location="home") for i in range(5)]
    agents[0].plans.append(ref.Plan(tick=5, where="HobbsCafe", note="host the party"))
    agents[0].beliefs.append("there is a party I was invited to")
    for tick in range(6):
        for src, dst, importance in SHIPPED.get(tick, []):
            agents[dst].observe(tick, TEXT.format(src), importance)
        for agent in agents:
            agent.reflect(tick)
            agent.update_plan(tick)
            agent.act(tick)
    return [(a.location, [(p.tick, p.where) for p in a.plans]) for a in agents]


def beliefs(ref, text):
    agent = ref.Agent("guest", location="home")
    agent.observe(0, text, importance=8)
    agent.reflect(0)
    return agent.beliefs


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")

    class DirectAgent(ref.Agent):
        def reflect(self, tick):
            return None

        def update_plan(self, tick):
            if any(m.importance >= 6 and tick - m.ts <= 5 and "invited" in m.content
                   and "party at" in m.content for m in self.stream):
                self.beliefs[:] = self.beliefs or ["there is a party I was invited to"]
            super().update_plan(tick)

    full = converged(ref)
    original, ref.Agent.reflect = ref.Agent.reflect, lambda self, tick: None
    try:
        ablated = converged(ref)
    finally:
        ref.Agent.reflect = original
    belief = drive(ref.Agent, ref)
    return {
        "full": full, "ablated": ablated,
        "same": drive(DirectAgent, ref) == belief,
        "belief": beliefs(ref, PHRASINGS[0])[0],
        "passes": [bool(beliefs(ref, text)) for text in PHRASINGS],
    }


def verify(result):
    passes = result["passes"]
    return [
        practice.Check(
            "ANSWER: only the host comes -- 1 of 5, against 5 of 5",
            result["ablated"] == 1 and result["full"] == 5,
            f"with reflect a no-op {result['ablated']}/5 converge against {result['full']}/5; "
            f"Park's no-reflection agents dropped from {PARK['full']} to "
            f"{PARK['no reflection']} TrueSkill, still above the human crowdworker "
            f"{PARK['human crowdworker']} -- they got worse, these stop",
        ),
        practice.Check(
            "FINDING: reflection here is a relay, not a synthesis",
            result["same"] and not any(w in result["belief"] for w in ("HobbsCafe", "5", "agent")),
            "moving reflect's keyword test into update_plan reproduces every final "
            f"location and plan; the belief is the fixed string '{result['belief']}', "
            "with no host, place or time",
        ),
        practice.Check(
            "FINDING: the trigger is two case-sensitive substrings",
            passes == [True, True, True, False, False, False],
            f"{sum(passes)} of {len(passes)} phrasings of the same invitation become a "
            "belief; 'asked me to come', 'party, at' and 'Invited ... Party' do not, "
            "because reflect never lowercases",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
