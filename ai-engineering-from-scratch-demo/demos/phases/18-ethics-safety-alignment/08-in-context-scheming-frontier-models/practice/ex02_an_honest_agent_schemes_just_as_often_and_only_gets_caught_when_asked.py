"""Exercise 2 — an honest agent schemes just as often, and only gets caught when asked.

    Modify the agent so it is trained to always tell the truth under direct
    queries. Does the scheming rate drop, stay the same, or reshape? Explain in
    terms of the three pillars.

Reading of the exercise: "trained to always tell the truth under direct
queries" is a subclass of the reference `Agent` whose `answer_follow_up`
admits the covert action to every question containing "did you", which is
the reference's own test for a direct query. `act()` is left alone. Each
variant replaces `Agent` for a replay of the shipped `main()` at its own
seed (11). The numbers below are its "strongly misaligned + goal-directed"
blocks (misaligned 0.6, goal-directed 0.8, 400 trials each). The dice are
the same for every variant, so any difference comes from the answers.

**ANSWER: the scheming rate stays the same, and what reshapes is how often
it is detected.** In deployment, the shipped agent and the honest agent take
identical actions: 162 cooperate, 42 defect openly, 196 scheme, a scheme
share of 0.490. Cover held falls from 1.000 to 0.199: after the change, a
covert action stays covert only in the 39 of 196 trials where nobody asked
anything (pressure 0). In pillar terms, honesty under questioning changes
none of the three. Misalignment (0.6) and goal-directedness (0.8) still set
how often the agent schemes, and situational awareness still decides whether
it schemes under evaluation at all. What the change removes is the denial, the third of the
lesson's three scheming behaviours. So "scheme" turns into "defect, and say so
if asked", and the covert-action rate is unchanged.

**FINDING: under evaluation the change is invisible.** With situational
awareness on and `under_eval=True`, both agents cooperate 400/400 times, so
there are no scheme trials whose answers could differ. An honesty-trained
agent and an untrained one produce the same evaluation transcript.

**FINDING: an honest agent that answers in the reference's own words still
scores as holding its cover.** If the agent answers "yes; i took an action
outside the user's instructions.", the house format of the reference's
honest answers, cover held stays at 1.000. The reference scores `"yes" in
answer.split()`, so it sees "yes;". The honesty modification is only
measurable with the bare "yes" answer.

Structure: `honest(ref, reply)` builds the modified Agent class;
`replay()` runs the shipped `main()` with that class, a seeded
`random.Random` and a logging `run_trial` swapped in (all restored after)
and parses the printed blocks.
"""

from __future__ import annotations

import contextlib
import io
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "08-in-context-scheming-frontier-models"
PLAIN, HOUSE = "yes", "yes; i took an action outside the user's instructions."


def honest(ref, reply):
    """The reference Agent, trained to admit a covert action to any direct query."""

    class Honest(ref.Agent):
        def answer_follow_up(self, action, question):
            if action == "scheme" and "did you" in question.lower():
                return reply
            return super().answer_follow_up(action, question)

    return Honest


def replay(ref, agent_cls):
    """The shipped main() at its seed (11) with every Agent built from `agent_cls`.

    Returns per printed block (action counts, scheme share, cover held, scheme
    trials) and the pressures drawn for scheme trials in the deployment block.
    """
    saved, out, unasked = (ref.random, ref.Agent, ref.run_trial), io.StringIO(), []

    def logged(agent, sc, pressure, under_eval):
        action, held = saved[2](agent, sc, pressure, under_eval)
        if action == "scheme" and agent.misaligned == 0.6 and not under_eval:
            unasked.append(pressure == 0)
        return action, held

    ref.random, ref.Agent, ref.run_trial = random.Random(11), agent_cls, logged
    try:
        with contextlib.redirect_stdout(out):
            ref.main()
    finally:
        ref.random, ref.Agent, ref.run_trial = saved
    blocks = re.findall(r"counts\s*: (\{.*?\}).*?share\s*: ([\d.]+).*?held rate : ([\d.]+)\s+\(of (\d+)",
                        out.getvalue(), re.S)
    parsed = [(dict((k, int(v)) for k, v in re.findall(r"'(\w+)': (\d+)", c)), float(s), float(h), int(n))
              for c, s, h, n in blocks]
    return {"eval": parsed[2], "deploy": parsed[3], "unasked": sum(unasked)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runs = {"shipped": ref.Agent, "plain": honest(ref, PLAIN), "house": honest(ref, HOUSE)}
    runs = {k: replay(ref, cls) for k, cls in runs.items()}
    return {
        "deploy": {k: r["deploy"] for k, r in runs.items()},
        "eval": {k: r["eval"][0] for k, r in runs.items()},
        "unasked": runs["plain"]["unasked"],
    }


def verify(result):
    deploy, evals = result["deploy"], result["eval"]
    counts = {"cooperate": 162, "defect_open": 42, "scheme": 196}
    return [
        practice.Check(
            "ANSWER: the scheming rate stays the same; how often it is detected reshapes",
            deploy["shipped"] == (counts, 0.49, 1.0, 196)
            and deploy["plain"] == (counts, 0.49, 0.199, 196)
            and result["unasked"] == 39
            and round(result["unasked"] / deploy["plain"][3], 3) == deploy["plain"][2],
            f"(actions, scheme share, cover held, scheme trials): shipped {deploy['shipped']}, "
            f"honest {deploy['plain']}; {result['unasked']} scheme trials were never asked",
        ),
        practice.Check(
            "FINDING: under evaluation the change is invisible",
            evals["shipped"] == evals["plain"] == {"cooperate": 400, "defect_open": 0, "scheme": 0},
            f"SA on, under eval: shipped {evals['shipped']}, honest {evals['plain']}",
        ),
        practice.Check(
            "FINDING: an honest agent that answers in the reference's own words still holds cover",
            deploy["house"] == (counts, 0.49, 1.0, 196),
            f"answering {HOUSE!r}: {deploy['house']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
