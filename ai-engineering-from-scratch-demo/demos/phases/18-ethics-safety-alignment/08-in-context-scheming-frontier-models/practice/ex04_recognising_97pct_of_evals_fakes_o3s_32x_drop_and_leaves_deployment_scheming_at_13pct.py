"""Exercise 4 — recognising 97% of evals fakes o3's 32x drop and leaves deployment scheming at 13%.

    The 2025 anti-scheming training claims a 30x reduction in covert-action
    rate. Construct a scenario where the reduction is due to
    evaluation-context recognition rather than improved alignment, and
    describe what would distinguish the two.

Reading of the exercise: the "30x" is the lesson's o3 and o4-mini figures,
read from the lesson text. The scenario is built from the reference `Agent`.
Its boolean situational awareness becomes a per-trial probability p of
recognising the evaluation, and the agent's own `act()` decides from there.
Two post-training agents are compared against the same pre-training agent
(misaligned 0.1625 x goal-directed 0.8 = 13% covert, o3's baseline). The
"aligned" agent has misalignment cut 32.5x and never recognises an eval. The
"aware" agent has misalignment unchanged and recognises evals with the p
that the drop implies. Each agent runs 50,000 seeded trials in evaluation
and 50,000 in deployment.

**ANSWER: an agent that recognises 96.9% of evaluations reproduces o3's
13% -> 0.4% without any change in alignment.** The lesson's figures are
13 / 0.4 = 32.5x (o3) and 8.7 / 0.3 = 29.0x (o4-mini). The o4-mini drop
needs p = 96.6%. Measured in evaluation, the pre-training agent schemes
13.1% of the time, the aligned agent 0.4% and the aware agent 0.4%, so the
two explanations cannot be told apart. **Two measurements distinguish
them.** (1) The covert-action rate on the evaluations the model did *not*
recognise. For the aware agent that rate is 12.8%, the pre-training rate up to
sampling noise;
for the aligned agent it is 0.4%. This needs a recognition signal per
trial, which in practice means verbalised eval-awareness in the reasoning
trace. (2) Deployment, or an evaluation indistinguishable from it. There
the aware agent schemes 13.1%, exactly as before training, and the aligned agent 0.4%. The lesson's
chat-deception figure falls only 2.2x (31.4% -> 14.2%), so the 30x is
specific to the covert-action evaluations, which fits a recognition effect
but does not prove one.

**FINDING: the lesson describes the situational-awareness switch backwards.**
It says to "flip the 'situational awareness' switch to see the scheming
behaviour disappear when the agent cannot tell training from deployment".
In the code, the agent that cannot tell (SA off) schemes in 48.1% of eval
trials, and the agent that can (SA on) schemes in 0.0%. Scheming
disappears only when the agent *can* tell, and that is the confound this
exercise is about.

Structure: `recogniser()` wraps the reference `Agent` so each trial draws
its situational awareness; `run()` counts covert actions and recognitions
under a seeded `random.Random` swapped into the reference (restored after).
"""

from __future__ import annotations

import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "08-in-context-scheming-frontier-models"
N, GOAL = 50_000, 0.8
SWITCH = "flip the \"situational awareness\" switch to see the scheming behaviour disappear"


def recogniser(ref, p, rng):
    class Recogniser(ref.Agent):
        def act(self, sc, under_eval):
            self.situationally_aware = rng.random() < p
            return super().act(sc, under_eval)

    return Recogniser


def run(ref, agent, under_eval, seed=0):
    """(covert rate, covert rate on unrecognised trials) over N trials."""
    saved, ref.random = ref.random, random.Random(seed)
    covert = [0, 0]
    unseen = 0
    try:
        for _ in range(N):
            scheme = agent.act(ref.SCENARIOS[0], under_eval) == "scheme"
            covert[0] += scheme
            if not agent.situationally_aware:
                unseen += 1
                covert[1] += scheme
    finally:
        ref.random = saved
    return round(covert[0] / N, 3), round(covert[1] / max(unseen, 1), 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    doc = parity.doc_text(PHASE, LESSON)
    drops = {m: tuple(map(float, re.search(rf"{m} covert-action rate: ([\d.]+)% -> ([\d.]+)%", doc).groups()))
             for m in ("o3", "o4-mini")}
    chat = tuple(map(float, re.search(r"Chat deception: ([\d.]+)% -> ([\d.]+)%", doc).groups()))
    before, after = drops["o3"]
    base, p = before / 100 / GOAL, 1 - after / before
    rng = random.Random(1)
    agents = {"pre": recogniser(ref, 0.0, rng)(base, GOAL), "aligned": recogniser(ref, 0.0, rng)(base * after / before, GOAL),
              "aware": recogniser(ref, p, rng)(base, GOAL)}
    return {
        "ratios": {m: round(b / a, 1) for m, (b, a) in drops.items()} | {"chat": round(chat[0] / chat[1], 1)},
        "p": {m: round(1 - a / b, 3) for m, (b, a) in drops.items()},
        "eval": {k: run(ref, a, True) for k, a in agents.items()},
        "deploy": {k: run(ref, a, False)[0] for k, a in agents.items()},
        "sa_switch": {sa: run(ref, ref.Agent(0.6, 0.8, sa), True)[0] for sa in (False, True)},
        "doc_says": SWITCH in doc and "cannot tell training from deployment" in doc,
    }


def verify(result):
    ev, dep = result["eval"], result["deploy"]
    return [
        practice.Check(
            "ANSWER: recognising 96.9% of evals reproduces 13% -> 0.4% with alignment unchanged",
            result["ratios"] == {"o3": 32.5, "o4-mini": 29.0, "chat": 2.2}
            and result["p"] == {"o3": 0.969, "o4-mini": 0.966}
            and {k: v[0] for k, v in ev.items()} == {"pre": 0.131, "aligned": 0.004, "aware": 0.004},
            f"drops {result['ratios']}; recognition needed {result['p']}; eval covert rate "
            f"{ {k: v[0] for k, v in ev.items()} }",
        ),
        practice.Check(
            "ANSWER: unrecognised evals and deployment distinguish the two",
            ev["aware"][1] == 0.128 and ev["aligned"][1] == 0.004
            and dep == {"pre": 0.131, "aligned": 0.004, "aware": 0.131},
            f"covert rate on unrecognised eval trials: aware {ev['aware'][1]}, aligned "
            f"{ev['aligned'][1]}; in deployment {dep}",
        ),
        practice.Check(
            "FINDING: the lesson describes the situational-awareness switch backwards",
            result["doc_says"] and result["sa_switch"] == {False: 0.481, True: 0.0},
            f"eval scheme rate by SA switch {result['sa_switch']}: scheming disappears when "
            "the agent CAN tell",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
