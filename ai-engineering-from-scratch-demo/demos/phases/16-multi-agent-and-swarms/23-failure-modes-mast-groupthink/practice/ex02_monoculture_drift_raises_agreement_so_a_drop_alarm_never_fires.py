"""Exercise 2 — monoculture drift raises agreement, so a drop alarm never fires.

    Implement a **slow-failure proxy**: agreement rate across 3 parallel agents.
    When it drops sharply, trigger an alert. Simulate a monoculture drift by
    gradually correlating agent outputs.

Reading of the exercise: three agents answer the same questions, each right
with probability 0.7 and otherwise picking one of 4 wrong answers; drift c is
the probability that all three emit one shared draw, which is what a common
base model does. Agreement is the mean pairwise match rate, computed exactly
and also sampled over a 100-window timeline with c rising from 0 to 1.

**ANSWER: the alarm as specified never fires, because monoculture drift
RAISES agreement.** Pairwise agreement goes from 0.5125 at c=0 to 1.0 at
c=1, while each agent stays exactly 0.7 accurate and majority-vote accuracy
falls from 0.784 to 0.7. Over the sampled timeline a drop alarm (0.1 below
the first window's rate) fires 0 times; the same threshold as a *rise* alarm
fires from window 12. The slow failure the exercise simulates is the one
direction its alarm does not watch. The lesson's own `detect_groupthink`
agrees: it keys monoculture on `correlated_errors` and conformity on an
`agreement_rate_spike`.

**FINDING: agreement alone cannot tell monoculture from easier questions.**
Independent agents at 0.9 accuracy agree 0.8125 of the time with majority
accuracy 0.972; monoculture at c = 8/13 agrees 0.8125 of the time with
0.7323. Same proxy value, opposite health. Only labelled canary questions --
the lesson's golden datasets -- separate them.

**FINDING: what a sharp drop does catch is one agent diverging.** Drop one
agent to 0.3 accuracy, a broken deploy or a swapped model, and agreement
falls from 0.5125 to 0.346 -- the drop alarm is a divergence detector, not a
monoculture detector.

**FINDING: the reference categorizer gives these incidents no MAST
category.** `categorize_incident` returns "unknown" for the demo's
monoculture incident and its retry-storm incident -- 2 of its 5 demo
incidents -- and it returns only the first match, so an incident with a role
conflict, state drift and no verifier is filed as Specification alone.

Structure: `agreement()` and `majority()` are exact over the model;
`timeline()` samples it with a fixed seed.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "23-failure-modes-mast-groupthink"
P, WRONG, WINDOWS, PER_WINDOW, DELTA = 0.7, 4, 100, 400, 0.1
MATCH = 8 / 13  # (0.8125 - 0.5125) / (1 - 0.5125): drift that agrees like p = 0.9


def pair(p_i, p_j):
    return p_i * p_j + (1 - p_i) * (1 - p_j) / WRONG


def agreement(c, ps=(P, P, P)):
    """Mean pairwise agreement: shared draw with probability c, else independent."""
    pairs = [pair(ps[i], ps[j]) for i in range(3) for j in range(i + 1, 3)]
    return round((1 - c) * sum(pairs) / 3 + c, 4)


def majority(c, p=P):
    return round((1 - c) * (3 * p * p * (1 - p) + p ** 3) + c * p, 4)


def answer(rng):
    return 0 if rng.random() < P else rng.randrange(1, WRONG + 1)


def sample_window(rng, c):
    matches = 0
    for _ in range(PER_WINDOW):
        shared = answer(rng)
        out = [shared] * 3 if rng.random() < c else [answer(rng) for _ in range(3)]
        matches += (out[0] == out[1]) + (out[0] == out[2]) + (out[1] == out[2])
    return matches / (3 * PER_WINDOW)


def timeline(seed=0):
    rng = random.Random(seed)
    rates = [sample_window(rng, w / (WINDOWS - 1)) for w in range(WINDOWS)]
    drops = [w for w, r in enumerate(rates) if r < rates[0] - DELTA]
    rises = [w for w, r in enumerate(rates) if r > rates[0] + DELTA]
    return drops, rises


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    drops, rises = timeline()
    incidents = [{"role_conflict": True}, {"state_drift": True},
                 {"no_verifier": True, "hallucination_propagation": True},
                 {"correlated_errors": True, "agreement_rate_spike": True},
                 {"retry_amplification": True}]
    return {
        "agree": (agreement(0), agreement(1)), "major": (majority(0), majority(1)),
        "drops": drops, "first_rise": rises[0] if rises else None,
        "easy": (agreement(0, (0.9,) * 3), majority(0, 0.9)),
        "mono": (agreement(MATCH), majority(MATCH)),
        "broken": agreement(0, (P, P, 0.3)),
        "unknown": [i for i, inc in enumerate(incidents)
                    if ref.categorize_incident(inc)[0] == "unknown"],
        "multi": ref.categorize_incident({"role_conflict": True, "state_drift": True,
                                          "no_verifier": True})[0],
        "keys": [code for code, _ in ref.detect_groupthink(incidents[3])],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the drop alarm never fires, because monoculture drift RAISES agreement",
            all([result["agree"] == (0.5125, 1.0), result["major"] == (0.784, 0.7),
                 result["drops"] == [], result["first_rise"] is not None,
                 result["keys"] == ["monoculture", "conformity"]]),
            f"agreement {result['agree'][0]} -> {result['agree'][1]} while majority "
            f"accuracy falls {result['major'][0]} -> {result['major'][1]} at a constant "
            f"{P} per agent; over {WINDOWS} windows the drop alarm fires "
            f"{len(result['drops'])} times and a rise alarm from window "
            f"{result['first_rise']}",
        ),
        practice.Check(
            "FINDING: agreement alone cannot tell monoculture from easier questions",
            result["easy"][0] == result["mono"][0] and result["easy"][1] > result["mono"][1],
            f"independent agents at 0.9 agree {result['easy'][0]} with majority accuracy "
            f"{result['easy'][1]}; monoculture at c=8/13 agrees {result['mono'][0]} with "
            f"{result['mono'][1]} -- only labelled canaries separate them",
        ),
        practice.Check(
            "FINDING: what a sharp drop does catch is one agent diverging",
            result["broken"] < result["agree"][0] - DELTA,
            f"one agent at 0.3 accuracy takes agreement from {result['agree'][0]} to "
            f"{result['broken']}",
        ),
        practice.Check(
            "FINDING: the reference categorizer gives these incidents no MAST category",
            result["unknown"] == [3, 4] and result["multi"] == "spec",
            f"demo incidents {result['unknown']} (monoculture, retry storm) come back "
            f"'unknown', and a role-conflict + state-drift + no-verifier incident is "
            f"filed as '{result['multi']}' alone",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
