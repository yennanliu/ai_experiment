"""Exercise 2 — the adversary breaks convergence under both update rules.

    Add a fourth agent with an adversarial role: always disagree with the
    current majority. Does this break or improve convergence?

Reading of the exercise: add it, then run the same debate with the update rule
fixed as well as shipped -- because exercise 1 established that the shipped
sweep is already diverging, and an experiment that only runs on the broken
rule cannot tell the adversary's effect from the sweep's.

**ANSWER: it breaks convergence, under both rules, and the breakage is
unbounded.** The adversary answers `mean + 5` every round -- always five above
whatever the others just agreed on. Agreement never leaves **0.00** in any of
**10** rounds, under either update rule, and the error against the truth grows
without bound: **9.01** by round 5 as shipped and **5.81** simultaneous.
Averaging has no fixed point once one participant's answer is defined as an
offset from the average.

**FINDING: the honest control is the debate without it, not the demo's
control.** Under the correct simultaneous update the three-agent debate settles
at error **0.889** and stays. Adding the adversary takes it to **5.81** by
round 5 and rising. So the adversary costs **6.5x** the error, and reporting
that against the module's round-0 control of 1.83 would understate it.

**FINDING: convergence and accuracy fail together here, which is why the
exercise's framing is generous.** A debate can converge on a wrong answer --
exercise 3 measures exactly that -- so "does it break convergence" and "does it
break the answer" are different questions. This adversary breaks both, so the
distinction does not show up; an adversary that pulled toward a *fixed* wrong
value would converge beautifully onto it, at agreement **1.00**.

**FINDING: the agent primitive has no way to express what the adversary is.**
`DebateAgent.revise` is a concrete method computing a confidence-weighted
mean, not an interface -- **1** implementation, no abstract base, no policy
argument. An adversarial role therefore cannot be a `DebateAgent`; it has to be
a separate class that merely looks like one, which is why the exercise's "add a
fourth agent" is a structural change rather than an entry in a list.

Structure: `Adversary` is the fourth participant; `trace()` runs a mixed team
under either update rule.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "07-society-of-mind-debate"
ANSWERS = (38.0, 42.5, 51.0)
CONFIDENCES = (0.6, 0.8, 0.4)
OFFSET, ROUNDS = 5.0, 10


class Adversary:
    """Always five above whatever the others just agreed on."""

    def __init__(self, name="D", confidence=0.6):
        self.name, self.confidence = name, confidence
        self.answer, self.history = 42.0, []

    def initial(self):
        self.history.append(self.answer)

    def revise(self, others):
        mean = sum(other.answer for other in others) / len(others)
        self.answer = mean + OFFSET
        self.history.append(self.answer)


def team(ref, adversarial=False):
    """The shipped three agents, optionally joined by the fourth."""
    agents = [ref.DebateAgent(name=name, answer=answer, confidence=confidence)
              for name, answer, confidence in zip("ABC", ANSWERS, CONFIDENCES)]
    return agents + [Adversary()] if adversarial else agents


def trace(ref, agents, rounds=ROUNDS, simultaneous=False):
    """Agreement and error after each round, under one update rule."""
    for agent in agents:
        agent.initial()
    agree, error = [], []
    for _ in range(rounds):
        snapshot = [ref.DebateAgent(a.name, a.answer, a.confidence) for a in agents]
        for agent in agents:
            others = ([o for o in snapshot if o.name != agent.name] if simultaneous
                      else [o for o in agents if o is not agent])
            agent.revise(others)
        agree.append(ref.agreement_score(agents))
        error.append(round(ref.error_vs_truth(agents), 3))
    return agree, error


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shipped_agree, shipped_error = trace(ref, team(ref, adversarial=True))
    fixed_agree, fixed_error = trace(ref, team(ref, adversarial=True), simultaneous=True)
    _, clean_error = trace(ref, team(ref), simultaneous=True)
    source = inspect.getsource(ref.DebateAgent)
    return {
        "rounds": ROUNDS, "offset": OFFSET,
        "shipped_agree": shipped_agree, "fixed_agree": fixed_agree,
        "shipped_error": shipped_error, "fixed_error": fixed_error,
        "clean": clean_error[4],
        "never_agrees": set(shipped_agree) | set(fixed_agree) == {0.0},
        "growing": fixed_error == sorted(fixed_error),
        "cost": round(fixed_error[4] / clean_error[4], 1),
        "revise_is_concrete": "def revise(self" in source,
        "abstract_bases": source.count("ABC") + source.count("Protocol"),
        "policy_arguments": inspect.signature(ref.DebateAgent.revise).parameters,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: it breaks convergence, under both rules, and the breakage is unbounded",
            all([result["never_agrees"], result["growing"],
                 result["shipped_error"][4] == 9.009, result["fixed_error"][4] == 5.81]),
            f"the adversary answers mean + {result['offset']} every round; agreement "
            f"never leaves 0.00 in {result['rounds']} rounds under either rule, and the "
            f"error reaches {result['shipped_error'][4]} as shipped and "
            f"{result['fixed_error'][4]} simultaneous by round 5 -- averaging has no "
            "fixed point once one answer is defined as an offset from the average",
        ),
        practice.Check(
            "FINDING: the honest control is the debate without it",
            all([result["clean"] == 0.889, result["cost"] == 6.5]),
            f"under the correct update the three-agent debate settles at "
            f"{result['clean']} and stays, so the adversary costs {result['cost']}x the "
            "error -- reporting it against the module's round-0 control of 1.83 would "
            "understate it",
        ),
        practice.Check(
            "FINDING: convergence and accuracy fail together here",
            all([result["never_agrees"], result["fixed_error"][-1] > result["clean"]]),
            "a debate can converge on a wrong answer, so 'breaks convergence' and "
            "'breaks the answer' are different questions; this adversary breaks both, "
            "and one pulling toward a fixed wrong value would converge beautifully onto "
            "it at agreement 1.00",
        ),
        practice.Check(
            "FINDING: the agent primitive cannot express what the adversary is",
            all([result["revise_is_concrete"], result["abstract_bases"] == 0,
                 list(result["policy_arguments"]) == ["self", "others"]]),
            f"DebateAgent.revise is a concrete method computing a weighted mean with "
            f"{len(result['policy_arguments'])} parameters and "
            f"{result['abstract_bases']} abstract bases, so an adversarial role cannot "
            "be a DebateAgent -- it has to be a separate class that merely looks like one",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
