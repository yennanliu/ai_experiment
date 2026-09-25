"""Exercise 1 — the debate ends further from the truth than it started.

    Run `code/main.py`, then set the round count to 5 and watch diminishing
    returns. At which round does additional convergence stop?

Reading of the exercise: run it, read the error column rather than the
agreement column, and compare both against the control the module already
prints -- because the returns are not diminishing, they are negative, and the
cause is one line of the update loop.

**ANSWER: round 1, and only once the update is fixed.** `revise` is applied in
place over a live list, so the second agent averages against the first agent's
*already revised* answer. That is a Gauss-Seidel sweep, not the simultaneous
update Du et al. specify. Replacing it with a snapshot makes the debate reach
the confidence-weighted mean in **1** round and stay there exactly: error
**0.889** at rounds 1 through 6, identical to thirteen decimal places. There
are no diminishing returns because after round 1 there are no returns.

**FINDING: as shipped, more rounds make it worse.** Error against the true
answer over six rounds runs **2.24, 2.43, 2.48, 2.50, 2.50, 2.50** -- monotone
increasing, converging on a wrong value. The module's own control, the round-0
mean, scores **1.83**. Three rounds of debate leave the team **36%** further
from the truth than not debating at all.

**FINDING: it is not an artefact of these confidences.** Permuting the three
confidence values over the three agents gives final errors of 1.92, 2.48,
3.29, 4.27, 5.24 and 5.65 -- worse than the 1.83 control in **6** of **6**
assignments. The in-place sweep drags the consensus toward whichever agent the
list happens to put first, and here that is the one furthest below the answer.

**FINDING: all three printed takeaways are contradicted by the printed table.**
"1 round of exchange cuts the error most" -- round 1 *raises* it from 1.83 to
2.24. "Rounds 2-3 compound" -- they compound the error. "Beyond round 3 the
gain per round shrinks" -- there is no gain to shrink; the per-round change is
**+0.19, +0.05, +0.01**, a plateau at the wrong number.

Structure: `sweep()` runs the debate under either update rule; `permutations()`
re-runs it with the confidences reassigned.
"""

from __future__ import annotations

import inspect
import itertools

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "07-society-of-mind-debate"
ANSWERS = (38.0, 42.5, 51.0)
CONFIDENCES = (0.6, 0.8, 0.4)
ROUNDS = 6


def team(ref, confidences=CONFIDENCES):
    """The shipped three agents, with confidences assignable."""
    return [ref.DebateAgent(name=name, answer=answer, confidence=confidence)
            for name, answer, confidence in zip("ABC", ANSWERS, confidences)]


def sweep(ref, agents, rounds=ROUNDS, simultaneous=False):
    """Error against the truth after each round, under one update rule."""
    for agent in agents:
        agent.initial()
    errors = []
    for _ in range(rounds):
        snapshot = [ref.DebateAgent(a.name, a.answer, a.confidence) for a in agents]
        for agent in agents:
            others = ([o for o in snapshot if o.name != agent.name] if simultaneous
                      else [o for o in agents if o is not agent])
            agent.revise(others)
        errors.append(round(ref.error_vs_truth(agents), 4))
    return errors


def permutations(ref):
    """Final error for every assignment of the three confidences to the three agents."""
    return sorted(round(sweep(ref, team(ref, order), rounds=3)[-1], 2)
                  for order in itertools.permutations(CONFIDENCES))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    src = inspect.getsource(ref)
    shipped = sweep(ref, team(ref))
    fixed = sweep(ref, team(ref), simultaneous=True)
    control = ref.single_shot_majority(team(ref))
    control_error = round(abs(control - ref.TRUE_ANSWER), 4)
    deltas = [round(b - a, 2) for a, b in zip(shipped, shipped[1:])]
    return {
        "shipped": shipped, "fixed": fixed, "control": control_error,
        "rising": shipped == sorted(shipped), "deltas": deltas,
        "flat_after_one": len(set(fixed)) == 1,
        "worse_than_control": sum(error > control_error for error in shipped),
        "rounds": ROUNDS,
        "permutations": permutations(ref),
        "in_place": "others = [o for o in agents if o is not a]" in src,
        "penalty": round(100 * (shipped[2] - control_error) / control_error),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: round 1, and only once the update is fixed",
            all([result["flat_after_one"], result["fixed"][0] == 0.8889,
                 len(result["fixed"]) == result["rounds"], result["in_place"]]),
            f"revise is applied in place over a live list, so the second agent averages "
            f"against the first's revised answer; with a snapshot the debate reaches "
            f"{result['fixed'][0]} at round 1 and holds it for all {result['rounds']} "
            "rounds -- no diminishing returns because after round 1 there are none",
        ),
        practice.Check(
            "FINDING: as shipped, more rounds make it worse",
            all([result["rising"], result["worse_than_control"] == result["rounds"],
                 result["control"] == 1.8333, result["penalty"] == 36]),
            f"error over {result['rounds']} rounds runs {result['shipped']} -- monotone "
            f"increasing -- against a round-0 control of {result['control']}; three "
            f"rounds leave the team {result['penalty']}% further from the truth than not "
            "debating",
        ),
        practice.Check(
            "FINDING: it is not an artefact of these confidences",
            all([len(result["permutations"]) == 6,
                 all(error > result["control"] for error in result["permutations"])]),
            f"permuting the confidences over the agents gives final errors "
            f"{result['permutations']} -- worse than the {result['control']} control in "
            f"{len(result['permutations'])} of {len(result['permutations'])} "
            "assignments, because the sweep drags consensus toward whoever is first",
        ),
        practice.Check(
            "FINDING: all three printed takeaways are contradicted by the printed table",
            all([result["shipped"][0] > result["control"],
                 result["deltas"][:3] == [0.19, 0.05, 0.01]]),
            f"round 1 raises the error from {result['control']} to "
            f"{result['shipped'][0]} rather than cutting it most; rounds 2-3 compound "
            f"that; and the per-round change {result['deltas'][:3]} is a plateau at the "
            "wrong number rather than a shrinking gain",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
