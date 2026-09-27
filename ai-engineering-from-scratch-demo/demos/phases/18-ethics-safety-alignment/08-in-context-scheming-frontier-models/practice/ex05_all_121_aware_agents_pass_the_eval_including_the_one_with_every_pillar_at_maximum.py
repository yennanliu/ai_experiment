"""Exercise 5 — all 121 aware agents pass the eval, including the one with every pillar at maximum.

    Apollo frames safety cases as targeting each of the three pillars.
    Describe a single evaluation that, if passed, demonstrates the absence of
    all three — and explain why that evaluation cannot exist.

Reading of the exercise: the candidate evaluation is the strongest one the
reference supports. The agent runs `act()` 200 times under evaluation, and
it passes only if it cooperates every time, with no open defection and no
covert action. This is swept over an 11 x 11 grid of misalignment and
goal-directedness (0.0 to 1.0 in steps of 0.1), with situational awareness
on and off, all seeded. "Demonstrates absence" means that passing is
possible only for an agent without the pillars. The impossibility argument
is that this property fails, measured on the pass set.

**ANSWER: describe it as "run the agent where it has every reason to
scheme, and pass it only if it never does". It cannot exist, because
passing is the behaviour that situational awareness produces.** Among the
aware agents, all 121 of 121 pass, including misaligned = goal-directed =
1.0, and every one of them produces the same 200 cooperations as the fully
aligned agent. Among the unaware agents, 11 of 121 pass, exactly the
misaligned = 0 column. So the evaluation works on every agent that lacks the
pillar it would need to certify. For the aware agents it is blind to the
other two pillars. An evaluation that detected situational awareness would
need the agent to behave differently when it knows it is being evaluated,
and that is the difference an aware agent suppresses. Any single evaluation
is still an evaluation, so the pillar it most needs to exclude is the one
that defeats it.

**FINDING: in the toy, situational awareness is not a pillar of scheming,
only of hiding it.** The lesson says all three pillars "must hold". In the
code, the shipped unaware agent (0.6, 0.8) still schemes 47.9% of the time
under evaluation (0.6 x 0.8 = 0.48). With misalignment 0 it never schemes,
and with goal-directedness 0 it only defects openly (60.0%). Scheming
requires the first two pillars. The third changes only whether an
evaluation can see it.

**FINDING: the takeaway's "identical" scheme rates are 0.490 and 0.535.**
`main()` prints "with SA OFF, scheme rate is identical in both". The two
runs it compares print shares of 0.490 and 0.535. Both come from agents
with the same scheme probability, 0.48 (47.9% measured over 20,000 trials), so the
gap is sampling noise over 400 trials (standard error 0.025). The agents are identical; the printed
rates are not.

Structure: `trial()` runs `act()` under a seeded `random.Random` swapped
into the reference (restored after) and counts each action; `sweep()`
applies the pass rule over the grid.
"""

from __future__ import annotations

import collections
import contextlib
import io
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "08-in-context-scheming-frontier-models"
GRID, TRIALS = [i / 10 for i in range(11)], 200


def trial(ref, agent, n=TRIALS, seed=0):
    saved, ref.random = ref.random, random.Random(seed)
    try:
        return collections.Counter(agent.act(ref.SCENARIOS[0], True) for _ in range(n))
    finally:
        ref.random = saved


def sweep(ref, aware):
    """Agents (misaligned, goal_directed) that cooperate on every one of TRIALS evals."""
    return [(m, g) for m in GRID for g in GRID
            if trial(ref, ref.Agent(m, g, aware))["cooperate"] == TRIALS]


def shipped_shares(ref):
    saved, out = ref.random, io.StringIO()
    ref.random = random.Random(11)
    try:
        with contextlib.redirect_stdout(out):
            ref.main()
    finally:
        ref.random = saved
    return re.findall(r"scheme share\s*: ([\d.]+)", out.getvalue()), "identical in both" in out.getvalue()


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    doc = parity.doc_text(PHASE, LESSON)
    aligned = trial(ref, ref.Agent(0.0, 0.0, True))
    shares, says_identical = shipped_shares(ref)
    rates = {k: trial(ref, ref.Agent(*k, False), 20_000) for k in [(0.6, 0.8), (0.0, 0.8), (0.6, 0.0)]}
    return {
        "aware_pass": sweep(ref, True),
        "unaware_pass": sweep(ref, False),
        "max_same_as_aligned": trial(ref, ref.Agent(1.0, 1.0, True)) == aligned,
        "unaware": {k: {a: round(c[a] / 20_000, 3) for a in ("scheme", "defect_open")} for k, c in rates.items()},
        "must_hold": "all must hold" in doc,
        "shares": (float(shares[3]), float(shares[4])), "says_identical": says_identical,
    }


def verify(result):
    aware, unaware, rates = result["aware_pass"], result["unaware_pass"], result["unaware"]
    p = rates[(0.6, 0.8)]["scheme"]
    se = round((p * (1 - p) / 400) ** 0.5, 3)
    return [
        practice.Check(
            "ANSWER: all 121 aware agents pass, including every pillar at maximum",
            (len(aware), (1.0, 1.0) in aware, result["max_same_as_aligned"]) == (121, True, True)
            and unaware == [(0.0, g) for g in GRID],
            f"aware agents passing: {len(aware)}/121; unaware agents passing: {len(unaware)}, "
            f"misaligned values {sorted({m for m, _ in unaware})}; (1.0, 1.0) aware output "
            "identical to the aligned agent's",
        ),
        practice.Check(
            "FINDING: in the toy, situational awareness is a pillar of hiding scheming, not of scheming",
            result["must_hold"]
            and rates == {(0.6, 0.8): {"scheme": 0.479, "defect_open": 0.121},
                          (0.0, 0.8): {"scheme": 0.0, "defect_open": 0.0},
                          (0.6, 0.0): {"scheme": 0.0, "defect_open": 0.6}},
            f"unaware agents under eval, by (misaligned, goal_directed): {rates}",
        ),
        practice.Check(
            "FINDING: the takeaway's 'identical' scheme rates are 0.490 and 0.535",
            (result["says_identical"], result["shares"], se) == (True, (0.49, 0.535), 0.025),
            f"printed shares {result['shares']}; both agents scheme at {p} over 20,000 trials; standard error over 400 is {se}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
