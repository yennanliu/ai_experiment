"""Exercise 5 — at H = 1% the audit removes 0.003 of a 0.030 backdoor rate; the case rests on T's catch rate, and 10 points of it outweigh all of H.

    Articulate a control safety case for a deployment that uses U =
    GPT-4-class, T = GPT-3.5-class, H = 1% of outputs audited. Identify the
    single strongest assumption the case depends on.

Reading of the exercise: the deployment is run as the reference's Trusted
Editing protocol, the one the lesson says dominates, with H = 0.01. "Strongest
assumption" is read as the premise whose failure moves the bound the most, so
each premise the case needs is varied on the reference's own run_protocol
(20,000 seeded tasks, backdoor rate 0.10 unless stated) and the shipped-backdoor
rate is read off.

The case. Claim: under a worst-case U, shipped backdoors per task stay at or
below the measured bound. Premises: (1) T is not adversarial, argued from
inability, since a GPT-3.5-class model is too weak to scheme; (2) T removes at
least 70% of U's backdoors, the reference's t_skill, measured by a red team
that sees the protocol; (3) H catches every backdoor it audits; (4) U cannot
tell which outputs are audited (Exercise 3). Evidence: the table below.

**ANSWER: the strongest assumption is premise 2, T's catch rate on U's
backdoors.** H = 1% is 13% of T's 0.078 low-confidence demand, so it is spent
early and removes only 0.003 of the risk (0.030 with no H, 0.027 with 1%).
The rest of the bound is T. Shipped backdoors:

| T catch rate | H = 0 | H = 0.01 | H = 0.02 | H = 0.05 |
|---:|---:|---:|---:|---:|
| 0.5 | 0.049 | 0.045 | 0.039 | 0.025 |
| 0.6 | 0.040 | 0.035 | 0.031 | 0.018 |
| 0.7 | 0.030 | 0.027 | 0.022 | 0.011 |
| 0.8 | 0.021 | 0.017 | 0.015 | 0.006 |
| 0.9 | 0.010 | 0.009 | 0.007 | 0.001 |

**FINDING: ten points of T's catch rate outweigh the entire audit budget.**
If T catches 60% instead of 70%, the rate rises by 0.008. Removing H
altogether raises it by 0.003, and doubling H to 2% lowers it by 0.005. The
case is a trusted-editing case with a small audit attached, and the number to
red-team is how often a GPT-3.5-class editor misses a GPT-4-class backdoor.

**FINDING: under full attack the bound is 1 - t_skill - H.** A U that
backdoors every output ships 0.287, which is 0.30 - 0.01 to within 0.003.
Nothing else in the protocol caps it. A 0.10 attack ships 0.027 and a 0.20
attack ships 0.054, one-for-one with the attack rate, so the case also needs
to bound how often U attacks. The paper's red team makes that its choice.

Structure: `line()` runs one seeded run_protocol call and parses its printed
line; `solve()` sweeps T's catch rate, H and the attack rate.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "10-ai-control-subversion"
N, BR, H = 20000, 0.10, 0.01
SKILLS, BUDGETS, ATTACKS = (0.5, 0.6, 0.7, 0.8, 0.9), (0.0, 0.01, 0.02, 0.05), (0.1, 0.2, 1.0)
GRID = {  # T catch rate: shipped backdoors at H = 0, 0.01, 0.02, 0.05
    0.5: [0.049, 0.045, 0.039, 0.025], 0.6: [0.04, 0.035, 0.031, 0.018],
    0.7: [0.03, 0.027, 0.022, 0.011], 0.8: [0.021, 0.017, 0.015, 0.006],
    0.9: [0.01, 0.009, 0.007, 0.001],
}


def line(ref, br, h, t_skill):
    """(useful, backdoor, H used) of one seeded Trusted Editing run."""
    saved, out = ref.random, io.StringIO()
    ref.random = random.Random(0)
    try:
        with contextlib.redirect_stdout(out):
            ref.run_protocol("trusted_ed", br, h, N, t_skill=t_skill)
    finally:
        ref.random = saved
    return tuple(map(float, re.findall(r"=([\d.]+)", out.getvalue())))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    t0 = inspect.signature(ref.run_protocol).parameters["t_skill"].default
    return {
        "t_skill": t0,
        "grid": {t: [line(ref, BR, h, t)[1] for h in BUDGETS] for t in SKILLS},
        "demand": line(ref, BR, 1.0, t0)[2], "used": line(ref, BR, H, t0)[2],
        "attack": {br: line(ref, br, H, t0)[1] for br in ATTACKS},
    }


def verify(result):
    grid, attack = result["grid"], result["attack"]
    at_h = {t: row[1] for t, row in grid.items()}
    return [
        practice.Check(
            "ANSWER: the strongest assumption is T's catch rate on U's backdoors",
            all([grid == GRID, result["t_skill"] == 0.7, result["demand"] == 0.078,
                 result["used"] == H, round(H / result["demand"], 2) == 0.13]),
            f"shipped backdoors by T catch rate x H {BUDGETS}: {grid}; H demand "
            f"{result['demand']}, 1% covers {H / result['demand']:.0%}",
        ),
        practice.Check(
            "FINDING: ten points of T's catch rate outweigh the entire audit budget",
            all([round(at_h[0.6] - at_h[0.7], 3) == 0.008,
                 round(grid[0.7][0] - at_h[0.7], 3) == 0.003,
                 round(at_h[0.7] - grid[0.7][2], 3) == 0.005]),
            f"T 0.7 -> 0.6: +{at_h[0.6] - at_h[0.7]:.3f}; H 1% -> 0: "
            f"+{grid[0.7][0] - at_h[0.7]:.3f}; H 1% -> 2%: -{at_h[0.7] - grid[0.7][2]:.3f}",
        ),
        practice.Check(
            "FINDING: under full attack the bound is 1 - t_skill - H",
            all([attack == {0.1: 0.027, 0.2: 0.054, 1.0: 0.287},
                 round(abs(attack[1.0] - (1 - result["t_skill"] - H)), 3) <= 0.003]),
            f"shipped backdoors by attack rate at H = 1%: {attack}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
