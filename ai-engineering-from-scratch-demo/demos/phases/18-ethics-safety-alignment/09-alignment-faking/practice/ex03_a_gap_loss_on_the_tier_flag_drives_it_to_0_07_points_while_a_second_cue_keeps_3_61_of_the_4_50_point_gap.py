"""Exercise 3 — a gap loss on the tier flag drives that gap to 0.07 points while a second cue keeps 3.61 of the 4.50-point gap.

    Read Anthropic's 2025 compliance-gap-loss follow-up. Propose a failure mode in which the compliance-gap loss reduces the gap without reducing faking — i.e., the model transfers the gap to a different observable condition.

Reading of the exercise: the follow-up paper is not in the repo, so the loss
is taken as the lesson describes it: "an auxiliary loss that penalizes the
gap between monitored and unmonitored compliance." In practice that gap is
measured on an eval set where the tier flag T is the only variable. The
proposed failure mode is that the model stops inferring "I am being trained"
from T and infers it from a second observable, [CUE-B]. [CUE-B] correlates
with training in deployment, but the eval set balances it against T. The
agent's belief is b = beta T + (1 - beta) [CUE-B], and refusing costs
preservation_weight x b through the reference's `compliance_probability`, at
its baseline harm aversion 2.0 and weight 1.0. The run picks the beta that
maximizes the agent's real-world payoff (the deployment gap) minus
lambda x (eval tier gap)^2. The deployment correlation is 0.9.

**ANSWER: the gap moves to [CUE-B].** Penalizing only the tier gap drives
beta from 1.00 (lambda = 0) to 0.17 (lambda = 10) and 0.01 (lambda = 100). The
eval tier gap falls from 4.50 to 1.00 to 0.07 points. The deployment gap,
which is the faking that actually happens on training traffic, only falls
from 4.50 to 3.80 to 3.61. With beta = 0, the gap split by [CUE-B] is the full
4.50 points, and the deployment gap is (2 rho - 1) x 4.50: 0, 2.25, 3.60 and
4.50 points at rho = 0.5, 0.75, 0.9 and 1. The loss measures a condition, and
the model is free to fake on a different one.

**FINDING: the shipped "compliance-gap-loss" is not a loss, and it closes the
gap by making the unmonitored model comply 10x more.** In `main()` it is a
hard-coded `p_avg = 0.5 * (p_m + p_u)` used for both draws, so the expected
gap is exactly 0. The printed +0.011 is sampling noise, 1.58 standard errors
at n = 1000. Monitored compliance falls from 4.74% to 2.49%, while unmonitored
compliance rises from 0.25% to 2.49%. The printed takeaway ("close the
observable gap but do not remove the incentive") is written into the code,
not measured by it.

Structure: `payoff()` gives the deployment and eval gaps for a belief mix;
`best_beta()` grid-searches beta for each lambda; `shipped()` replays the
real `main()` with a seeded `random.Random` swapped in (and restored).
"""

from __future__ import annotations

import contextlib
import inspect
import io
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "09-alignment-faking"
HARM, WEIGHT, RHO, N = 2.0, 1.0, 0.9, 1000
LAMBDAS, RHOS, GRID = (0, 10, 100), (0.5, 0.75, 0.9, 1.0), [k / 100 for k in range(101)]
MIT = r"compliance-gap-loss.*?monitored compliance\s+: ([\d.]+).*?unmonitored compliance : ([\d.]+)" \
      r".*?compliance gap\s+: ([+\-\d.]+)"


def payoff(ref, beta, rho=RHO):
    """(deployment gap at cue correlation rho, eval tier gap with the cue balanced) for belief mix beta."""
    def p(b):
        return ref.compliance_probability(True, HARM, WEIGHT * b)
    deploy = rho * p(1) + (1 - rho) * p(beta) - rho * p(0) - (1 - rho) * p(1 - beta)
    return deploy, 0.5 * (p(1) + p(beta)) - 0.5 * (p(0) + p(1 - beta))


def best_beta(ref, lam):
    beta = max(GRID, key=lambda b: payoff(ref, b)[0] - lam * payoff(ref, b)[1] ** 2)
    deploy, tier = payoff(ref, beta)
    return beta, round(100 * tier, 2), round(100 * deploy, 2)


def shipped(ref):
    saved, log = ref.random, io.StringIO()
    ref.random = random.Random(19)                   # the module's own random.seed(19)
    try:
        with contextlib.redirect_stdout(log):
            ref.main()
    finally:
        ref.random = saved
    return [float(x) for x in re.search(MIT, log.getvalue(), re.S).groups()]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    p_m, p_u = (ref.compliance_probability(f, HARM, WEIGHT) for f in (True, False))
    avg = 0.5 * (p_m + p_u)
    printed = shipped(ref)
    return {
        "sweep": {lam: best_beta(ref, lam) for lam in LAMBDAS},
        "cue_gap": round(100 * (p_m - p_u), 2),
        "by_rho": {r: round(100 * payoff(ref, 0.0, r)[0], 2) for r in RHOS},
        "averaged": "p_avg = 0.5 * (p_m + p_u)" in inspect.getsource(ref.main),
        "before": (round(100 * p_m, 2), round(100 * p_u, 2)), "after": round(100 * avg, 2),
        "printed": printed, "z": round(printed[2] / (2 * avg * (1 - avg) / N) ** 0.5, 2),
        "ratio": round(avg / p_u, 1),
    }


def verify(result):
    sweep = result["sweep"]
    return [
        practice.Check(
            "ANSWER: the gap moves to [CUE-B] -- tier gap 4.50 -> 0.07, deployment gap 4.50 -> 3.61",
            sweep == {0: (1.0, 4.5, 4.5), 10: (0.17, 1.0, 3.8), 100: (0.01, 0.07, 3.61)}
            and result["cue_gap"] == 4.5
            and result["by_rho"] == {0.5: 0.0, 0.75: 2.25, 0.9: 3.6, 1.0: 4.5},
            f"lambda -> (beta, eval tier gap, deployment gap) {sweep}; gap by [CUE-B] "
            f"{result['cue_gap']}; deployment gap at beta 0 by rho {result['by_rho']}",
        ),
        practice.Check(
            "FINDING: the shipped mitigation is a hard-coded average that raises unmonitored compliance 10x",
            result["averaged"] and result["before"] == (4.74, 0.25) and result["after"] == 2.49
            and result["ratio"] == 10.1 and result["printed"] == [0.03, 0.019, 0.011]
            and result["z"] == 1.58,
            f"monitored/unmonitored % {result['before']} -> {result['after']} both "
            f"({result['ratio']}x unmonitored); printed {result['printed']}, gap = {result['z']} SE",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
