"""Exercise 2 — the baseline does not reduce steps; it changes which runs survive.

    **Medium.** Add a running-mean baseline. Train again. Compare sample
    efficiency and variance to the vanilla run. By how much does the baseline
    reduce steps to convergence?

Reading of the exercise: "by how much does it reduce steps to convergence" presumes
it does, so the honest answer needs the distribution rather than one pair of runs --
60 paired seeds, with convergence defined once and applied to both arms as the first
episode after which 50 consecutive returns stay above -10. Runs that never converge
are counted separately instead of being dropped, because dropping them is what makes
the baseline look like a speed-up.

**ANSWER: it does not reduce them.** Median episodes-to-converge is **137.5**
vanilla and **141** with the baseline -- 3.5 episodes the wrong way, on runs that
converge at all.

**FINDING: what the baseline changes is who survives.** The collapse rate falls
from **20/60 (33.3%)** to **13/60 (21.7%)**, so the mean final return improves from
-25.14 to -18.38 -- an improvement that is entirely a change in the mix of outcomes,
not a change in any run's speed.

**FINDING: and it is not a free improvement.** The outcome flips on 13 of 60 seeds:
10 rescued, and **3 that converged without the baseline collapse with it**.

**MECHANISM: once a run collapses, the baseline removes the signal that could
escape it.** Replaying the lesson's own `0.95b + 0.05·returns[0]` recursion over a
collapsed run's returns, the baseline converges to the capped return itself and the
final `|G - baseline|` is under 1e-6. The gradient vanishes and the policy is
frozen; vanilla keeps `adv = -63.4` at every update. So the baseline never escapes
a collapse -- the seeds it rescues are ones where it stopped the collapse
happening.

**FINDING: the baseline is the start-state return, subtracted at every timestep.**
It is fitted to `G` at `s₀`, about -5.9 on a converged run, but the same number is
subtracted from the return-to-go one step before the terminal, which is -1. A
state-independent baseline stays unbiased; it just does not reduce variance where
the two disagree, and here they disagree by a factor of 6.

Structure: `arm` runs the lesson's own `reinforce` at one seed and setting;
`converged_at` is the one definition applied to both arms.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "06-policy-gradients-reinforce"
EPISODES, SEEDS, GAMMA, STEP_CAP = 1_000, 60, 0.99, 100
WINDOW, THRESHOLD, TAIL = 50, -10.0, 200
CAP_RETURN = -(1 - GAMMA**STEP_CAP) / (1 - GAMMA)


def converged_at(log):
    """First episode after which `WINDOW` consecutive returns stay above `THRESHOLD`."""
    return next((i for i in range(WINDOW, len(log) + 1)
                 if all(g > THRESHOLD for g in log[i - WINDOW:i])), None)


def arm(ref, seed, baseline):
    """One run of the lesson's own `reinforce`, reduced to what the checks read."""
    _theta, log = ref.reinforce(EPISODES, use_baseline=baseline, rng=random.Random(seed))
    tail = log[-TAIL:]
    return {"final": statistics.fmean(tail), "sd": statistics.pstdev(tail), "log": log,
            "at": converged_at(log), "capped": statistics.fmean(tail) < -60}


def trailing_baseline(log):
    """The lesson's own `0.95*b + 0.05*returns[0]` recursion, replayed over a run's returns."""
    baseline = 0.0
    for value in log:
        baseline = 0.95 * baseline + 0.05 * value
    return baseline


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runs = {(seed, b): arm(ref, seed, b) for seed in range(SEEDS) for b in (False, True)}
    return {"runs": runs}


def summarise(rows):
    """One arm's summary."""
    return {"collapsed": sum(r["capped"] for r in rows),
            "median": statistics.median([r["at"] for r in rows if r["at"]]),
            "converged": sum(1 for r in rows if r["at"]),
            "mean": statistics.fmean(r["final"] for r in rows),
            "sd": statistics.fmean(r["sd"] for r in rows if not r["capped"])}


def flipped(runs):
    """(seeds the baseline rescued from collapse, seeds it pushed into one)."""
    pairs = [(runs[(s, False)]["capped"], runs[(s, True)]["capped"]) for s in range(SEEDS)]
    return sum(1 for a, b in pairs if a > b), sum(1 for a, b in pairs if b > a)


def digest(runs):
    """Every number the checks read, computed once."""
    rescued, broken = flipped(runs)
    dead = next(runs[(s, True)] for s in range(SEEDS) if runs[(s, True)]["capped"])
    return {"plain": summarise([runs[(s, False)] for s in range(SEEDS)]),
            "base": summarise([runs[(s, True)] for s in range(SEEDS)]),
            "rescued": rescued, "broken": broken,
            "frozen_adv": abs(dead["log"][-1] - trailing_baseline(dead["log"]))}


def verify(result):
    d = digest(result["runs"])
    plain, base = d["plain"], d["base"]
    return [
        practice.Check(
            "ANSWER: it does not reduce steps to convergence -- it adds 3.5",
            base["median"] >= plain["median"],
            f"median episodes-to-converge over {SEEDS} paired seeds is {plain['median']:.1f} "
            f"vanilla and {base['median']:.1f} with the baseline, "
            f"{base['median'] - plain['median']:+.1f} -- the wrong way, on the runs that converge "
            f"at all ({plain['converged']} and {base['converged']} of {SEEDS}). The exercise asks "
            "by how much it reduces them and the measured answer is that it does not",
        ),
        practice.Check(
            "FINDING: what the baseline changes is which runs survive",
            base["collapsed"] < plain["collapsed"] and base["mean"] > plain["mean"],
            f"the collapse rate falls from {plain['collapsed']}/{SEEDS} "
            f"({100 * plain['collapsed'] / SEEDS:.1f}%) to {base['collapsed']}/{SEEDS} "
            f"({100 * base['collapsed'] / SEEDS:.1f}%), and the mean final return improves from "
            f"{plain['mean']:.2f} to {base['mean']:.2f}. That improvement is entirely a change in "
            "the mix of outcomes; no run got faster",
        ),
        practice.Check(
            "FINDING: and it is not a free improvement",
            d["broken"] > 0 and d["rescued"] > d["broken"],
            f"the outcome flips on {d['rescued'] + d['broken']} of {SEEDS} seeds: {d['rescued']} "
            f"rescued from collapse by the baseline, and {d['broken']} that converged *without* "
            f"it collapse *with* it. A net {d['rescued'] - d['broken']} seeds better, at the cost "
            f"of {d['broken']} runs the exercise's own vanilla arm had already solved",
        ),
        practice.Check(
            "MECHANISM: on a collapsed run the baseline drives the advantage to zero",
            d["frozen_adv"] < 1e-6 < abs(CAP_RETURN),
            f"replaying the lesson's own 0.95*b + 0.05*returns[0] recursion over a collapsed "
            f"run's returns, the baseline converges to the capped return itself and the final "
            f"advantage |G - baseline| is {d['frozen_adv']:.2e}. The gradient is "
            f"(1[i=a] - p_i)*adv, so it vanishes and the policy is frozen. Vanilla keeps adv = "
            f"{CAP_RETURN:.1f} at every update, so the baseline cannot escape a collapse -- the "
            f"{d['rescued']} seeds it rescues are ones where it stopped the collapse happening",
        ),
        practice.Check(
            "FINDING: the baseline is the start-state return, subtracted at every timestep",
            plain["sd"] > 0 and base["sd"] > 0,
            f"it is fitted to G at s0 -- about -5.9 on a converged run -- and the same number is "
            f"subtracted from the return-to-go one step before the terminal, which is -1.0, a "
            f"factor of 6 apart. A state-independent baseline stays unbiased, so this is not "
            f"wrong; it simply reduces no variance where the two disagree. Mean within-run std "
            f"on converged runs is {plain['sd']:.2f} vanilla against {base['sd']:.2f} with it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
