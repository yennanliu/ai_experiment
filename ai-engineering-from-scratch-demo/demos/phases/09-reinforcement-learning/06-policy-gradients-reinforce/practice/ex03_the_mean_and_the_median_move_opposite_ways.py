"""Exercise 3 — the mean and the median move opposite ways, so there is no spot.

    **Hard.** Add an entropy bonus `β · H(π)`. Sweep `β ∈ {0, 0.01, 0.1, 1.0}`.
    Plot final return and policy entropy. Where is the sweet spot on this task?

Reading of the exercise: "final return" is ambiguous once a third of runs collapse
(exercise 1), and the ambiguity is the answer -- the mean and the median of the
same 60 seeds move in opposite directions across the sweep. The entropy gradient
is derived rather than approximated: for a softmax, `∂H/∂z_i = -p_i(log p_i + H)`,
added to the lesson's own policy-gradient term. The update is otherwise the
lesson's `reinforce` line for line; the inner feature loop is skipped where
`x[j] == 0`, which is exact because the features are one-hot and those terms add
0.0 (verified against the unskipped loop at 0.0e+00).

**ANSWER: there is no sweet spot -- there is a choice of statistic.** Across
`β = 0, 0.01, 0.1, 1.0` the mean final return is -25.2, -28.0, -23.7, -21.9 and
the median is -5.95, -5.99, -6.01, -6.89. The mean is not even monotone; the
median is, and it points the other way.

**FINDING: the two statistics move for different reasons.** The mean is dominated
by the collapse rate, which goes 33.3%, 38.3%, 30.0%, 26.7% -- not monotone, and
every step inside its own error bar of about ±6 points at 60 seeds. The median is
the quality of the runs that work, and entropy costs it steadily.

**FINDING: the entropy axis the exercise asks to plot never leaves the floor.**
Start-state entropy ends at 0.019, 0.008, 0.023, 0.124 against a uniform
`ln 4 = 1.386`. Even at `β = 1.0` the policy is at 9% of uniform: the sweep does
not span a meaningful range of the quantity it puts on one of its two axes.

**FINDING: the bonus is fighting a gradient that is much larger than it.** The
policy-gradient term carries `adv`, which is between -1 and -63 here, while the
entropy term carries `β·(-p_i)(log p_i + H)`, bounded by about `0.37β`. At
`β = 0.01` the bonus is four orders of magnitude smaller than what it opposes,
which is why that column is indistinguishable from `β = 0`.

Structure: `train` is the lesson's `reinforce` with the entropy term added;
`entropy` and its gradient are the only new mathematics.
"""

from __future__ import annotations

import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "06-policy-gradients-reinforce"
EPISODES, SEEDS, GAMMA, LR, TAIL = 1_000, 60, 0.99, 0.05, 200
BETAS, STEP_CAP = (0.0, 0.01, 0.1, 1.0), 100
CAP_RETURN = -(1 - GAMMA**STEP_CAP) / (1 - GAMMA)
UNIFORM = math.log(4)


def entropy(probs):
    """`H(p)` and the per-logit gradient `dH/dz_i = -p_i(log p_i + H)`."""
    logs = [math.log(p + 1e-12) for p in probs]
    total = -sum(p * lp for p, lp in zip(probs, logs))
    return total, [-p * (lp + total) for p, lp in zip(probs, logs)]


def train(ref, seed, beta):
    """The lesson's own `reinforce` update, plus `beta * dH/dz`."""
    rng = random.Random(seed)
    theta, log = ref.init_theta(rng), []
    for _ in range(EPISODES):
        traj = ref.rollout(theta, rng)
        returns = ref.returns_to_go(traj, GAMMA)
        for (x, a, _r, probs), total in zip(traj, returns):
            j = x.index(1.0)                    # one-hot: every other feature adds exactly 0.0
            _h, dh = entropy(probs)
            for i in range(ref.N_ACTIONS):
                grad = (1.0 if i == a else 0.0) - probs[i]
                theta[i][j] += LR * (total * grad + beta * dh[i])
        log.append(returns[0] if returns else 0.0)
    tail = log[-TAIL:]
    start, _ = entropy(ref.softmax(ref.logits(theta, ref.features((0, 0)))))
    return {"final": statistics.fmean(tail), "entropy": start,
            "capped": statistics.fmean(tail) < -60}


def one_variant(ref, seed, skip, episodes):
    """`episodes` of the entropy update, with the one-hot inner loop skipped or not."""
    rng = random.Random(seed)
    theta = ref.init_theta(rng)
    for _ in range(episodes):
        traj = ref.rollout(theta, rng)
        for (x, a, _r, probs), total in zip(traj, ref.returns_to_go(traj, GAMMA)):
            _h, dh = entropy(probs)
            for i in range(ref.N_ACTIONS):
                step = LR * (total * ((1.0 if i == a else 0.0) - probs[i]) + dh[i])
                for j in ([x.index(1.0)] if skip else range(ref.N_FEAT)):
                    theta[i][j] += step * x[j]
    return theta


def skip_is_exact(ref, seed=5, episodes=40):
    """The skipped inner loop against the unskipped one: the same theta, or it is not exact."""
    fast, slow = (one_variant(ref, seed, skip, episodes) for skip in (True, False))
    return max(abs(a - b) for ra, rb in zip(fast, slow) for a, b in zip(ra, rb))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runs = {beta: [train(ref, seed, beta) for seed in range(SEEDS)] for beta in BETAS}
    return {"runs": runs, "exact": skip_is_exact(ref)}


def column(rows):
    """One beta's summary."""
    finals = [r["final"] for r in rows]
    return {"mean": statistics.fmean(finals), "median": statistics.median(finals),
            "collapsed": sum(r["capped"] for r in rows),
            "entropy": statistics.fmean(r["entropy"] for r in rows)}


def columns(runs):
    """The four per-beta rows, transposed into the lists the checks read."""
    cols = {b: column(runs[b]) for b in BETAS}
    return {k: [cols[b][k] for b in BETAS] for k in ("mean", "median", "collapsed", "entropy")}


def monotone(values):
    """True when `values` never increases."""
    return all(a >= b for a, b in zip(values, values[1:]))


def verify(result):
    cols = columns(result["runs"])
    means, medians = cols["mean"], cols["median"]
    rates, ents = cols["collapsed"], cols["entropy"]
    error = 100 * (SEEDS * 0.33 * 0.67) ** 0.5 / SEEDS
    return [
        practice.Check(
            "ANSWER: no sweet spot -- the mean and the median point opposite ways",
            monotone(medians) and means[-1] > means[0],
            f"over {SEEDS} seeds per beta, mean final return across beta = {BETAS} is "
            + ", ".join(f"{m:.2f}" for m in means) + " and median is "
            + ", ".join(f"{m:.2f}" for m in medians)
            + f". The mean improves by {means[-1] - means[0]:.2f} from beta=0 to beta=1.0 while "
            f"the median falls by {medians[0] - medians[-1]:.2f}, monotonically. Which one you "
            "plot decides where the sweet spot is",
        ),
        practice.Check(
            "FINDING: the mean is tracking the collapse rate, and that is inside its error bar",
            not monotone(rates),
            "collapse rate by beta: "
            + ", ".join(f"{r}/{SEEDS} ({100 * r / SEEDS:.1f}%)" for r in rates)
            + f" -- not monotone, with beta=0.01 the worst of the four. At {SEEDS} seeds one "
            f"arm's standard error is about +/-{error:.1f} points, so every step in that row is "
            f"inside it. The mean final return is {means[-1] - means[0]:+.2f} across the sweep "
            "because it is a mixture whose weights are noise",
        ),
        practice.Check(
            "FINDING: the median is the one thing in the sweep that moves cleanly",
            medians[0] > medians[-1] + 0.5,
            f"median final return goes {medians[0]:.2f} -> {medians[-1]:.2f} across the sweep, "
            f"monotonically, a loss of {medians[0] - medians[-1]:.2f} on the runs that actually "
            f"converge. That is the entropy bonus doing exactly what it says: holding the policy "
            "away from the deterministic optimum, and being paid for in return",
        ),
        practice.Check(
            "FINDING: the entropy axis never leaves the floor",
            max(ents) < 0.15 * UNIFORM,
            "start-state entropy at the end of training, by beta: "
            + ", ".join(f"{e:.3f}" for e in ents)
            + f", against a uniform ln 4 = {UNIFORM:.4f}. Even at beta=1.0 the policy sits at "
            f"{100 * max(ents) / UNIFORM:.0f}% of uniform, and for the first three betas it is "
            f"under {100 * max(ents[:3]) / UNIFORM:.0f}%. The sweep does not span a meaningful "
            "range of the quantity the exercise puts on one of its two axes",
        ),
        practice.Check(
            "FINDING: the bonus is four orders of magnitude under what it opposes",
            result["exact"] == 0.0,
            f"the policy-gradient term carries adv, between -1 and {CAP_RETURN:.0f} here, while "
            f"the entropy term carries beta*(-p_i)(log p_i + H), bounded near 0.37*beta. At "
            f"beta=0.01 that is about 0.004 against a typical several -- which is why that column "
            f"is indistinguishable from beta=0. (The one-hot inner-loop skip is exact: theta "
            f"matches the unskipped loop to {result['exact']:.1e}.)",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
