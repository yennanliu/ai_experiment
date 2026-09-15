"""Exercise 1 — a third of the runs never reach the terminal at all.

    **Easy.** Implement REINFORCE on 4×4 GridWorld with a linear softmax policy.
    Train for 1,000 episodes without a baseline. Plot the learning curve; measure
    variance (std of returns).

Reading of the exercise: the lesson ships `reinforce`, so "implement" is read as
*run it and find out what the std is measuring*. One run cannot answer that, so
60 seeds of the exercise's own setting are run and the outcome distribution is
reported; the learning curve ships as a table of the lesson's own `block_mean`.

**ANSWER: the curve converges to about -5.92 on the runs that converge -- and
33% of them do not.** 20 of 60 seeds end near **-63.3968**, which is
`-(1 - 0.99¹⁰⁰)/0.01`: the discounted return of a 100-step episode that never
reaches the terminal and is cut by `rollout`'s cap. 18 of those 20 sit on that
value to within one ULP, on every one of their last 200 episodes.

**FINDING: the "std of returns" is a mixture, not a noise level.** Inside a frozen
run it is exactly **0.0** -- every episode is the same capped trajectory. Inside a
converged run it is about 0.5. Across seeds it is 27, and that number is the
distance between the two outcomes, not the variance of either.

**MECHANISM: with no baseline and every reward negative, REINFORCE only ever
punishes.** Every return-to-go is strictly negative, so `adv < 0` at 100% of
updates; the gradient `(1[i=a] - p_i)·adv` therefore *lowers* the probability of
the action just taken and raises the other three, at every step of every episode.
Nothing is ever reinforced.

**FINDING: the curve is discounted and the target it is compared against is not.**
`returns_log` stores `returns[0]`, a `γ = 0.99` return, while the lesson's `main`
prints "optimal return on this 4x4 GridWorld = -6.0". The discounted optimum is
-5.8520, and -6.0 is the undiscounted one.

Structure: `outcome` runs the lesson's own `reinforce` at one seed; `CAP_RETURN`
is the closed-form value a capped episode must take, so a collapsed run is
identified exactly rather than by a threshold.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "06-policy-gradients-reinforce"
EPISODES, SEEDS, GAMMA, STEP_CAP, BLOCK = 1_000, 60, 0.99, 100, 200
CAP_RETURN = -(1 - GAMMA**STEP_CAP) / (1 - GAMMA)
CELLS = [(r, c) for r in range(4) for c in range(4)]


def outcome(ref, seed):
    """One run of the lesson's own `reinforce`, reduced to what the checks read."""
    _theta, log = ref.reinforce(EPISODES, use_baseline=False, rng=random.Random(seed))
    tail = log[-BLOCK:]
    return {"final": statistics.fmean(tail), "sd": statistics.pstdev(tail),
            "blocks": ref.block_mean(log, BLOCK), "capped": statistics.fmean(tail) < -60}


def optimal(ref, gamma):
    """`V*(0,0)` at `gamma`, swept from the lesson's own `step`."""
    values = {s: 0.0 for s in CELLS}
    for _ in range(60):
        values = {s: (0.0 if s == ref.TERMINAL else
                      max((lambda n, r, d: r + (0.0 if d else gamma * values[n]))(*ref.step(s, a))
                          for a in range(ref.N_ACTIONS))) for s in CELLS}
    return values[(0, 0)]


def all_negative(ref, seed=7, trials=200):
    """The fraction of returns-to-go that are negative, over fresh trajectories."""
    rng = random.Random(seed)
    theta, seen, negative = ref.init_theta(rng), 0, 0
    for _ in range(trials):
        for value in ref.returns_to_go(ref.rollout(theta, rng), GAMMA):
            seen += 1
            negative += value < 0.0
    return negative / seen


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runs = [outcome(ref, seed) for seed in range(SEEDS)]
    return {"runs": runs, "negative": all_negative(ref),
            "discounted": optimal(ref, GAMMA), "undiscounted": optimal(ref, 1.0)}


def split(runs):
    """(collapsed runs, converged runs, the collapsed ones frozen exactly on the cap)."""
    collapsed = [r for r in runs if r["capped"]]
    return (collapsed, [r for r in runs if not r["capped"]],
            [r for r in collapsed if r["sd"] == 0.0])


def digest(runs):
    """Every number the checks read, computed once."""
    collapsed, good, frozen = split(runs)
    return {"collapsed": collapsed, "good": good, "frozen": frozen,
            "across": statistics.pstdev([r["final"] for r in runs]),
            "median": statistics.median([r["final"] for r in good]),
            "good_sd": statistics.fmean(r["sd"] for r in good),
            "off": max(abs(r["final"] - CAP_RETURN) for r in collapsed),
            "ulp": max(abs(r["final"] - CAP_RETURN) for r in frozen)}


def verify(result):
    d = digest(result["runs"])
    collapsed, good, frozen = d["collapsed"], d["good"], d["frozen"]
    across = d["across"]
    return [
        practice.Check(
            "ANSWER: about -5.95 on the runs that converge, and a third of them do not",
            0.2 < len(collapsed) / SEEDS < 0.5 and -6.2 < d["median"] < -5.8,
            f"over {SEEDS} seeds of the exercise's own setting, {len(good)} runs converge to a "
            f"median final return of {d['median']:.2f} and {len(collapsed)} "
            f"({100 * len(collapsed) / SEEDS:.1f}%) do not. A converged curve by {BLOCK}-episode "
            "block: " + ", ".join(f"{b:.1f}" for b in good[0]["blocks"]),
        ),
        practice.Check(
            "FINDING: most collapsed runs sit on a closed-form value, not near one",
            len(frozen) > 0.8 * len(collapsed)
            and d["ulp"] < 1e-12,
            f"{len(frozen)} of the {len(collapsed)} collapsed runs end at exactly "
            f"{CAP_RETURN:.4f} = -(1 - {GAMMA}^{STEP_CAP})/(1 - {GAMMA}), the discounted return "
            f"of a {STEP_CAP}-step episode that never reaches the terminal -- to {d['ulp']:.1e}, "
            f"one ULP, on every one of their last {BLOCK} episodes. The other "
            f"{len(collapsed) - len(frozen)} still finish occasionally and land within "
            f"{d['off']:.3f}. The policy has not become poor; it has stopped finishing",
        ),
        practice.Check(
            "FINDING: the std the exercise asks for is a mixture of two populations",
            len(frozen) > 0.8 * len(collapsed) and across > 20,
            f"inside the {len(frozen)} frozen runs the std of the last {BLOCK} returns is exactly "
            f"0.0 -- every episode is the same capped trajectory. Inside a converged run it is "
            f"{d['good_sd']:.2f}. Across seeds it is {across:.1f}, "
            "which is the distance between the two outcomes rather than the variance of either. "
            "One number cannot describe a bimodal result, and the exercise asks for one number",
        ),
        practice.Check(
            "MECHANISM: with no baseline and every reward negative, REINFORCE only punishes",
            result["negative"] == 1.0,
            f"{100 * result['negative']:.0f}% of returns-to-go are strictly negative, so with no "
            f"baseline adv < 0 at every update. The gradient (1[i=a] - p_i)*adv then lowers the "
            f"probability of the action just taken and raises the other {3}, at every step of "
            "every episode: nothing is ever reinforced, only made less likely than its "
            "alternatives",
        ),
        practice.Check(
            "FINDING: the curve is discounted and the target it is read against is not",
            abs(result["discounted"] - result["undiscounted"]) > 0.1,
            f"returns_log stores returns[0], a gamma={GAMMA} return, while the lesson's main "
            f"prints 'optimal return on this 4x4 GridWorld = -6.0'. The discounted optimum is "
            f"{result['discounted']:.4f} and the undiscounted one {result['undiscounted']:.1f}, "
            f"{abs(result['discounted'] - result['undiscounted']):.3f} apart -- so a perfectly "
            "converged run is expected to sit just above the number it is being compared to",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
