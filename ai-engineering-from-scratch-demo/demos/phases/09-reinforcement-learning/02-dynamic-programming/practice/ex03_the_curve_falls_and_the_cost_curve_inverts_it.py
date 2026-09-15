"""Exercise 3 — the curve falls, and the cost curve inverts it.

    **Hard.** Build modified policy iteration: in the evaluation step, run only
    `k` sweeps instead of to convergence. Plot `V*(0,0)` error vs `k` for
    `k ∈ {1, 2, 5, 10, 50}`. What does the curve tell you about the
    evaluation/improvement tradeoff?

Reading of the exercise: "error" needs a referent the exercise does not give, so
it is measured against the fixed point the same operator reaches at `tol = 1e-14`,
and the plot ships as the table `audit_practice` asks for rather than a figure.
The evaluation block warm-starts from the current `V`, which is what makes this
modified policy iteration rather than the lesson's Step 4 with a smaller budget;
the cold-start variant is reported beside it, since the lesson's own
`policy_evaluation` cold-starts and the difference turns out to be the exercise.

**ANSWER: the curve falls, and off the bottom of the scale.** 8.50e-08, 1.29e-08,
2.19e-08, 2.24e-12 and exactly 0 at `k = 1, 2, 5, 10, 50` -- and it is not
monotone, `k = 2` beating `k = 5`.

**FINDING: that fall is not a tradeoff, it is overshoot.** Modified policy
iteration converges for every `k ≥ 1`, so all five reach `V*` and all five pass
the same stopping test; the plotted error is only how far past it the last
evaluation block happened to run.

**FINDING: priced per sweep the ordering inverts.** Reaching 1e-06 and staying
there costs 15 sweeps at `k = 1` and 209 at `k = 50` -- 14x more for an accuracy
both already have. The tradeoff curve the question is after is cost to fixed
accuracy, and it runs the other way to the one it asks for.

**MECHANISM: large `k` spends its budget on a policy that cannot terminate.**
The initial all-`up` policy never reaches the terminal, so its evaluation
converges at exactly `γ` toward -100. At 16 sweeps, where `k = 1` has already
stopped, `k = 5, 10, 50` are still 8.6 to 8.9 away from `V*`.

**FINDING: the `k` axis is a dial between two algorithms already in the lesson,
and the warm start is the dial.** `k = 1` is Step 5's value iteration -- 16 sweeps
against its 15 -- and `k → ∞` is Step 4's policy iteration at 3,728. Cold-started
the way the lesson's own `policy_evaluation` starts, `k = 1` does not converge at
all, while `k = 50` is unchanged: the low-`k` end, which is the end that wins,
exists only because `V` carries across the improvement step.

Structure: `mpi` is the one loop, parameterised by `k` and by warm start;
`first_below` reads cost-to-accuracy off its per-sweep trace.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "02-dynamic-programming"
GAMMA, TOL, KS = 0.99, 1e-6, (1, 2, 5, 10, 50)


def fixed_point():
    """`V*` to machine precision: the same operator, 10^8 tighter than the stopping rule."""
    ref = parity.load_reference(PHASE, LESSON, "main")
    return ref.value_iteration(gamma=GAMMA, tol=1e-14)[0][(0, 0)]


def block(ref, policy, values, k, trace, target):
    """`k` in-place evaluation sweeps of `policy`; returns the last max |dV|."""
    delta = 1.0
    for _ in range(k):
        delta = 0.0
        for state in ref.states():
            if state == ref.TERMINAL:
                continue
            new = ref.q_value(state, policy[state], values, GAMMA)
            delta = max(delta, abs(new - values[state]))
            values[state] = new
        trace.append(abs(values[(0, 0)] - target))
    return delta


def mpi(k, target, warm=True, cap=4000):
    """Modified policy iteration: `k` evaluation sweeps per improvement step."""
    ref = parity.load_reference(PHASE, LESSON, "main")
    zero = {s: 0.0 for s in ref.states()}
    values, policy, trace = dict(zero), {s: "up" for s in ref.states()}, []
    for outer in range(cap):
        if not warm:
            values = dict(zero)
        delta = block(ref, policy, values, k, trace, target)
        improved = ref.greedy_from_V(values, GAMMA)
        stable = improved == policy
        policy = improved
        if stable and delta < TOL:
            break
    return {"outer": outer + 1, "sweeps": len(trace), "error": trace[-1], "trace": trace}


def first_below(trace, eps):
    """Sweeps until the error is under `eps` and stays there; `None` if it never is."""
    for i, value in enumerate(trace):
        if value < eps and all(later < eps for later in trace[i:]):
            return i + 1
    return None


def solve():
    target = fixed_point()
    warm = {k: mpi(k, target) for k in KS}
    for k, row in warm.items():
        row["to_tol"] = first_below(row["trace"], TOL)
        row["at_16"] = row["trace"][min(15, len(row["trace"]) - 1)]
    return {"warm": warm, "cold": {k: mpi(k, target, warm=False) for k in (1, 50)}}


def stalled(warm):
    """At 16 sweeps, where k=1 has finished, the large blocks are still far from V*."""
    return min(warm[k]["at_16"] for k in (5, 10, 50)) > 8.0 and warm[1]["at_16"] < TOL


def warm_is_the_dial(warm, cold):
    """k=1 is value iteration, and only warm; k=50 does not notice the difference."""
    return (abs(warm[1]["sweeps"] - 16) <= 1 and cold[1]["error"] > 1.0
            and cold[50]["sweeps"] == warm[50]["sweeps"])


def verify(result):
    warm, cold = result["warm"], result["cold"]
    errors = [warm[k]["error"] for k in KS]
    cost = [warm[k]["to_tol"] for k in KS]
    table = "  ".join(f"k={k}: {warm[k]['error']:.2e} in {warm[k]['sweeps']} sweeps" for k in KS)
    return [
        practice.Check(
            "ANSWER: the curve falls to exactly zero, and not monotonically",
            min(errors[0] - 1e7 * max(errors[-1], 1e-16), errors[2] - errors[1]) > 0,
            f"error against the tol=1e-14 fixed point -- {table}. It falls at least 1e7x from "
            f"k=1 to k=50, which lands on that fixed point exactly, but k=2 ({errors[1]:.2e}) "
            f"beats k=5 ({errors[2]:.2e}), so the plot the exercise asks for is not a curve a "
            "tradeoff can be read off in the first place",
        ),
        practice.Check(
            "FINDING: the fall is overshoot, not a tradeoff -- every k reaches V*",
            max(errors) < TOL,
            f"modified policy iteration converges for every k >= 1, so all five land on V* and "
            f"all five clear the same max|dV| < {TOL:g} test -- the largest error among them is "
            f"{max(errors):.2e}. What the curve plots is how far past that test the final "
            "evaluation block happened to run, which is a property of the block size and not "
            "of the evaluation/improvement balance",
        ),
        practice.Check(
            "FINDING: priced per sweep the ordering inverts",
            min(cost[-1] - 10 * cost[0], warm[50]["sweeps"] - 10 * warm[1]["sweeps"]) > 0,
            f"sweeps to reach {TOL:g} and stay there: "
            + ", ".join(f"k={k} -> {warm[k]['to_tol']}" for k in KS)
            + f". k=50 pays {cost[-1] / cost[0]:.0f}x what k=1 pays for an accuracy k=1 "
            f"already has, and {warm[50]['sweeps']} total sweeps against "
            f"{warm[1]['sweeps']}. Cost to fixed accuracy is the tradeoff curve, and it runs "
            "opposite to the error-at-stop curve the exercise asks to plot",
        ),
        practice.Check(
            "MECHANISM: large k spends its budget on a policy that cannot terminate",
            stalled(warm),
            f"the initial all-'up' policy never reaches the terminal, so its evaluation "
            f"contracts at exactly gamma toward -100 and a large block chases it. At 16 sweeps, "
            f"where k=1 is already finished ({warm[1]['at_16']:.1e}), k=5, 10 and 50 are still "
            f"{warm[5]['at_16']:.1f}, {warm[10]['at_16']:.1f} and {warm[50]['at_16']:.1f} from "
            "V*. The improvement step is what makes a policy proper, and large k delays it",
        ),
        practice.Check(
            "FINDING: k is one dial between two of the lesson's algorithms, and warm start is it",
            warm_is_the_dial(warm, cold),
            f"k=1 is Step 5's value iteration: {warm[1]['sweeps']} sweeps against its 15, same "
            f"fixed point. k -> infinity is Step 4's policy iteration at 3,728 sweeps, measured "
            f"in exercise 2. What spans them is the warm start: cold-starting each evaluation "
            f"from V=0, as the lesson's own policy_evaluation does, leaves k=1 "
            f"{cold[1]['error']:.2f} from V* after "
            f"{cold[1]['sweeps']:,} sweeps -- one sweep from zero accumulates nothing "
            f"-- while k=50 is untouched at {cold[50]['sweeps']}. The end of the dial "
            "that wins is the end that only exists warm",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
