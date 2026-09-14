"""Exercise 3 — neither: the uniform policy is invariant under the slip.

    **Hard.** Turn the GridWorld stochastic: each action slips to an adjacent
    direction with probability `p = 0.1`. Re-evaluate the uniform policy. Does
    `V[start]` get better or worse? Why?

Reading of the exercise: the slip is applied to the *action*, before the
environment sees it, and it does not depend on the state -- the standard reading,
and the one that makes "slips to an adjacent direction" a property of the
actuator rather than of the board. That makes slip a stochastic matrix `K` on the
four directions, so the agent executing `π` under slip is exactly an agent
executing `π·K` without it, and the lesson's own `policy_evaluation` can evaluate
it unchanged. "Adjacent" is read as the two perpendicular directions, with the
looser reading (all three others) measured as a control.

**ANSWER: neither. `V[start]` does not move at all** -- -39.4116480543 before
and after, bit for bit, and no state on the board moves by more than 0.

**MECHANISM: `K` is doubly stochastic and the uniform policy is its fixed row.**
Every direction receives `1-p` from itself and `p/2` from each of the two
directions it is perpendicular to, so each column of `K` sums to 1. Composing:
`(π·K)(a') = Σ_a (1/4)·K(a'|a) = 1/4`. The slipped uniform policy *is* the
uniform policy, so it induces the same Markov chain, the same Bellman system and
the same fixed point.

**ROBUSTNESS: nothing here depends on `p` or on the reading.** Both readings of
"adjacent" give a doubly stochastic `K`, so `V` is unchanged at p = 0.1, 0.3, 0.5
and 0.9 alike. Even a total-slip actuator, p = 1.0, leaves the uniform policy's
value untouched.

**CONTROL: the environment really did become stochastic.** The same slip applied
to the lesson's own greedy down+right policy costs 0.77 of value, -7.59 to
-8.36, and the optimal value degrades from -5.85 to -6.43 -- so "better or worse"
has an answer for every policy on this board except the one the exercise names.

Structure: `kernel` is the slip matrix; `slipped` composes it onto a policy,
which is the whole mechanism in one function; `optimal_value` is value iteration
under the slipped dynamics, for the control.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "01-mdps-states-actions-rewards"
PERPENDICULAR = {"up": ("left", "right"), "down": ("left", "right"),
                 "left": ("up", "down"), "right": ("up", "down")}
GAMMA, SLIP, START = 0.99, 0.1, (0, 0)


def kernel(action, p, perpendicular=True):
    """The slip matrix row for `action`: 1-p on itself, p spread over the neighbours."""
    others = PERPENDICULAR[action] if perpendicular else \
        [a for a in PERPENDICULAR if a != action]
    row = {a: 0.0 for a in PERPENDICULAR}
    row[action] = 1.0 - p
    for other in others:
        row[other] += p / len(others)
    return row


def slipped(policy, p, perpendicular=True):
    """`pi` composed with the slip: what the actuator actually executes."""
    def effective(state):
        out = {a: 0.0 for a in PERPENDICULAR}
        for action, prob in policy(state).items():
            for direction, weight in kernel(action, p, perpendicular).items():
                out[direction] += prob * weight
        return out
    return effective


def optimal_value(ref, p, sweeps=3000, tol=1e-12):
    """Value iteration under the slipped dynamics; returns V*(start)."""
    values = {s: 0.0 for s in ref.all_states()}
    for _ in range(sweeps):
        new, delta = {}, 0.0
        for state in ref.all_states():
            new[state] = 0.0 if state == ref.TERMINAL else max(
                sum(w * (ref.step(state, d)[1] + GAMMA * values[ref.step(state, d)[0]])
                    for d, w in kernel(action, p).items())
                for action in ref.ACTIONS)
            delta = max(delta, abs(new[state] - values[state]))
        values = new
        if delta < tol:
            break
    return values[START]


def evaluate(ref, policy):
    return ref.policy_evaluation(policy, gamma=GAMMA, tol=1e-12, max_iter=50_000)


def column_error(ref, p):
    """How far the slip matrix is from doubly stochastic: worst column sum minus 1."""
    return max(abs(sum(kernel(a, p)[d] for a in ref.ACTIONS) - 1.0) for d in ref.ACTIONS)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    base = evaluate(ref, ref.uniform_policy)
    slip = evaluate(ref, slipped(ref.uniform_policy, SLIP))
    effective = slipped(ref.uniform_policy, SLIP)(START)
    return {
        "base": base[START],
        "slip": slip[START],
        "worst_state": max(abs(slip[s] - base[s]) for s in ref.all_states()),
        "effective_error": max(abs(p - 0.25) for p in effective.values()),
        "column_error": column_error(ref, SLIP),
        "sweep": {p: evaluate(ref, slipped(ref.uniform_policy, p))[START]
                  for p in (0.3, 0.5, 0.9, 1.0)},
        "loose": evaluate(ref, slipped(ref.uniform_policy, SLIP, False))[START],
        "greedy": (evaluate(ref, ref.greedy_policy)[START],
                   evaluate(ref, slipped(ref.greedy_policy, SLIP))[START]),
        "optimal": (optimal_value(ref, 0.0), optimal_value(ref, SLIP)),
    }


def verify(result):
    base, slip = result["base"], result["slip"]
    greedy_dry, greedy_wet = result["greedy"]
    sharp, blunt = result["optimal"]
    sweep = result["sweep"]
    worst_p = max(abs(v - base) for v in sweep.values())
    return [
        practice.Check(
            "ANSWER: neither -- V[start] does not move, bit for bit",
            abs(slip - base) < 1e-12 and result["worst_state"] < 1e-12,
            f"V[start] = {base:.10f} deterministic and {slip:.10f} with p={SLIP} slip, a "
            f"difference of {abs(slip - base):.1e}; the worst-moved state on the whole board "
            f"moves {result['worst_state']:.1e}. This is not a small change but no change: the "
            "question offers two answers and the MDP takes neither",
        ),
        practice.Check(
            "MECHANISM: the slip matrix is doubly stochastic, and uniform is its fixed row",
            result["column_error"] < 1e-15 and result["effective_error"] < 1e-15,
            "each direction receives 1-p from itself and p/2 from each perpendicular one, so "
            f"every column of K sums to 1 (worst {result['column_error']:.1e}). "
            "Composing, (pi.K)(a') = sum_a (1/4) K(a'|a) = 1/4 exactly (worst deviation "
            f"{result['effective_error']:.1e}), so the slipped uniform policy is the uniform "
            "policy and induces the same chain, the same Bellman system, the same fixed point",
        ),
        practice.Check(
            "ROBUSTNESS: the invariance survives every p and both readings of 'adjacent'",
            worst_p < 1e-12 and abs(result["loose"] - base) < 1e-12,
            "at p = 0.3, 0.5, 0.9, 1.0 the value is unchanged to "
            + f"{worst_p:.0e}, p=1.0 included -- an "
            "actuator that always slips. Reading 'adjacent' as all three other directions "
            f"instead gives {result['loose']:.10f}, also unchanged: any symmetric slip is "
            "doubly stochastic, so the reading the exercise leaves open cannot change the answer",
        ),
        practice.Check(
            "CONTROL: the board did become stochastic -- the greedy policy loses 0.77",
            greedy_wet < greedy_dry - 0.5,
            f"the same slip applied to the lesson's own down+right policy takes V[start] from "
            f"{greedy_dry:.3f} to {greedy_wet:.3f}, {100 * (greedy_wet / greedy_dry - 1):.1f}% "
            "worse. The slip is doing work; it is the uniform policy, not the implementation, "
            "that is blind to it",
        ),
        practice.Check(
            "FINDING: every policy on this board answers the question except this one",
            blunt < sharp - 0.5 and abs(slip - base) < 1e-12,
            f"under slip the optimal value falls from {sharp:.3f} to {blunt:.3f} and the greedy "
            f"policy from {greedy_dry:.3f} to {greedy_wet:.3f}, both worse, because a policy "
            "that prefers a direction can be pushed off it. A policy with no preference has "
            "nothing to be pushed off, so the exercise asks 'better or worse' of the one "
            "policy in its own lesson for which the question has no answer",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
