"""Exercise 2 — the first three evaluations are the whole bill.

    **Medium.** Compare policy iteration vs value iteration on the *stochastic*
    GridWorld (slip probability `0.1`). Count: sweeps, wall-clock time, final
    `V*(0,0)`. Which converges faster in iterations? In wall-clock?

Reading of the exercise: "count sweeps" is read strictly -- a sweep is one pass
over the 15 non-terminal states, wherever it happens, so policy iteration's inner
evaluation sweeps are counted rather than its outer iterations. That reading is
forced: the lesson's `policy_iteration` returns `it + 1` and carries a local named
`sweeps` that also counts outer iterations, so the one number a learner would
reach for is not the one the exercise names. Bellman backups are reported beside
the sweeps as the machine-independent cost, since wall-clock is not reproducible.

**ANSWER: value iteration, on every honest unit.** 15 sweeps against 3,728, 960
backups against 56,220, and about 60x less wall-clock. Both land on
`V*(0,0) = -6.4283`.

**FINDING: the only unit policy iteration wins is the one that is not a sweep.**
5 outer iterations against 15 sweeps looks like a 3x win and is a unit error --
each of those 5 contains a full evaluation to `tol`.

**FINDING: 99.2% of the bill is three evaluations that get thrown away.** Inner
sweeps per outer iteration are 1329, 1313, 1056, 15, 15. The first three evaluate
policies that never reach the terminal, so their modulus is exactly `γ` and each
costs more than the entire value-iteration run -- and each result is discarded one
improvement step later.

**FINDING: the gap widens with `γ`, because that waste is the only `γ`-sensitive
part.** Policy iteration costs 7.2x the backups at `γ = 0.9` and 58.6x at
`γ = 0.99`; value iteration itself goes 14 sweeps to 15.

**FINDING: the two agree with each other 10x better than either agrees with
`V*`.** Sup-norm between them is 6.3e-09; each is 6.1e-08 from the fixed point the
same operator reaches at `tol = 1e-14`, in the same direction, because one
stopping rule stops them both short -- and that fixed point is only 10 sweeps
further on.

Structure: `run_pi` re-runs the lesson's policy iteration with the sweeps
instrumented; `run_vi` calls the lesson's
`value_iteration` untouched, and at `tol = 1e-14` it is also the reference `V*`.
"""

from __future__ import annotations

import time

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "02-dynamic-programming"
TOL = 1e-6


def counted(ref):
    """The lesson's `q_value`, wrapped so Bellman backups can be counted."""
    tally, original = [0], ref.q_value

    def wrapper(state, action, values, gamma):
        tally[0] += 1
        return original(state, action, values, gamma)

    ref.q_value = wrapper
    return tally


def evaluate(ref, policy, gamma):
    """The lesson's `policy_evaluation` for a deterministic policy, with sweeps counted."""
    values = {s: 0.0 for s in ref.states()}
    for sweep in range(20_000):
        delta = 0.0
        for state in ref.states():
            if state == ref.TERMINAL:
                continue
            new = ref.q_value(state, policy[state], values, gamma)
            delta = max(delta, abs(new - values[state]))
            values[state] = new
        if delta < TOL:
            break
    return values, sweep + 1


def run_pi(gamma):
    ref = parity.load_reference(PHASE, LESSON, "main")
    tally = counted(ref)
    policy, per_outer = {s: "up" for s in ref.states()}, []
    start = time.perf_counter()
    for outer in range(100):
        values, sweeps = evaluate(ref, policy, gamma)
        per_outer.append(sweeps)
        improved = ref.greedy_from_V(values, gamma)
        if improved == policy:
            break
        policy = improved
    return {"outer": outer + 1, "per_outer": per_outer, "sweeps": sum(per_outer),
            "backups": tally[0], "seconds": time.perf_counter() - start, "values": values,
            "start": values[(0, 0)]}


def run_vi(gamma, tol=TOL):
    """The lesson's `value_iteration`, untouched. At `tol=1e-14` this is the reference V*."""
    ref = parity.load_reference(PHASE, LESSON, "main")
    tally = counted(ref)
    start = time.perf_counter()
    values, _policy, sweeps = ref.value_iteration(gamma=gamma, tol=tol)
    elapsed = time.perf_counter() - start
    return {"sweeps": sweeps, "backups": tally[0], "seconds": elapsed,
            "start": values[(0, 0)], "values": values}


def vi_wins(pi, vi):
    """Fewer sweeps, fewer backups, less wall-clock, same answer -- as one predicate."""
    cheaper = vi["sweeps"] < pi["sweeps"] and vi["backups"] < pi["backups"]
    return cheaper and vi["seconds"] < pi["seconds"] and abs(pi["start"] - vi["start"]) < 1e-6


def front_loaded(pi, vi, head):
    """The first three evaluations are the bill and none of the rest comes close."""
    dominant = sum(head) / pi["sweeps"] > 0.98 and min(head) > vi["sweeps"]
    return dominant and max(pi["per_outer"][3:]) <= vi["sweeps"] + 1


def deviations(pi, vi, tight):
    """(how far the two sit from each other, how far either sits from the fixed point)."""
    return (max(abs(pi["values"][s] - vi["values"][s]) for s in vi["values"]),
            max(abs(vi["values"][s] - tight[s]) for s in tight))


def solve():
    return {"pi": {g: run_pi(g) for g in (0.9, 0.99)}, "reference": run_vi(0.99, tol=1e-14),
            "vi": {g: run_vi(g) for g in (0.9, 0.99)}}


def verify(result):
    pi, vi, pi9, vi9 = (result[k][g] for g in (0.99, 0.9) for k in ("pi", "vi"))
    tight, deep = result["reference"]["values"], result["reference"]["sweeps"]
    head, agree, off = pi["per_outer"][:3], *deviations(pi, vi, tight)
    widening = (pi["backups"] / vi["backups"]) / (pi9["backups"] / vi9["backups"])
    return [
        practice.Check(
            "ANSWER: value iteration wins in sweeps, backups and wall-clock",
            vi_wins(pi, vi),
            f"at gamma=0.99: {vi['sweeps']} sweeps against {pi['sweeps']}, {vi['backups']:,} "
            f"backups against {pi['backups']:,} ({pi['backups'] / vi['backups']:.1f}x), and "
            f"{1000 * vi['seconds']:.2f} ms against {1000 * pi['seconds']:.2f} ms "
            f"({pi['seconds'] / vi['seconds']:.0f}x). Both land on V*(0,0) = {vi['start']:.4f}. "
            "The exercise asks which wins in iterations and which in wall-clock as if the two "
            "could differ here; they do not",
        ),
        practice.Check(
            "FINDING: the one unit policy iteration wins is not a sweep",
            pi["outer"] < vi["sweeps"] < pi["sweeps"],
            f"{pi['outer']} outer iterations against {vi['sweeps']} sweeps reads as a "
            f"{vi['sweeps'] / pi['outer']:.0f}x win and is a unit error: each outer iteration "
            "contains a full evaluation to tol. The lesson's policy_iteration returns it+1 and "
            "its local named `sweeps` counts the same thing, so the number a learner reaches for "
            "is the only one that flatters the loser",
        ),
        practice.Check(
            "FINDING: three discarded evaluations are 99% of the bill",
            front_loaded(pi, vi, head),
            f"inner sweeps per outer iteration are {pi['per_outer']}: the first three are "
            f"{100 * sum(head) / pi['sweeps']:.1f}% of the total. They evaluate policies that "
            "never reach the terminal -- all-'up' and its first two successors -- so the "
            "transient spectral radius is 1, the modulus is exactly gamma, each costs more than "
            f"the whole {vi['sweeps']}-sweep value iteration, and all three are then discarded",
        ),
        practice.Check(
            "FINDING: the gap widens with gamma because only the waste is gamma-bound",
            min(0.2 * pi["backups"] / vi["backups"] - pi9["backups"] / vi9["backups"],
                1 - vi["sweeps"] + vi9["sweeps"]) >= 0,
            f"policy iteration costs {pi9['backups'] / vi9['backups']:.1f}x the backups at "
            f"gamma=0.9 and {pi['backups'] / vi['backups']:.1f}x at gamma=0.99, an "
            f"{widening:.1f}x widening, while value iteration goes {vi9['sweeps']} sweeps to "
            f"{vi['sweeps']}. The discount bites only where a policy cannot terminate, the one "
            "thing policy iteration does and value iteration skips",
        ),
        practice.Check(
            "FINDING: they agree with each other 10x better than with V*",
            max(agree - off / 5, off - TOL * 99, deep - 2 * vi["sweeps"]) < 0,
            f"sup-norm |V_pi - V_vi| = {agree:.2e}, while each sits {off:.2e} from the fixed "
            f"point the same operator reaches at tol=1e-14 -- same size, same direction, because "
            f"one stopping rule stops them both short. That fixed point costs {deep} sweeps "
            f"against {vi['sweeps']}, so {deep - vi['sweeps']} more buy eight orders of "
            "magnitude: it is the tolerance, not the algorithm, that pays here",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
