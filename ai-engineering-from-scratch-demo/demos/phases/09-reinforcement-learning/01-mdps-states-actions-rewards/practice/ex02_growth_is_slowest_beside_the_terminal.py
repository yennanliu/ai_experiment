"""Exercise 2 — the values near the terminal are the ones that grow least.

    **Medium.** Run `policy_evaluation` with `γ ∈ {0.5, 0.9, 0.99}` for the
    uniform-random policy. Print `V` as a 4×4 grid for each. Explain why the
    state values near the terminal grow faster with larger `γ`.

Reading of the exercise: "grow" is read as growth in magnitude, since every
value here is negative and all three grids move down as `γ` rises -- the reading
that makes the question non-trivial. "Near the terminal" is read as a ring of
Manhattan distance to (3,3), so the claim becomes a statement about growth as a
function of that distance and can be measured rather than argued. The explanation
is therefore attempted on the premise first, and the premise does not survive it.

**ANSWER: every value grows toward a ceiling of 1/(1-γ) -- 2, 10, 100.** The
start state reaches 100%, 94% and 39% of that ceiling at the three discounts, so
the ceiling rises 50x while the fraction of it actually reached falls.

**FINDING: the premise is backwards, monotonically.** From γ=0.5 to γ=0.99 the
two states adjacent to the terminal grow 12.0x; the start corner grows 19.7x.
Ring by ring the factor is 12.0, 15.2, 17.1, 18.2, 19.1, 19.7 -- it increases
with distance from the terminal at every step, and the states the exercise names
are the slowest-growing ones on the board.

**MECHANISM: raising γ can only restore return that a shorter horizon cut off.**
A state next to the terminal has the least return out past any horizon, by
definition of being next to it. In closed form `V(s) = -(1 - E[γ^T])/(1-γ)`, so
`(1-γ)|V|` is the fraction of the ceiling a state reaches: at γ=0.99 that is 20%
beside the terminal and 39% at the start. Growth is the 50x ceiling rise scaled
by the ratio of those fractions -- 50 x 0.204/0.848 = 12.0 and 50 x 0.394/1.000
= 19.7.

**FINDING: γ=0.5 collapses the grid instead of ranking it.** An effective horizon
of 2 steps on a board 6 deep leaves 13 of the 15 non-terminal states inside
0.08 of each other, and the whole grid inside 0.30 of a ceiling of 2.

**CONTROL: the shipped `policy_evaluation` is not the one the lesson prints.**
Step 3's snippet assigns `V[s] = v` in place; `code/main.py` builds `new_values`
and updates synchronously. Same fixed point, 1.5x the sweeps -- and in-place
versus synchronous is exactly what the next lesson's Key Terms table contrasts.

Structure: `evaluate` is the lesson's own loop with the doc's in-place variant as
a switch and a sweep counter; `rings` groups states by distance to the terminal.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "01-mdps-states-actions-rewards"
GAMMAS = (0.5, 0.9, 0.99)
START, BESIDE = (0, 0), (2, 3)


def evaluate(ref, gamma, inplace=False, tol=1e-12, max_iter=50_000):
    """The lesson's `policy_evaluation`, plus the doc's in-place read and a sweep count."""
    values = {s: 0.0 for s in ref.all_states()}
    for sweeps in range(1, max_iter + 1):
        delta, new = 0.0, dict(values)
        for state in ref.all_states():
            if state == ref.TERMINAL:
                continue
            source = new if inplace else values
            total = 0.0
            for action, prob in ref.uniform_policy(state).items():
                nxt, reward, _ = ref.step(state, action)
                total += prob * (reward + gamma * source[nxt])
            delta = max(delta, abs(total - values[state]))
            new[state] = total
        values = new
        if delta < tol:
            return values, sweeps
    return values, max_iter


def rings(ref):
    """Non-terminal states grouped by Manhattan distance to the terminal."""
    out = {}
    for state in ref.all_states():
        distance = abs(state[0] - ref.TERMINAL[0]) + abs(state[1] - ref.TERMINAL[1])
        if distance:
            out.setdefault(distance, []).append(state)
    return out


def growth(low, high, states):
    """Mean magnitude ratio between two discounts over a set of states."""
    return sum(high[s] / low[s] for s in states) / len(states)


def reached(grids, state):
    """(1-gamma)|V(state)| per gamma: the share of the 1/(1-gamma) ceiling reached."""
    return [(1 - g) * abs(grids[g][state]) for g in GAMMAS]


def sweep_counts(ref):
    """(synchronous, in-place) sweeps to tol=1e-6, per gamma."""
    return [(evaluate(ref, g, tol=1e-6)[1], evaluate(ref, g, True, tol=1e-6)[1])
            for g in GAMMAS]


def flatness(grids):
    """(spread of the gamma=0.5 grid, how many states sit within 0.08 of its floor)."""
    values = [v for v in grids[GAMMAS[0]].values() if v]
    return max(values) - min(values), sum(1 for v in values if v - min(values) < 0.08)


def joined(values, spec):
    """`values` formatted and comma-joined, for a one-line detail string."""
    return ", ".join(spec.format(v) for v in values)


def monotone_out(growths):
    """Growth rises with distance from the terminal -- the premise, inverted."""
    return growths == sorted(growths) and growths[0] < growths[-1]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    grids = {g: evaluate(ref, g)[0] for g in GAMMAS}
    by_ring = rings(ref)
    lo, hi = grids[GAMMAS[0]], grids[GAMMAS[-1]]
    spread, flat = flatness(grids)
    return {
        "starts": [grids[g][START] for g in GAMMAS],
        "ring_growth": [growth(lo, hi, by_ring[d]) for d in sorted(by_ring)],
        "start_share": reached(grids, START),
        "beside_share": reached(grids, BESIDE),
        "spread": spread,
        "flat_states": flat,
        "sweeps": sweep_counts(ref),
        "inplace_agrees": max(abs(evaluate(ref, 0.99, True)[0][s] - hi[s]) for s in hi),
    }


def verify(result):
    growths, starts = result["ring_growth"], result["starts"]
    near, far = growths[0], growths[-1]
    share, beside, sweeps = result["start_share"], result["beside_share"], result["sweeps"]
    return [
        practice.Check(
            "ANSWER: every value grows toward the ceiling 1/(1-gamma) = 2, 10, 100",
            all(s <= 1.0 + 1e-9 for s in share) and abs(starts[-1]) > abs(starts[0]),
            "V(0,0) = " + joined(starts, "{:.2f}") + " at gamma = 0.5, 0.9, 0.99, which is "
            + joined(share, "{:.0%}") + " of the ceiling. The ceiling rises 50x and the "
            "fraction of it reached falls: a longer horizon is worth only what lies inside it",
        ),
        practice.Check(
            "FINDING: growth is monotone in distance, so the premise is backwards",
            monotone_out(growths),
            "ring by ring out from the terminal the magnitude ratio between gamma=0.5 and "
            "gamma=0.99 is " + joined(growths, "{:.1f}x")
            + f" -- monotone increasing. The two states beside the terminal grow {near:.1f}x and "
            f"the start corner {far:.1f}x, a factor of {far / near:.2f} the other way, so the "
            "states the exercise names as the fastest are the slowest on the board",
        ),
        practice.Check(
            "MECHANISM: growth is the ceiling rise scaled by the fraction of it reached",
            abs(near - 50 * beside[-1] / beside[0]) < 0.2
            and abs(far - 50 * share[-1] / share[0]) < 0.2,
            "V(s) = -(1 - E[gamma^T])/(1 - gamma), so (1-gamma)|V| is the fraction of the "
            f"ceiling a state reaches: {beside[-1]:.0%} beside the terminal against "
            f"{share[-1]:.0%} at the start. Growth is the 50x ceiling rise times the ratio of "
            f"those fractions -- {near:.1f}x and {far:.1f}x -- and a state beside the terminal "
            "has the least return outside any horizon to restore",
        ),
        practice.Check(
            "FINDING: gamma=0.5 collapses the grid rather than ranking it",
            result["spread"] < 0.35 and result["flat_states"] >= 12,
            f"an effective horizon of 2 steps on a board 6 deep puts all 15 non-terminal states "
            f"inside {result['spread']:.2f} of one another on a ceiling of 2, "
            f"{result['flat_states']} of them inside 0.08. The grid the exercise asks for is "
            "nearly constant at the smallest gamma, so it cannot order the states it shows",
        ),
        practice.Check(
            "CONTROL: the shipped evaluator is synchronous, the printed one is in place",
            result["inplace_agrees"] < 1e-6 and all(f < s for s, f in sweeps),
            "Step 3's snippet writes V[s] = v and reads it back within the sweep; code/main.py "
            f"builds new_values and updates synchronously. Same fixed point to "
            f"{result['inplace_agrees']:.1e}, but " + ", ".join(f"{s} vs {f}" for s, f in sweeps)
            + " sweeps to tol=1e-6 at gamma = 0.5, 0.9, 0.99 -- in-place against synchronous is "
            "the distinction the next lesson's Key Terms table teaches",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
