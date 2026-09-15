"""Exercise 1 — ten times the horizon buys one extra sweep.

    **Easy.** Run value iteration on the 4×4 GridWorld with `γ ∈ {0.9, 0.99}`.
    How many sweeps until `max |ΔV| < 1e-6`? Print `V*` as a 4×4 grid.

Reading of the exercise: "the 4×4 GridWorld" is ambiguous in this lesson and the
ambiguity is worth resolving rather than picking a side -- `code/main.py` hard-codes
`SLIP = 0.1` inside `transitions`, so the only board it ships is the stochastic one
Exercise 2 introduces as a change. Both arms are run: `SLIP = 0.0`, which is the
board the wording names, and the shipped `SLIP = 0.1`. Only the module constant is
touched; every sweep is the lesson's own `value_iteration`.

**ANSWER: 7 and 7 sweeps deterministic, 14 and 15 stochastic.** The grids are
printed below; `V*(0,0)` is -4.6856 / -5.8520 and -4.9888 / -6.4283.

**FINDING: the count is almost independent of the discount.** Ten times the
effective horizon buys one extra sweep on the stochastic board and none at all on
the deterministic one, where 7 = the board's 6-step diameter plus the sweep that
detects `delta < tol`.

**FINDING: the lesson's own contraction bound is valid and 122x conservative.**
`γ`-contraction in sup-norm predicts `log(ε(1-γ))/log γ` = 153 and 1833 sweeps
against the measured 14 and 15 -- high by 11x and 122x. The guarantee is not wrong:
`γ` is the sup-norm modulus of the Bellman *optimality* operator and is tight in the
worst case over all MDPs. It is simply far from tight on this board.

**MECHANISM: the observed rate is `γρ`, and `ρ` is a property of the board.** Once
the greedy policy stops changing, value iteration is linear iteration with `γP_π`,
so it converges asymptotically at `γ` times the spectral radius of that policy's
transient sub-matrix, `ρ = 0.3873` -- the same at both discounts. `γρ` is 0.349 and
0.383, so `γ` barely moves the local rate, and the in-place sweep does better still.
This is a rate for this board, not a replacement for the `γ` bound. On the
deterministic board `ρ = 0` exactly -- nilpotent, not merely small -- which is why 7
sweeps is the answer at every `γ`.

**FINDING: the grid moves more than the count does.** The two `V*` the exercise
asks to print differ by 1.44 at the start state while the sweep count it asks to
count differs by one, so the cheaper question is the one with the flat answer.

Structure: `run` is the lesson's `value_iteration` at one board and one discount;
`tail_ratio` measures the asymptotic per-sweep contraction; `rho` is power
iteration on the greedy policy's transient sub-matrix.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "02-dynamic-programming"
TOL, DIAMETER = 1e-6, 6


def board(slip):
    """The lesson's module with only its `SLIP` constant set."""
    ref = parity.load_reference(PHASE, LESSON, "main")
    ref.SLIP = slip
    return ref


def sweep_once(ref, values, gamma):
    """One in-place Bellman optimality sweep; returns max |dV| over it."""
    delta = 0.0
    for state in ref.states():
        if state == ref.TERMINAL:
            continue
        new = max(ref.q_value(state, a, values, gamma) for a in ref.ACTIONS)
        delta = max(delta, abs(new - values[state]))
        values[state] = new
    return delta


def tail_ratio(ref, gamma):
    """The per-sweep |dV| ratio once propagation across the board has finished."""
    values = {s: 0.0 for s in ref.states()}
    deltas = []
    while not deltas or deltas[-1] >= TOL:
        deltas.append(sweep_once(ref, values, gamma))
    usable = [d for d in deltas if d > 0]
    return usable[-1] / usable[-2] if len(usable) > 1 else 0.0


def push(ref, policy, vector, transient):
    """One application of the policy's transient sub-matrix to `vector`."""
    nxt = {s: 0.0 for s in transient}
    for state in transient:
        for succ, _reward, prob in ref.transitions(state, policy[state]):
            if succ in nxt:
                nxt[succ] += prob * vector[state]
    return nxt


def rho(ref, policy):
    """Spectral radius of the greedy policy's transition matrix over the transient states."""
    transient = [s for s in ref.states() if s != ref.TERMINAL]
    vector = {s: 1.0 for s in transient}
    scale = 0.0
    for _ in range(4000):
        nxt = push(ref, policy, vector, transient)
        scale, prior = max(nxt.values()), scale
        if scale == 0.0 or abs(scale - prior) < 1e-15:
            return scale                    # nilpotent, or converged
        vector = {s: v / scale for s, v in nxt.items()}
    return scale


def run(slip, gamma):
    ref = board(slip)
    values, policy, sweeps = ref.value_iteration(gamma=gamma, tol=TOL)
    return {"sweeps": sweeps, "start": values[(0, 0)], "rate": tail_ratio(ref, gamma),
            "grid": [[values[(r, c)] for c in range(ref.GRID)] for r in range(ref.GRID)],
            "rho": rho(ref, policy),
            "bound": math.log(TOL * (1 - gamma)) / math.log(gamma)}


def predict(row):
    """Diameter plus the sweeps the measured rate needs to cover `TOL`."""
    return DIAMETER + math.log(TOL / 1.05) / math.log(row["rate"])


def modulus_holds(sto9, sto99, det99, guess):
    """The gamma*rho rate, as one predicate: same rho, beaten by in-place, and nilpotent."""
    shared = abs(sto9["rho"] - sto99["rho"]) < 1e-9 and det99["rho"] == 0.0
    beats = sto9["rate"] < 0.81 * sto9["rho"] and sto99["rate"] < 0.891 * sto99["rho"]
    close = abs(guess[0] - sto9["sweeps"]) <= 2 and abs(guess[1] - sto99["sweeps"]) <= 2
    return shared and beats and close


def solve():
    return {(slip, g): run(slip, g) for slip in (0.0, 0.1) for g in (0.9, 0.99)}


def verify(result):
    det9, det99 = result[(0.0, 0.9)], result[(0.0, 0.99)]
    sto9, sto99 = result[(0.1, 0.9)], result[(0.1, 0.99)]
    rows = " | ".join(f"{r['grid'][0][0]:.4f}" for r in (det9, det99, sto9, sto99))
    guess = [predict(sto9), predict(sto99)]
    return [
        practice.Check(
            "ANSWER: 7 and 7 sweeps deterministic, 14 and 15 stochastic",
            (det9["sweeps"], det99["sweeps"], sto9["sweeps"], sto99["sweeps"]) == (7, 7, 14, 15),
            f"to max|dV| < {TOL:g} at gamma = 0.9 and 0.99: {det9['sweeps']} and {det99['sweeps']} "
            f"sweeps on the deterministic board, {sto9['sweeps']} and {sto99['sweeps']} on the "
            f"shipped SLIP=0.1 one. V*(0,0) across the four grids printed: {rows}. Only the "
            "lesson's SLIP constant is set; every sweep is its own value_iteration",
        ),
        practice.Check(
            "FINDING: 10x the horizon buys one sweep, and on one board none",
            (det99["sweeps"] - det9["sweeps"], sto99["sweeps"] - sto9["sweeps"]) == (0, 1),
            f"(1-gamma) falls 10x and the count moves {sto9['sweeps']} -> {sto99['sweeps']} on the "
            f"stochastic board and not at all on the deterministic one, where {det99['sweeps']} is "
            f"the {DIAMETER}-step diameter plus the sweep that observes delta < tol: the question "
            "the exercise asks has a nearly discount-free answer",
        ),
        practice.Check(
            "FINDING: the sup-norm contraction bound holds, and is 11x and 122x conservative",
            min(sto9["bound"] / sto9["sweeps"], sto99["bound"] / sto99["sweeps"]) > 10,
            f"log(eps(1-gamma))/log(gamma) -- geometric convergence at modulus gamma, the "
            f"lesson's stated guarantee, correct and worst-case tight but loose here -- predicts "
            f"{sto9['bound']:.0f} and {sto99['bound']:.0f} sweeps against a measured "
            f"{sto9['sweeps']} and {sto99['sweeps']}: high by "
            f"{sto9['bound'] / sto9['sweeps']:.0f}x and {sto99['bound'] / sto99['sweeps']:.0f}x",
        ),
        practice.Check(
            "MECHANISM: the observed rate is gamma*rho, and rho belongs to the board",
            modulus_holds(sto9, sto99, det99, guess),
            f"once the greedy policy settles, a sweep is linear iteration with gamma*P_pi, so the "
            f"asymptotic rate is gamma times the spectral radius of its transient block, rho = "
            f"{sto99['rho']:.4f}, the same at both discounts -- a local rate here, not a smaller "
            f"sup-norm modulus. gamma*rho is {0.9 * sto9['rho']:.3f} and "
            f"{0.99 * sto99['rho']:.3f} where gamma alone says 0.9 and 0.99; the in-place sweep "
            f"measures {sto9['rate']:.3f} and {sto99['rate']:.3f}, better than either, and "
            f"diameter + log(eps)/log(rate) predicts {guess[0]:.0f} and {guess[1]:.0f}. The "
            f"deterministic arm is the limit: rho is exactly {det99['rho']:.0f}, nilpotent rather "
            "than small, so that board is exact after the diameter at every gamma",
        ),
        practice.Check(
            "FINDING: the grid moves 1.44 while the count moves 1",
            abs(sto9["start"] - sto99["start"]) > 1.0,
            f"V*(0,0) goes {sto9['start']:.4f} -> {sto99['start']:.4f}, a shift of "
            f"{abs(sto9['start'] - sto99['start']):.2f}, while the count goes {sto9['sweeps']} -> "
            f"{sto99['sweeps']}. Discounting rescales the answer and barely touches the cost of "
            f"getting it -- and the smaller gamma reports the better-looking {sto9['start']:.2f} "
            f"against {sto99['start']:.2f} for the same policy, not a better one",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
