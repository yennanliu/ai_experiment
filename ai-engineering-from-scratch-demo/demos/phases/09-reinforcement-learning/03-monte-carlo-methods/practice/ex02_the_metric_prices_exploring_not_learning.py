"""Exercise 2 — the metric prices the exploring, not the learning.

    **Medium.** Implement ε-greedy MC control with `ε ∈ {0.01, 0.1, 0.3}`.
    Compare mean return after 20,000 episodes. What does the curve look like?
    Where does the bias-variance tradeoff live?

Reading of the exercise: the lesson ships `mc_control`, so the three runs are its
own loop at three `epsilon` values and one seed. "Mean return after 20,000
episodes" is reported both over the whole run and over the last 5,000, since the
two differ only by the learning transient. "Where does the tradeoff live" is
answered by asking what each run actually learned, which the stated metric does
not measure -- so the recovered greedy policy is evaluated exactly, everywhere on
the board, against value iteration over the lesson's own `step`.

**ANSWER: -6.038, -6.585, -8.060 over the run; -5.930, -6.561, -7.990 over the
last 5,000.** Monotone in `ε`, and flat after the first few thousand episodes.

**FINDING: that ordering is fixed before a single episode runs.** The exact
on-policy value of an ε-soft policy over the optimal action is -5.9041, -6.4208
and -7.9906; each run lands within 0.14 of its own ceiling. The metric measures
what exploring costs, not how well the run learned.

**FINDING: all three learn the same thing.** The greedy policy each one recovers
is exactly optimal from the start state -- `V(0,0) = -5.8520` at every `ε` -- so
the number being compared is the one quantity that does not separate them.

**FINDING: the tradeoff lives off the measured path.** At `ε = 0.01` four states
are worth less than `V*` under the recovered policy, of which three choose a
suboptimal action themselves and the fourth inherits the loss from a successor --
two counts worth keeping apart. One of the three is the top-right corner, where the
learned action walks into the wall forever: -100 against an optimal -2.97.
`ε = 0.3` leaves none. Every episode starts at (0,0), so the stated metric cannot
see any of it.

**CONTROL: exploration is a threshold here, not a dial.** At `ε = 0` the same
code touches 6 of 60 state-action pairs and returns a policy worth -100, mean
return -86.6. From 0 to 0.01 the outcome moves 94 points; from 0.01 to 0.3 it
moves 2.1, and all of that is cost.

Structure: `evaluate` and `optimal` sweep the lesson's own `step` for exact
values; `soft_value` is the same sweep under an ε-soft policy, which is what the
measured mean is compared against.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "03-monte-carlo-methods"
GAMMA, EPISODES, SEED = 0.99, 20_000, 20260914
EPSILONS = (0.01, 0.1, 0.3)


def fixpoint(ref, backup, tol=1e-13):
    """Sweep `backup` over the board until it stops moving."""
    values = {s: 0.0 for s in ref.states()}
    for _ in range(200_000):
        nxt = {s: (0.0 if s == ref.TERMINAL else backup(s, values)) for s in ref.states()}
        if max(abs(nxt[s] - values[s]) for s in values) < tol:
            return nxt
        values = nxt
    return values


def look(ref, state, action, values):
    """One step of the lesson's own `step`, backed up."""
    nxt, reward, done = ref.step(state, action)
    return reward + (0.0 if done else GAMMA * values[nxt])


def optimal(ref):
    """`V*` and the greedy policy it induces, from the lesson's own transitions."""
    values = fixpoint(ref, lambda s, v: max(look(ref, s, a, v) for a in ref.ACTIONS))
    return values, {s: max(ref.ACTIONS, key=lambda a: look(ref, s, a, values))
                    for s in ref.states()}


def evaluate(ref, policy):
    """`V^policy`, exactly, for a deterministic policy."""
    return fixpoint(ref, lambda s, v: look(ref, s, policy[s], v))


def soft_value(ref, greedy, eps):
    """`V` of the eps-soft policy greedy toward `greedy` -- the ceiling for that eps."""
    return fixpoint(ref, lambda s, v: sum(
        (eps / 4 + (1 - eps if a == greedy[s] else 0.0)) * look(ref, s, a, v)
        for a in ref.ACTIONS))


def run(ref, eps, episodes=EPISODES):
    _q, greedy, log = ref.mc_control(episodes=episodes, gamma=GAMMA, epsilon=eps,
                                     rng=random.Random(SEED))
    return {"mean": statistics.fmean(log), "tail": statistics.fmean(log[-5000:]),
            "greedy": {s: greedy.get(s, "up") for s in ref.states()},
            "tried": sum(1 for s in _q for a in _q[s] if _q[s][a] != 0.0)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    star, best = optimal(ref)
    runs = {}
    for eps in EPSILONS:
        row = run(ref, eps)
        learned = evaluate(ref, row["greedy"])
        runs[eps] = {**row, "start": learned[(0, 0)], "ceiling": soft_value(ref, best, eps)[(0, 0)],
                     "wrong": [(s, row["greedy"][s], learned[s], star[s]) for s in ref.states()
                               if s != ref.TERMINAL and abs(learned[s] - star[s]) > 1e-9],
                     "misact": [s for s in ref.states() if s != ref.TERMINAL
                                and abs(look(ref, s, row["greedy"][s], star) - star[s]) > 1e-9]}
    control = run(ref, 0.0, episodes=5_000)
    return {"runs": runs, "optimal": star[(0, 0)], "control": control,
            "control_start": evaluate(ref, control["greedy"])[(0, 0)]}


def worst(row):
    """The state that loses most to the optimum under the recovered policy."""
    return max(row["wrong"], key=lambda w: w[3] - w[2], default=(None, None, 0.0, 0.0))


def widest(runs, left, right, offset=0.0):
    """The largest gap between two per-eps quantities, across the three runs."""
    return max(abs(runs[e][left] - (runs[e][right] if right else offset)) for e in EPSILONS)


def verify(result):
    runs, star = result["runs"], result["optimal"]
    low, high = runs[0.01], runs[0.3]
    table = ", ".join(f"eps={e}: {runs[e]['mean']:+.3f} / {runs[e]['tail']:+.3f}" for e in EPSILONS)
    gaps = [worst(runs[e])[3] - worst(runs[e])[2] for e in EPSILONS]
    hole = worst(low)
    return [
        practice.Check(
            "ANSWER: -6.04, -6.59, -8.06 over the run, monotone in epsilon",
            low["mean"] > runs[0.1]["mean"] > high["mean"],
            f"mean return over all {EPISODES:,} episodes / over the last 5,000 -- {table}. The "
            f"two agree to within {widest(runs, 'mean', 'tail'):.3f}, so the curve asked about is "
            "flat after the first few thousand episodes and the whole-run mean is the converged "
            "mean plus a short transient",
        ),
        practice.Check(
            "FINDING: the ordering is fixed by epsilon before a single episode runs",
            widest(runs, 'tail', 'ceiling') < 0.2,
            "the exact on-policy value of an eps-soft policy over the optimal action is "
            + ", ".join(f"{runs[e]['ceiling']:.4f}" for e in EPSILONS)
            + " at eps = 0.01, 0.1, 0.3, and the measured tails sit within "
            f"{widest(runs, 'tail', 'ceiling'):.3f} of their own "
            "ceilings. What 'compare mean return' compares is the price of exploring; the "
            "ranking is arithmetic on eps, available before the first episode",
        ),
        practice.Check(
            "FINDING: all three learn the same thing from the start state",
            widest(runs, 'start', None, star) < 1e-9,
            f"the greedy policy each run recovers is worth exactly {star:.4f} from (0,0) -- the "
            f"optimum, at every eps, to {widest(runs, 'start', None, star):.1e}. The quantity the "
            "exercise asks to compare is the one that does not separate the three runs, and the "
            "quantity that does separate them is not measured",
        ),
        practice.Check(
            "FINDING: the tradeoff lives off the path the metric walks",
            min(gaps[0] - 90, 0.5 - gaps[-1], 59.5 - low["tried"]) > 0,
            f"states the recovered policy is worth less than V* from: {len(low['wrong'])} at "
            f"eps=0.01, {len(runs[0.1]['wrong'])} at eps=0.1, {len(high['wrong'])} at eps=0.3, "
            f"with worst gaps {gaps[0]:.2f}, {gaps[1]:.2f}, {gaps[2]:.2f}. Of the "
            f"{len(low['wrong'])} at eps=0.01 only {len(low['misact'])} pick a suboptimal action "
            f"themselves; the rest inherit the loss downstream, which is why the two counts have "
            f"to be kept apart. At eps=0.01 the corner {hole[0]} is assigned '{hole[1]}', which "
            f"walks into the wall forever -- {hole[2]:.1f} against an optimal {hole[3]:.2f} -- "
            f"and only {low['tried']} of 60 state-action pairs are ever tried. Every episode "
            "starts at (0,0), so the stated metric sees none of this",
        ),
        practice.Check(
            "CONTROL: exploration is a threshold here, not a dial",
            max(result["control_start"] + 99, result["control"]["tried"] - 10) < 0,
            f"at eps=0 the same code, over 5,000 episodes, touches "
            f"{result['control']['tried']} of 60 state-action pairs and returns a greedy policy "
            f"worth {result['control_start']:.1f} from (0,0), mean return "
            f"{result['control']['mean']:.1f}. From eps=0 to 0.01 the outcome moves "
            f"{abs(result['control_start'] - low['start']):.0f} points; from 0.01 to 0.3 it moves "
            f"{abs(low['tail'] - high['tail']):.1f}, and all of that second move is cost rather "
            "than capability",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
