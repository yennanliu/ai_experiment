"""Exercise 3 — two biases of the same size cancel at the state the exercise names.

    **Hard.** Implement Double Q-learning. On a noisy-reward GridWorld (Gaussian
    noise σ=5 added to per-step reward), show Q-learning overestimates `V*(0,0)`
    by a meaningful amount while Double Q-learning does not.

Reading of the exercise: "show" is read as *measure over enough runs to support a
claim* -- 20 paired seeds, since one run of this cannot resolve a bias of any
size. Q-learning is the lesson's own, run against a `step` wrapped to add the
noise; Double Q-learning is written here, as the exercise asks. `V*(0,0)` is
value iteration over the clean `step`, since the noise is zero-mean and does not
move the true values.

**ANSWER: the claim does not reproduce, and its second half is backwards.**
Q-learning's bias at (0,0) is -0.16 ± 1.13, 0.6 standard errors from zero. Double
Q-learning's is -2.04, 3.7 standard errors from zero and an *under*estimate. The
measurable bias belongs to the estimator the exercise says has none.

**MECHANISM: the maximization bias is real and it is being cancelled.** The four
`Q(0,0,·)` estimates carry a spread of 1.41 at σ=5, and `E[max_a (Q* + noise)]` at
that spread is -4.758 against `max_a Q* = -5.852` -- **+1.09** from the max alone,
with no learning in it. But every entry it maximizes over is depressed by -1.56 to
-2.24, stably, so `max_a Q(0,0,a)` shows neither number.

**FINDING: the depression is the noise alone.** The same code on the same 2,000
episodes at σ=0 gives per-action errors of 0.00 -- so the budget is sufficient and
the implementation is right; only the noise separates that from -1.94.

**FINDING: one run cannot show any of it.** Per-run spread of `max_a Q(0,0,a)` is
1.13 -- wider than Q-learning's bias, wider than the maximization bias itself, and
comparable to Double Q's. "By a meaningful amount" is asked of a measurement whose
own error bar is larger than every effect in the experiment.

**FINDING: Double Q-learning's underestimate is the price it was designed to
charge.** Each table sees half the updates, and `QB[argmax QA]` is deliberately
not a maximum. At this budget that costs -2.04 against a maximization bias of
+1.09 it exists to remove: an over-correction of 1.9x, which is a trade rather
than the fix the exercise frames it as.

Structure: `noisy` wraps the lesson's `step`; `double_q` is the new algorithm;
`max_bias` isolates the max operator from all learning dynamics.
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "04-q-learning-sarsa"
GAMMA, ALPHA, EPSILON, SIGMA, CAP = 0.99, 0.1, 0.1, 5.0, 200
EPISODES, SEEDS, CELLS = 2_000, 20, [(r, c) for r in range(4) for c in range(4)]


def noisy(clean, sigma, seed):
    """The lesson's own `step`, with Gaussian noise on every non-zero reward."""
    rng = random.Random(seed)
    return lambda s, a: (lambda n, r, d: (n, r + rng.gauss(0.0, sigma) if r else r, d))(
        *clean(s, a))


def look(ref, state, action, values):
    """One clean step of the lesson's own `step`, backed up."""
    nxt, reward, done = ref.step(state, action)
    return reward + (0.0 if done else GAMMA * values[nxt])


def optimal(ref):
    """`V*` and `Q*(0,0,·)` from the clean `step`; the noise is zero-mean, so these are true.

    Forty sweeps, no convergence test: the board is deterministic with an absorbing
    terminal, so value iteration is exact after its six-step diameter.
    """
    values = {s: 0.0 for s in CELLS}
    for _ in range(40):
        values = {s: (0.0 if s == ref.TERMINAL else
                      max(look(ref, s, a, values) for a in ref.ACTIONS)) for s in CELLS}
    return values, {a: look(ref, (0, 0), a, values) for a in ref.ACTIONS}


def double_q(ref, step, rng, episodes=EPISODES):
    """Double Q-learning: one table picks the action, the other prices it."""
    tables = [defaultdict(lambda: dict.fromkeys(ref.ACTIONS, 0.0)) for _ in range(2)]
    for _ in range(episodes):
        state = (0, 0)
        for _ in range(CAP):
            if rng.random() < EPSILON:
                action = rng.choice(ref.ACTIONS)
            else:
                action = max(ref.ACTIONS, key=lambda a: tables[0][state][a] + tables[1][state][a])
            nxt, reward, done = step(state, action)
            pick = rng.randrange(2)
            chooser, pricer = tables[pick], tables[1 - pick]
            best = max(ref.ACTIONS, key=lambda a: chooser[nxt][a])
            target = reward if done else reward + GAMMA * pricer[nxt][best]
            chooser[state][action] += ALPHA * (target - chooser[state][action])
            if done:
                break
            state = nxt
    return {a: (tables[0][(0, 0)][a] + tables[1][(0, 0)][a]) / 2 for a in ref.ACTIONS}


def one_seed(seed, sigma, episodes=EPISODES):
    """One paired run: the lesson's Q-learning and Double Q on the same noisy board."""
    ref = parity.load_reference(PHASE, LESSON, "main")
    clean = ref.step
    ref.step = noisy(clean, sigma, 7 * seed + 1)
    table, _log = ref.q_learning(episodes, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON,
                                 rng=random.Random(1000 + seed))
    ref.step = noisy(clean, sigma, 7 * seed + 1)      # the same noise stream, replayed
    return dict(table[(0, 0)]), double_q(ref, ref.step, random.Random(1000 + seed), episodes)


def max_bias(star, spread, draws=120_000):
    """`E[max_a (Q* + noise)] - max_a Q*`: the max operator, with no learning in it."""
    rng = random.Random(0)
    return statistics.fmean(max(rng.gauss(v, spread) for v in star.values())
                            for _ in range(draws)) - max(star.values())


def bias_of(rows, truth):
    """(mean, spread, standard error) of `max_a Q(0,0,a) - V*` over the seeds."""
    xs = [max(row.values()) - truth for row in rows]
    sd = statistics.pstdev(xs)
    return statistics.fmean(xs), sd, sd / len(xs) ** 0.5


def per_action(rows, star):
    """Mean error of each `Q(0,0,a)` against `Q*(0,0,a)`, over the seeds."""
    return {a: statistics.fmean(row[a] for row in rows) - star[a] for a in star}


def solve():
    values, star = optimal(parity.load_reference(PHASE, LESSON, "main"))
    singles, doubles = (list(x) for x in zip(*[one_seed(s, SIGMA) for s in range(SEEDS)]))
    spread = statistics.fmean(statistics.pstdev(row[a] for row in singles) for a in star)
    return {"single": bias_of(singles, values[(0, 0)]), "star": star, "spread": spread,
            "double": bias_of(doubles, values[(0, 0)]), "entries": per_action(singles, star),
            "max_bias": max_bias(star, spread),
            "clean": per_action([one_seed(s, 0.0)[0] for s in range(SEEDS)], star)}


def verify(result):
    star, single, double = result["star"], result["single"], result["double"]
    entries, bias, top = result["entries"], result["max_bias"], max(result["star"].values())
    depressed = statistics.fmean(entries.values())
    show = lambda table: ", ".join(f"{a}={table[a]:+.2f}" for a in star)
    return [
        practice.Check(
            "ANSWER: it does not reproduce, and the second half is backwards",
            abs(single[0]) < 2 * single[2] and double[0] < -3 * double[2],
            f"over {SEEDS} paired seeds at sigma={SIGMA:g}, Q-learning's bias in max_a Q(0,0,a) is "
            f"{single[0]:+.3f} +/- {single[1]:.2f}, {abs(single[0]) / single[2]:.1f} standard "
            f"errors out; Double Q-learning's is {double[0]:+.3f}, "
            f"{abs(double[0]) / double[2]:.1f} out and an underestimate. The measurable bias is "
            f"the one the exercise says does not exist, and the per-run spread {single[1]:.2f} "
            "beats every effect here, so one run shows nothing",
        ),
        practice.Check(
            "MECHANISM: the maximization bias is real, and it is being cancelled",
            bias > 1.0 and depressed < -1.0,
            f"the four Q(0,0,.) estimates carry a spread of {result['spread']:.2f}, and "
            f"E[max_a (Q* + noise)] at that spread is {bias + top:.3f} against max_a Q* = "
            f"{top:.3f} -- {bias:+.2f} from the max alone, no learning in it. But every entry it "
            f"maximises over is depressed ({show(entries)}, mean {depressed:+.2f}), and the two "
            "nearly cancel",
        ),
        practice.Check(
            "FINDING: the depression is the noise, not the code and not the budget",
            max(abs(v) for v in result["clean"].values()) < 0.05,
            f"the same code on the same {EPISODES:,} episodes with sigma=0 gives per-action "
            f"errors of {show(result['clean'])} -- exact. The budget is therefore sufficient and "
            f"the implementation correct; only the noise separates that from the "
            f"{statistics.fmean(entries.values()):+.2f} mean error at sigma={SIGMA:g}",
        ),
        practice.Check(
            "FINDING: Double Q's underestimate is the price it was designed to charge",
            double[0] < -bias,
            f"each table sees half the updates and QB[argmax QA] is deliberately not a maximum, "
            f"so the estimator is pessimistic by construction. That costs {abs(double[0]):.2f} "
            f"against the {bias:.2f} it exists to remove, over-correcting "
            f"{abs(double[0]) / bias:.1f}x -- a trade the exercise frames as a fix",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
