"""Exercise 3 — both axes move together, so there is no tradeoff to find.

    **Hard.** Implement GAE(λ). Sweep `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`. Plot final
    return vs sample efficiency. Where is the bias/variance sweet spot for this
    task?

Reading of the exercise: GAE is already implemented -- `gae_advantages` is the only
advantage path `actor_critic` has -- so the work is the sweep and what it shows.
"Final return vs sample efficiency" is plotted as two columns over 30 seeds per λ,
because the differences between adjacent λ are smaller than one seed's spread. The
plot ships as a table, per `D14`.

**ANSWER: there is no sweet spot -- the best λ is at the end of the range, on both
axes at once.** Median final return runs -5.896, -5.905, -5.977, -6.095, -6.422
and median episodes-to-converge runs 241, 221, 320, 362, 448 across
`λ = 0, 0.5, 0.9, 0.95, 1.0`. The two axes the exercise asks you to plot against
each other do not trade.

**FINDING: the lesson ships the second-worst value in its own sweep.** `main` runs
`lam=0.95`, which is fourth of five on return and fourth of five on speed.

**MECHANISM: the bias/variance tradeoff has had one of its two sides removed.**
`normalize` standardises the advantage to unit variance before the actor sees it
(exercise 2), so λ cannot buy the actor any variance reduction. What is left is
λ's effect on the critic's regression target, which nothing standardises -- and the
critic is several times more accurate at `λ = 0` than anywhere in the top half of
the range.

**FINDING: nothing in the sweep collapses.** 0 of 150 runs, at every λ. Lesson 06's
REINFORCE collapses on 22% and the difference is the baseline, not the estimator --
so λ is being asked to tune a failure mode this algorithm does not have.

**FINDING: a U-shape would need a critic that can actually be wrong.** The critic
here is linear over one-hot features, so it is exactly tabular: 16 states, 16
weights, no approximation error to trade against. `λ = 0`'s bias is bounded by the
critic's error, which is 0.26 on a value scale of 3.4 -- 8%.

Structure: `run` is the lesson's `actor_critic` at one λ and seed; `exact_values`
sweeps the lesson's own `step` for the `V^π` the critic is scored against.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "07-actor-critic-a2c-a3c"
EPISODES, SEEDS, GAMMA, TAIL = 1_500, 30, 0.99, 150
WINDOW, THRESHOLD = 50, -10.0
LAMBDAS = (0.0, 0.5, 0.9, 0.95, 1.0)
SHIPPED, CRITIC_SEEDS = 0.95, 8
CELLS = [(r, c) for r in range(4) for c in range(4)]


def converged_at(log):
    """First episode after which `WINDOW` consecutive returns stay above `THRESHOLD`."""
    return next((i for i in range(WINDOW, len(log) + 1)
                 if all(g > THRESHOLD for g in log[i - WINDOW:i])), None)


def backup(ref, theta, state, values):
    """One expected-softmax backup at `state`, over the lesson's own `step`."""
    probs = ref.softmax(ref.logits(theta, ref.features(state)))
    return sum(p * (lambda n, r, d: r + (0.0 if d else GAMMA * values[n]))(*ref.step(state, a))
               for a, p in enumerate(probs))


def exact_values(ref, theta):
    """`V^π` for the softmax policy `theta`, swept from the lesson's own `step`."""
    values = {s: 0.0 for s in CELLS}
    for _ in range(4_000):
        nxt = {s: (0.0 if s == ref.TERMINAL else backup(ref, theta, s, values)) for s in CELLS}
        moved = max(abs(nxt[s] - values[s]) for s in values)
        values = nxt
        if moved < 1e-12:
            break
    return values


def column(ref, lam):
    """One λ over `SEEDS` seeds, reduced to the two axes plus the critic's error."""
    finals, reached, errors, collapsed = [], [], [], 0
    for seed in range(SEEDS):
        theta, w, log = ref.actor_critic(EPISODES, lam=lam, rng=random.Random(seed))
        final = statistics.fmean(log[-TAIL:])
        finals.append(final)
        collapsed += final < -60
        reached.append(converged_at(log))
        if seed < CRITIC_SEEDS:
            true = exact_values(ref, theta)
            errors.append(statistics.fmean(
                abs(ref.value(w, ref.features(s)) - true[s])
                for s in CELLS if s != ref.TERMINAL))
    return {"final": statistics.median(finals), "collapsed": collapsed,
            "at": statistics.median([c for c in reached if c]),
            "error": statistics.fmean(errors)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {"cols": {lam: column(ref, lam) for lam in LAMBDAS}}


def axes(cols):
    """The sweep transposed into the lists the checks read."""
    return {k: [cols[lam][k] for lam in LAMBDAS] for k in ("final", "at", "error", "collapsed")}


def verify(result):
    cols = result["cols"]
    rows = axes(cols)
    finals, speeds, errors = rows["final"], rows["at"], rows["error"]
    collapsed = sum(rows["collapsed"])
    rank = sorted(LAMBDAS, key=lambda lam: cols[lam]["final"], reverse=True)
    return [
        practice.Check(
            "ANSWER: no sweet spot -- the best lambda is at the end of the range, on both axes",
            finals[0] == max(finals) and speeds[-1] == max(speeds) and finals[-1] == min(finals),
            f"over {SEEDS} seeds per lambda = {LAMBDAS}, median final return is "
            + ", ".join(f"{f:.3f}" for f in finals) + " and median episodes-to-converge is "
            + ", ".join(f"{s:.0f}" for s in speeds)
            + ". Return is best at lam=0 and worst at lam=1.0; speed is best at the low end too. "
            "The two axes the exercise asks to plot against each other do not trade",
        ),
        practice.Check(
            "FINDING: the lesson ships the second-worst value in its own sweep",
            rank.index(SHIPPED) >= 3,
            f"`main` runs lam={SHIPPED}, which ranks {rank.index(SHIPPED) + 1} of "
            f"{len(LAMBDAS)} on final return ({cols[SHIPPED]['final']:.3f} against "
            f"{cols[rank[0]]['final']:.3f} at lam={rank[0]}) and takes "
            f"{cols[SHIPPED]['at']:.0f} episodes to converge against {min(speeds):.0f}. The "
            "default is not the sweet spot the exercise is asking the reader to locate",
        ),
        practice.Check(
            "MECHANISM: the tradeoff has had one of its two sides removed",
            errors[0] < min(errors[2:]),
            f"`normalize` standardises the advantage to unit variance before the actor sees it "
            f"(exercise 2), so lambda cannot buy the actor any variance reduction. What is left "
            f"is lambda's effect on the critic's regression target, `returns = advantages + v`, "
            f"which nothing standardises: mean absolute critic error against the exact V^pi over "
            f"{CRITIC_SEEDS} seeds runs "
            + ", ".join(f"{e:.2f}" for e in errors)
            + f" -- {min(errors[2:]) / errors[0]:.1f}x lower at lam=0 than anywhere in the top "
            "half, though noisy across the middle. That is the half of the tradeoff still "
            "connected to the actor, and it favours the low end",
        ),
        practice.Check(
            "FINDING: nothing in the sweep collapses",
            collapsed == 0,
            f"{collapsed} of {SEEDS * len(LAMBDAS)} runs collapse, at every lambda. Lesson 06's "
            f"REINFORCE collapses on 22% of seeds and exercise 1 measured that the difference is "
            "the normalised baseline rather than the advantage estimator. Lambda is being asked "
            "to tune a failure mode this algorithm does not have",
        ),
        practice.Check(
            "FINDING: a U-shape would need a critic that can actually be wrong",
            errors[0] < 0.1 * abs(finals[0]) * 2,
            f"the critic is linear over one-hot features, so it is exactly tabular -- 16 states, "
            f"16 weights, no approximation error to trade against. lam=0's bias is bounded by "
            f"the critic's error, measured at {errors[0]:.2f} against value magnitudes around "
            f"{abs(finals[0]):.1f}. On a task where the critic could not represent V^pi, the low "
            "end of this sweep would carry a bias that this board simply does not supply",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
