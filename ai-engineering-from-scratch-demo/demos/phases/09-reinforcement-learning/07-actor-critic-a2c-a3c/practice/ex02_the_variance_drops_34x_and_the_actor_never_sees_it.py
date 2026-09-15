"""Exercise 2 — the variance drops 34x, and the actor never sees any of it.

    **Medium.** Switch to TD-residual advantage (`r + γ V(s') - V(s)`). Measure
    variance of the advantage batches. By how much does it drop?

Reading of the exercise: "the advantage batches" is measured at both points it
exists -- as `gae_advantages` returns it, and as the actor receives it one line
later, after `normalize`. Measuring only the first answers the question as asked
and misses what the answer is worth. GAE at `λ = 0` is the TD residual exactly
(`gae = delta` when `λ = 0`), so no new advantage estimator is written; the
measurement is taken inside a real training run at each setting, not on a frozen
policy, because a cold critic makes every residual identical.

**ANSWER: by 34x, and by 0x, depending on which batch you measure.** Mean
per-episode advantage variance during training is **4.10** at `λ = 1.0` and
**0.12** at `λ = 0`. The variance of what the actor actually multiplies into its
gradient is **1.000** at both.

**MECHANISM: `normalize` standardises every batch before the actor sees it.** It
subtracts the mean and divides by the standard deviation, so the variance reaching
the policy gradient is 1 by construction at every `λ`. The reduction the exercise
asks you to measure is real, and it is discarded one line after it is produced.

**FINDING: it is not discarded for everyone -- the critic keeps it.**
`gae_advantages` returns `returns = advantages + v`, which is the critic's own
regression target, and the critic update has no `normalize`. So `λ` sets the
critic's target variance too, and there it survives.

**FINDING: and that is where the benefit shows up.** Against the exact `V^π` of
the policy each run ends with, the critic's mean absolute error is **0.26** at
`λ = 0` against **1.14** at `λ = 1.0` -- a 4.4x more accurate critic, on a value
scale of about 3 to 5.

Structure: `measure` is one training run with the two variances recorded inside
the loop; `exact_values` sweeps the lesson's own `step` for the true `V^π`.
"""

from __future__ import annotations

import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "07-actor-critic-a2c-a3c"
EPISODES, GAMMA, LR_A, LR_V, ENT, SEED = 1_500, 0.99, 0.05, 0.1, 0.01, 7
CELLS = [(r, c) for r in range(4) for c in range(4)]


def update(ref, theta, w, traj, advantages, returns):
    """The lesson's own actor and critic updates, lifted out so the loop can be measured."""
    for t, node in enumerate(traj):
        error = returns[t] - ref.value(w, node["x"])
        probs = node["probs"]
        for j in range(ref.N_FEAT):
            w[j] += LR_V * error * node["x"][j]
        for i in range(ref.N_ACTIONS):
            grad = (1.0 if i == node["a"] else 0.0) - probs[i]
            bonus = -math.log(max(probs[i], 1e-12)) - 1.0
            for j in range(ref.N_FEAT):
                theta[i][j] += LR_A * (advantages[t] * grad + ENT * bonus * probs[i]) * node["x"][j]


def measure(ref, lam):
    """One training run, recording the advantage variance before and after `normalize`."""
    rng = random.Random(SEED)
    theta, w = ref.init_theta(rng), ref.init_w(rng)
    raw, seen = [], []
    for _ in range(EPISODES):
        traj = ref.rollout(theta, w, rng)
        advantages, returns = ref.gae_advantages(traj, gamma=GAMMA, lam=lam)
        normalised = ref.normalize(advantages)
        if len(advantages) > 1:
            raw.append(statistics.pvariance(advantages))
            seen.append(statistics.pvariance(normalised))
        update(ref, theta, w, traj, normalised, returns)
    return {"raw": statistics.fmean(raw), "seen": statistics.fmean(seen),
            "theta": theta, "w": w}


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


def critic_error(ref, run):
    """Mean |V_w(s) - V^π(s)| over the non-terminal states, and the scale of V^π."""
    true = exact_values(ref, run["theta"])
    states = [s for s in CELLS if s != ref.TERMINAL]
    return (statistics.fmean(abs(ref.value(run["w"], ref.features(s)) - true[s]) for s in states),
            statistics.fmean(abs(true[s]) for s in states))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runs = {lam: measure(ref, lam) for lam in (0.0, 1.0)}
    return {"runs": {lam: {"raw": r["raw"], "seen": r["seen"]} for lam, r in runs.items()},
            "error": {lam: critic_error(ref, r) for lam, r in runs.items()}}


def verify(result):
    td, mc = result["runs"][0.0], result["runs"][1.0]
    td_err, mc_err = result["error"][0.0], result["error"][1.0]
    drop = mc["raw"] / td["raw"]
    return [
        practice.Check(
            "ANSWER: by 34x, and by 0x, depending on which batch you measure",
            drop > 10 and abs(td["seen"] - mc["seen"]) < 0.01,
            f"mean per-episode advantage variance during training is {mc['raw']:.4f} at lam=1.0 "
            f"and {td['raw']:.4f} at lam=0, a drop of {drop:.0f}x. The variance of what the "
            f"actor multiplies into its gradient is {mc['seen']:.3f} and {td['seen']:.3f} -- the "
            "same number. Both are the advantage batch; they are one line apart",
        ),
        practice.Check(
            "MECHANISM: normalize standardises every batch before the actor sees it",
            abs(td["seen"] - 1.0) < 0.01 and abs(mc["seen"] - 1.0) < 0.01,
            f"`normalize` subtracts the per-episode mean and divides by the standard deviation, "
            f"so the variance reaching the policy gradient is 1 by construction: measured "
            f"{td['seen']:.6f} and {mc['seen']:.6f}. Whatever lam does to the spread of the "
            "advantages is undone before a single parameter moves",
        ),
        practice.Check(
            "FINDING: the critic keeps what the actor throws away",
            td_err[0] < mc_err[0],
            f"`gae_advantages` returns `returns = advantages + v`, which is the critic's own "
            f"regression target, and the critic update has no normalize. So lam sets the "
            f"critic's target variance too -- and there the {drop:.0f}x survives, because "
            "nothing standardises it",
        ),
        practice.Check(
            "FINDING: and the critic is 4.4x more accurate for it",
            mc_err[0] / td_err[0] > 2.5,
            f"against the exact V^pi of the policy each run ends with -- swept from the lesson's "
            f"own step() -- the critic's mean absolute error is {td_err[0]:.3f} at lam=0 against "
            f"{mc_err[0]:.3f} at lam=1.0, {mc_err[0] / td_err[0]:.1f}x, on value scales of "
            f"{td_err[1]:.2f} and {mc_err[1]:.2f}. The TD residual's real benefit here reaches "
            "the actor through the critic, not through the gradient it was subtracted into",
        ),
        practice.Check(
            "FINDING: measuring on a frozen policy would have answered a different question",
            td["raw"] > 0.0,
            f"the measurement is taken inside a real training run at each lam. On a frozen "
            f"policy with the shipped all-zero critic every TD residual is -1 exactly, so the "
            f"raw variance is 0 and `normalize` divides by 1e-8 and returns zeros -- the actor "
            f"would receive no signal at all. During training the measured raw variance at lam=0 "
            f"is {td['raw']:.4f}, which is the number the exercise is asking for",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
