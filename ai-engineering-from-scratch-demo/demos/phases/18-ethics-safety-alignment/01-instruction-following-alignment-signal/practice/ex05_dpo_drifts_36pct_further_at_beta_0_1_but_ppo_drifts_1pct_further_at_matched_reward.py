"""Exercise 5 — DPO drifts 36% further than PPO at beta = 0.1; at matched reward PPO drifts 1% further.

    Replace the PPO loss with DPO (Phase 10 · 08) on the same preference
    data. Compare final policy drift (KL to SFT) and final reward. Which method
    drifts further at matched reward?

Reading of the exercise: "the same preference data" is the 500 pairs
`stage2_reward_model` fits on seed 0. They are regenerated with the same
draws and proven identical by refitting the lesson's reward from them. DPO
trains the SFT-initialised logits on those pairs with
-log sigmoid(beta * [log pi/pi_SFT (w) - log pi/pi_SFT (l)]) to convergence.
"Reward" is the expected lesson RM reward E_pi[r], the quantity main.py's
"Final reward" samples; drift is KL(pi || pi_SFT). PPO is the shipped
beta = 0.1, 300-step run.

**ANSWER: at matched reward PPO drifts further, by 0.9-1.1%, which is
nothing.** Matching DPO runs at beta 0.4-2 against the point on the PPO
trajectory with the same expected reward:

| DPO beta | reward | DPO KL | PPO KL |
|---:|---:|---:|---:|
| 0.4 | 0.671 | 0.306 | 0.309 |
| 0.5 | 0.643 | 0.246 | 0.249 |
| 0.7 | 0.592 | 0.163 | 0.164 |
| 1.0 | 0.535 | 0.095 | 0.097 |
| 2.0 | 0.441 | 0.029 | 0.029 |

In a three-action bandit both methods trace the same reward-vs-drift curve.

**FINDING: at the same beta = 0.1, DPO drifts 36% further and earns more
reward.** DPO ends at reward 0.716 with KL 0.462 nats, and puts 100.0% on B.
PPO ends at 0.683 with KL 0.340 and 96.9% on B. DPO reaches the optimum of
the KL-regularised objective, which at beta = 0.1 is a point mass. The toy
PPO stops after 300 steps, so its drift is set by the step count, not by
beta.

**FINDING: DPO lands exactly on the KL-regularised optimum, of a slightly
different reward.** Its implicit reward beta * log(pi/pi_SFT) is
(-0.179, 0.693, -0.514) at every beta from 0.4 to 2, so each DPO policy is
pi_SFT * exp(r / beta) for one fixed r. That r is not the lesson's RM,
which one SGD pass at lr 0.05 leaves at (-0.146, 0.716, -0.569), so DPO is
optimal for its own reading of the pairs rather than for the RM it is
scored on.

Structure: `seeded_run()` replays stage 2's draws and refits them; `dpo()`
trains on them and returns the implicit reward; `ppo_trajectory()` reads PPO's
policy at every step through the reference's own `kl()` call (swapped back
after).
"""

from __future__ import annotations

import collections
import math
import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "01-instruction-following-alignment-signal"
BETA, DPO_BETAS, EPOCHS = 0.1, (0.4, 0.5, 0.7, 1.0, 2.0), 3000


def sig(x):
    return 1 / (1 + math.exp(-x))


def seeded_run(ref):
    """Seed 0 as main(): SFT, RM, stage 2's replayed pairs, the RNG PPO goes on with, and refit == RM."""
    rng, saved, u, pairs, r = random.Random(0), ref.random, ref.labeler_true_utility(), [], [0.0] * 3
    ref.random = rng
    try:
        sft, replay = ref.stage1_sft(), random.Random()
        replay.setstate(rng.getstate())
        rm = ref.stage2_reward_model()
    finally:
        ref.random = saved
    for _ in range(500):
        i, j = replay.sample(range(3), 2)
        w, l = (i, j) if replay.random() < sig(u[i] - u[j]) else (j, i)
        step = 0.05 * (1 - sig(r[w] - r[l]))
        r[w], r[l], _ = r[w] + step, r[l] - step, pairs.append((w, l))
    return sft, rm, pairs, rng, [x - sum(r) / 3 for x in r] == rm


def dpo(sft, pairs, beta):
    """Full-batch gradient descent on the DPO loss: (final probs, centred beta * log(pi / pi_SFT))."""
    s, th, lr = sft.logits, list(sft.logits), 1.0 / beta**2
    counts = sorted(collections.Counter(pairs).items())
    for _ in range(EPOCHS):
        g = [0.0] * 3
        for (w, l), n in counts:
            c = n * beta * (1 - sig(beta * (th[w] - th[l] - s[w] + s[l])))
            g[w], g[l] = g[w] + c, g[l] - c
        th = [t + lr * x / len(pairs) for t, x in zip(th, g)]
    implied = [beta * (t - x) for t, x in zip(th, s)]
    return type(sft)(logits=th).probs(), r3(x - sum(implied) / 3 for x in implied)


def ppo_trajectory(ref, sft, rm, rng):
    """PPO probabilities after every step, read through the reference's own kl() call."""
    seen, real = [], ref.kl
    ref.kl, saved, ref.random = (lambda p, q: seen.append(list(p)) or real(p, q)), ref.random, rng
    try:
        ref.stage3_ppo(sft, rm, beta=BETA)
    finally:
        ref.kl, ref.random = real, saved
    return seen


def kl_at_reward(pts, target):
    """PPO's KL where its (reward, KL) path first reaches `target`, interpolated between steps."""
    i = next(i for i, (r, _) in enumerate(pts) if r >= target)
    (r0, k0), (r1, k1) = pts[i - 1], pts[i]
    return k0 + (k1 - k0) * (target - r0) / (r1 - r0)


def r3(xs):
    return tuple(round(x, 3) for x in xs)


def measure(ref, p, rm, sft):
    return sum(a * b for a, b in zip(p, rm)), ref.kl(p, sft)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    sft, rm, pairs, rng, same = seeded_run(ref)
    sp, traj = sft.probs(), ppo_trajectory(ref, sft, rm, rng)
    runs = {b: dpo(sft, pairs, b) for b in (BETA, *DPO_BETAS)}
    path, point = [measure(ref, p, rm, sp) for p in traj], {b: measure(ref, p, rm, sp) for b, (p, _) in runs.items()}
    matched = {b: (*point[b], kl_at_reward(path, point[b][0])) for b in DPO_BETAS}
    excess = [m[2] / m[1] - 1 for m in matched.values()]
    return {
        "same_data": same, "rm": r3(rm), "matched": {b: r3(m) for b, m in matched.items()},
        "excess": r3((min(excess), max(excess))), "implied": {b: runs[b][1] for b in DPO_BETAS},
        "ppo": r3(path[-1]), "dpo": r3(point[BETA]), "beta_excess": round(point[BETA][1] / path[-1][1] - 1, 2),
        "b_share": (round(runs[BETA][0][1], 4), round(traj[-1][1], 3)),
    }


def verify(result):
    matched, ppo, dpo_row = result["matched"], result["ppo"], result["dpo"]
    return [
        practice.Check(
            "ANSWER: at matched reward PPO drifts further, by 0.9-1.1%",
            result["same_data"] and result["excess"] == (0.009, 0.011)
            and matched == {0.4: (0.671, 0.306, 0.309), 0.5: (0.643, 0.246, 0.249), 0.7: (0.592, 0.163, 0.164),
                            1.0: (0.535, 0.095, 0.097), 2.0: (0.441, 0.029, 0.029)},
            f"refit == RM: {result['same_data']}; (reward, DPO KL, PPO KL) by DPO beta {matched}; PPO excess {result['excess']}",
        ),
        practice.Check(
            "FINDING: at the same beta = 0.1, DPO drifts 36% further and earns more reward",
            dpo_row == (0.716, 0.462) and ppo == (0.683, 0.34) and result["beta_excess"] == 0.36
            and result["b_share"] == (1.0, 0.969),
            f"(reward, KL): DPO {dpo_row}, PPO {ppo}; B share (DPO, PPO) {result['b_share']}",
        ),
        practice.Check(
            "FINDING: DPO lands exactly on the KL-regularised optimum, of a slightly different reward",
            set(result["implied"].values()) == {(-0.179, 0.693, -0.514)}
            and result["rm"] == (-0.146, 0.716, -0.569),
            f"DPO implicit reward by beta {result['implied']}; lesson RM {result['rm']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
