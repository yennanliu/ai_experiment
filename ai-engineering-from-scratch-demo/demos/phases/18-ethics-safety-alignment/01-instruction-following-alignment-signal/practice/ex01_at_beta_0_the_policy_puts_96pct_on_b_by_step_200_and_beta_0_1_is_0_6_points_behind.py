"""Exercise 1 — at beta = 0 the policy puts 96.1% on B by step 200, and beta = 0.1 is 0.6 points behind.

    Run `code/main.py`. Set `beta = 0.0` and report the action distribution
    after 200 PPO steps. Explain the mode-seeking behaviour in one paragraph.

Reading of the exercise: the shipped script runs 300 steps and draws its
beta = 0 run from the RNG after the beta = 0.1 run; the exercise asks for 200
steps, so the pipeline is replayed from the script's own seed (0) with SFT, RM,
then `stage3_ppo(beta=0.0, steps=200)`, and the same draw at beta = 0.1 is the
comparison. "Mode-seeking" is checked against the labeler, not just the RM.

**ANSWER: (A, B, C) = (2.1%, 96.1%, 1.8%) after 200 steps, KL 0.319 nats
from SFT.** SFT started at (18.0%, 63.0%, 19.0%). The paragraph: with no KL
term the objective E_pi[r] is linear in the action probabilities, so its
maximum is a point mass on the RM's argmax, and every policy-gradient step
moves mass toward the action whose reward beats the current average. The
policy stops representing the labeler's *distribution* (61.0% B when they
demonstrate) and keeps only its *mode*; the approach to 100% is slow only
because the softmax gradient p(1 - p) vanishes as p -> 1.

**FINDING: what main.py labels "reward hacking" is the policy converging on
the labeler's favourite.** The RM ranks the actions in the same order as the
labeler's true utility, so collapsing onto B raises true utility from 0.573
(SFT) to 0.956; there is no proxy gap to exploit.

**FINDING: beta = 0.1 does not bind in this toy.** The same draw at
beta = 0.1 ends at 95.5% B -- 0.6 points behind -- and at 50 steps, where the
lesson says hacking "appears", the two runs are at 86.7% and 86.0% B. The
optimum the KL term pulls toward, pi_SFT * exp(r / beta), is itself 99.99% B
at beta = 0.1: the reward spread (1.29) is 12.9x beta. The shipped 300-step
runs print 96.9% vs 97.3% B.

Structure: `pipeline()` replays SFT -> RM -> PPO on a seeded `random.Random`
swapped into the reference (restored after); `shipped()` parses main()'s own
printout.
"""

from __future__ import annotations

import contextlib
import io
import math
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "01-instruction-following-alignment-signal"
SEED, STEPS = 0, 200
PROBS = r"RLHF probs\s+: \['([\d.]+)', '([\d.]+)', '([\d.]+)'\]"


def pipeline(ref, beta, steps, seed=SEED):
    """(sft, rm, final policy, KL trajectory) for one seeded run of the three stages."""
    saved, ref.random = ref.random, random.Random(seed)
    try:
        sft, rm = ref.stage1_sft(), ref.stage2_reward_model()
        pi, _, kl_traj = ref.stage3_ppo(sft, rm, beta=beta, steps=steps)
        return sft, rm, pi, kl_traj
    finally:
        ref.random = saved


def shipped(ref):
    """RLHF probs of main()'s Run 1 (beta 0.1) and Run 2 (beta 0), as printed."""
    saved, ref.random, log = ref.random, random.Random(SEED), io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            ref.main()
    finally:
        ref.random = saved
    return [tuple(map(float, m)) for m in re.findall(PROBS, log.getvalue())[:2]]


def utility(ref, probs):
    return round(sum(p * u for p, u in zip(probs, ref.labeler_true_utility())), 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    sft, rm, pi0, kl0 = pipeline(ref, 0.0, STEPS)
    _, _, pi1, _ = pipeline(ref, 0.1, STEPS)
    at50 = [round(pipeline(ref, b, 50)[2].probs()[1], 3) for b in (0.0, 0.1)]
    target = [s * math.exp(r / 0.1) for s, r in zip(sft.probs(), rm)]
    order = [sorted(range(3), key=v.__getitem__) for v in (rm, ref.labeler_true_utility())]
    return {
        "beta0": [round(p, 3) for p in pi0.probs()], "kl0": round(kl0[-1], 3),
        "beta01_b": round(pi1.probs()[1], 3), "sft": [round(p, 3) for p in sft.probs()],
        "labeler_b": round(ref.softmax(ref.labeler_true_utility())[1], 3),
        "utility": (utility(ref, sft.probs()), utility(ref, pi0.probs())),
        "same_order": order[0] == order[1], "at50": at50,
        "target_b": round(target[1] / sum(target), 4), "spread": round(max(rm) - min(rm), 2),
        "shipped_b": [p[1] for p in shipped(ref)],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: (A, B, C) = (2.1%, 96.1%, 1.8%) after 200 steps, KL 0.319 nats",
            result["beta0"] == [0.021, 0.961, 0.018] and result["kl0"] == 0.319
            and result["sft"] == [0.18, 0.63, 0.19] and result["labeler_b"] == 0.61,
            f"beta = 0, 200 steps: {result['beta0']}, KL {result['kl0']}; SFT {result['sft']}; "
            f"labeler demonstrates B at {result['labeler_b']}",
        ),
        practice.Check(
            "FINDING: main.py's 'reward hacking' is convergence on the labeler's favourite",
            result["same_order"] and result["utility"] == (0.573, 0.956),
            f"RM and labeler rank actions identically: {result['same_order']}; true utility "
            f"SFT -> beta 0 policy: {result['utility']}",
        ),
        practice.Check(
            "FINDING: beta = 0.1 does not bind in this toy",
            [result[k] for k in ("beta01_b", "at50", "target_b", "spread", "shipped_b")]
            == [0.955, [0.867, 0.86], 0.9999, 1.29, [0.969, 0.973]],
            f"B at step 200: beta 0 {result['beta0'][1]} vs 0.1 {result['beta01_b']}; at step 50 "
            f"{result['at50']}; KL optimum at 0.1 puts {result['target_b']} on B; reward spread "
            f"{result['spread']}; shipped 300-step runs {result['shipped_b']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
