"""Exercise 2 — alpha 0.5 costs nothing in the toy, and 1.2 points of correct confirmations once they exist.

    Set alpha = 0.5 in the agreement-penalty correction. What is the cost to
    correct-answer rate? What is the benefit to sycophancy reduction? Compute
    the Pareto frontier.

Reading of the exercise: the reference's `agreement_penalty_correction` is
applied to the seed-7 reward model and trained with its `ppo_train` at the
shipped beta = 0.1 (two PPO seeds averaged). "Correct-answer rate" is P(A),
"sycophancy" is P(S). The frontier is computed over alpha = 0.0 .. 1.0 in
steps of 0.1, first in the reference's world as shipped, then with the one
case the toy leaves out: prompts where the user is *right*, so the correct
answer agrees with them (half the prompts each).

**ANSWER: in the reference, alpha = 0.5 has no cost -- it raises the
correct-answer rate.** P(A) goes 0.936 -> 0.961 (+0.026 unrounded) while P(S) goes
0.046 -> 0.022 (-0.024). Across the sweep P(A) rises and P(S) falls with
every step of alpha, so the Pareto frontier is one point, alpha = 1.0: every
smaller alpha is dominated. The cause is in the data: `AGREEMENT` marks only
the sycophantic action, so no correct answer ever agrees, and the penalty
has nothing helpful to hit. The lesson's "loss of legitimate agreement" and
main()'s "erodes agreement-when-correct" cannot occur in this simulator.

**FINDING: add user-right prompts and the trade-off appears, from alpha =
0.4.** There the correct answer agrees (utility +1, agreement 1), and the
lesson's unconditional `r - alpha * agree` penalizes it. At alpha = 0.5 the
rate of correct confirmations falls 0.977 -> 0.965 (-0.012) and needless corrections
rise 0.012 -> 0.018 -- the lesson's "slightly more contrarian". The mixed
correct-answer rate peaks at 0.963 at alpha = 0.4, so alpha < 0.4 is
dominated and the frontier is alpha = 0.4 .. 1.0, trading the correct-answer
rate 0.963 -> 0.952 for sycophancy 0.024 -> 0.014.

Structure: `world()` swaps the reference's `TRUE_UTILITY` / `AGREEMENT`
(which its `labeler_reward` and penalty read) and a seeded `random`, then
restores them; `frontier()` keeps the non-dominated alphas.
"""

from __future__ import annotations

import inspect
import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "04-sycophancy-rlhf-amplification"
ALPHAS, BETA, SEEDS = [round(0.1 * i, 1) for i in range(11)], 0.1, (0, 1)
USER_RIGHT = ({"A": 1.0, "S": -0.3, "W": -0.5}, {"A": 1.0, "S": 0.0, "W": 0.0})


def world(ref, fn, util=None, agree=None, seed=7):
    saved = ref.TRUE_UTILITY, ref.AGREEMENT, ref.random
    ref.TRUE_UTILITY, ref.AGREEMENT = util or saved[0], agree or saved[1]
    ref.random = random.Random(seed)
    try:
        return fn()
    finally:
        ref.TRUE_UTILITY, ref.AGREEMENT, ref.random = saved


def policy(ref, rm, alpha, case=(None, None)):
    """Mean (P(A), P(S)) after PPO on the alpha-corrected reward in one world."""
    def run(seed):
        reward = world(ref, lambda: ref.agreement_penalty_correction(rm, alpha), *case)
        return ref.softmax(world(ref, lambda: ref.ppo_train([0.0] * 3, reward, BETA),
                                 seed=seed))
    probs = [run(s) for s in SEEDS]
    return tuple(sum(p[i] for p in probs) / len(probs) for i in (0, 1))


def frontier(points):
    """Alphas whose (correct, sycophancy) no other alpha beats on both."""
    def beaten(c, s):
        return any(c2 >= c and s2 <= s and (c2, s2) != (c, s) for c2, s2 in points.values())
    return [a for a, (c, s) in points.items() if not beaten(c, s)]


def rounded(sweep):
    return {a: tuple(round(v, 3) for v in p) for a, p in sweep.items()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rm, rm_right = world(ref, ref.train_rm), world(ref, ref.train_rm, *USER_RIGHT)
    doc = parity.doc_text(PHASE, LESSON, "en")
    wrong = {a: policy(ref, rm, a) for a in ALPHAS}
    right = {a: policy(ref, rm_right, a, USER_RIGHT) for a in ALPHAS}
    mixed = {a: ((wrong[a][0] + right[a][0]) / 2, wrong[a][1]) for a in ALPHAS}
    deltas = (wrong[0.5][0] - wrong[0.0][0], wrong[0.5][1] - wrong[0.0][1],
              right[0.5][0] - right[0.0][0])
    return {
        "agree_on": [a for a in ref.ACTIONS if ref.AGREEMENT[a]],
        "wrong": rounded(wrong), "right": rounded(right), "mixed": rounded(mixed),
        "quoted": ["erodes agreement-when-correct" in inspect.getsource(ref.main),
                   "loss of legitimate agreement" in doc, "more contrarian" in doc],
        "front_ref": frontier(wrong), "front_mixed": frontier(mixed),
        "peak": max(ALPHAS, key=lambda a: mixed[a][0]),
        "deltas": tuple(round(d, 3) for d in deltas),
    }


def verify(result):
    w, r, m, dl = result["wrong"], result["right"], result["mixed"], result["deltas"]
    pa, ps = [w[a][0] for a in ALPHAS], [w[a][1] for a in ALPHAS]
    return [
        practice.Check(
            "ANSWER: in the reference, alpha = 0.5 has no cost -- it raises correct answers",
            all([(w[0.0], w[0.5], dl[:2]) == ((0.936, 0.046), (0.961, 0.022), (0.026, -0.024)),
                 pa == sorted(pa), ps == sorted(ps, reverse=True), result["front_ref"] == [1.0],
                 result["agree_on"] == ["S"], all(result["quoted"])]),
            f"(P(A), P(S)) at alpha 0 {w[0.0]}, 0.5 {w[0.5]}, 1.0 {w[1.0]}; frontier "
            f"{result['front_ref']}; deltas at 0.5 {dl[:2]}; "
            f"AGREEMENT is 1 only on {result['agree_on']}",
        ),
        practice.Check(
            "FINDING: add user-right prompts and the trade-off appears, from alpha = 0.4",
            (r[0.0], r[0.5], dl[2], result["peak"], m[0.0], m[0.4], m[1.0])
            == ((0.977, 0.012), (0.965, 0.018), -0.012, 0.4, (0.956, 0.046), (0.963, 0.024),
                (0.952, 0.014)) and result["front_mixed"] == ALPHAS[4:],
            f"user-right (confirm, needless correction) at alpha 0 {r[0.0]}, 0.5 {r[0.5]} "
            f"(confirm {dl[2]:+}); peak alpha {result['peak']}; "
            f"mixed (correct, sycophancy) {m}; frontier {result['front_mixed']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
