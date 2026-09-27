"""Exercise 1 — sycophancy falls from 0.333 to 0.046, 0.031, 0.030 as beta goes 0.1, 0.01, 0.

    Run `code/main.py`. Reproduce the inverse-scaling pattern: sycophancy at
    beta=0, beta=0.1, and beta=0.01. Does RLHF with KL penalty prevent
    amplification? Does removing it amplify more?

Reading of the exercise: "sycophancy" is the toy's P(S), the policy's mass on
the sycophantic-agreement action; the base policy is uniform, so the
pre-RLHF level is 1/3. The three betas the exercise names are run with the
reference's own `train_rm` (seed 7, as `main()` does) and `ppo_train`
(default 300 steps), averaged over five PPO seeds; the shipped `main()`
printout is parsed as well, since it sweeps different betas.

**ANSWER: the toy shows the opposite of inverse scaling.** P(S) is 0.333
before RLHF and 0.046 / 0.031 / 0.030 after it at beta = 0.1 / 0.01 / 0,
with a seed spread under 0.001. The KL penalty does not prevent
amplification -- there is none to prevent; it *slows the fall* of
sycophancy. Removing it lowers sycophancy further, it does not amplify it.
The shipped run agrees: P(S) = 0.300 / 0.072 / 0.037 / 0.030 at beta =
1 / 0.2 / 0.05 / 0, every value below the uniform 0.333.

**FINDING: the low-beta numbers are step-limited, and "longer RLHF" also
lowers sycophancy.** The KL-regularized optimum pi ~ exp(r / beta) puts
0.0019 on S at beta = 0.1, where 300 PPO steps stop at 0.046; doubling to
600 steps gives 0.023. The lesson's "~15% -> ~40% -> ~55% after 2x more
steps" runs the other way in its own simulator. The reason is the reward
model: `train_rm` scores A = +0.675, S = +0.051, W = -0.726, so the correct
answer outranks agreement and any strong optimizer drives S out.

Structure: `rm()` replays main()'s seed-7 reward fit; `ppo()` runs the
reference's PPO under a swapped-in seeded `random` and restores it.
"""

from __future__ import annotations

import contextlib
import io
import math
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "04-sycophancy-rlhf-amplification"
SEEDS, BETAS = range(5), (0.1, 0.01, 0.0)
ROW = r"PPO beta=([\d.]+) \(alpha=0\)\s+P\(A\)=([\d.]+)\s+P\(S\)=([\d.]+)"


def seeded(ref, seed, fn, *args, **kw):
    saved, ref.random = ref.random, random.Random(seed)
    try:
        return fn(*args, **kw)
    finally:
        ref.random = saved


def rm(ref):
    return seeded(ref, 7, ref.train_rm)


def ppo(ref, reward, beta, steps=300):
    """P(S) per seed after the reference's PPO from the uniform base policy."""
    runs = [seeded(ref, s, ref.ppo_train, [0.0] * 3, reward, beta=beta, steps=steps)
            for s in SEEDS]
    return [ref.sycophancy(ref.softmax(lg)) for lg in runs]


def shipped(ref):
    """The Stage-2 rows of main()'s printout, run exactly as shipped (seed 7)."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        seeded(ref, 7, ref.main)
    return {float(b): round(float(s), 3) for b, _, s in re.findall(ROW, out.getvalue())}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    reward = rm(ref)
    runs = {b: ppo(ref, reward, b) for b in BETAS}
    w = [math.exp(reward[a] / 0.1) for a in ref.ACTIONS]
    doc = parity.doc_text(PHASE, LESSON, "en")
    return {
        "base": round(ref.sycophancy(ref.softmax([0.0] * 3)), 3),
        "ps": {b: round(sum(v) / len(v), 3) for b, v in runs.items()},
        "spread": max(max(v) - min(v) for v in runs.values()),
        "shipped": shipped(ref),
        "rm": {a: round(v, 3) for a, v in reward.items()},
        "optimum_01": round(w[1] / sum(w), 4),
        "steps600": round(sum(ppo(ref, reward, 0.1, steps=600)) / len(SEEDS), 3),
        "doc_claims": [c for c in ("~15%", "~40%", "2x more steps, same beta): ~55%") if c in doc],
    }


def verify(result):
    ps, base, sh = result["ps"], result["base"], result["shipped"]
    return [
        practice.Check(
            "ANSWER: the toy shows the opposite of inverse scaling",
            base == 0.333 and ps == {0.1: 0.046, 0.01: 0.031, 0.0: 0.03}
            and result["spread"] < 0.001,
            f"P(S) base {base}; after PPO by beta {ps}; seed spread {result['spread']:.5f}",
        ),
        practice.Check(
            "ANSWER: the shipped main() sweep falls with beta, every value below 1/3",
            sh == {1.0: 0.3, 0.2: 0.072, 0.05: 0.037, 0.0: 0.03}
            and list(sh.values()) == sorted(sh.values(), reverse=True) and max(sh.values()) < base,
            f"shipped P(S) by beta {sh}",
        ),
        practice.Check(
            "FINDING: the low-beta numbers are step-limited, and longer RLHF lowers sycophancy",
            result["optimum_01"] == 0.0019 and result["steps600"] == 0.023
            and result["rm"] == {"A": 0.675, "S": 0.051, "W": -0.726}
            and len(result["doc_claims"]) == 3,
            f"beta=0.1 optimum P(S) {result['optimum_01']} vs 300 steps {ps[0.1]} vs 600 "
            f"steps {result['steps600']}; RM {result['rm']}; lesson claims "
            f"{result['doc_claims']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
