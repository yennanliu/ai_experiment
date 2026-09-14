"""Exercise 1 — rank 2, and the r=1 row is Eckart-Young rather than a failure.

    **Easy.** In `code/main.py`, vary the LoRA rank `r` from 1 to 4. At what rank
    does the LoRA exactly match a rank-2 target delta?

Reading of the exercise: the lesson's own `main()` builds a **rank-1** delta
(`u[i] * v[j] * 0.5`), so a rank-2 target has to be constructed here. It is built
from two orthonormal pairs with chosen singular values **3.0** and **2.0**, which
means the answer is known in closed form before the sweep runs: a rank-`k`
approximation of it leaves exactly the tail of the squared singular values. The
lesson's own `train_lora` and `lora_forward` then do the fitting, unchanged.

**ANSWER: r=2, and exactly.** Residual mean squared error by rank:

| r | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| residual | 4.053 | **4.8e-31** | 4.2e-31 | 5.4e-31 |

Rank 2 matches to machine precision, and ranks 3 and 4 buy nothing further --
there is nothing left to fit.

**FINDING: the r=1 row is not a failure, it is the Eckart-Young optimum.** For a
delta with singular values 3 and 2, the best possible rank-1 approximation leaves
exactly `2.0^2 = 4.0` of squared error, and since the probe vectors are isotropic
that is also the mean squared error per input. The trainer reaches **4.053**,
within **1.3%** of a bound it cannot beat. The sweep is not finding where the
optimiser starts working; it is measuring a theorem.

**FINDING: whether the sweep is readable at all depends on the delta's size.**
The same `train_lora` at the same `lr=0.01`, given a delta with singular values
**8.96** and **4.99** instead, returns **nan** at r=1 -- the update diverges
rather than settling at that delta's own Eckart-Young value of 24.9. Dropping the
learning rate to 0.001 recovers 23.6. The exercise's question has a clean answer
only while the target stays small enough for a learning rate the lesson hard-codes.

**CONTROL: the delta really is rank 2.** It is assembled from orthonormal `u` and
`v` pairs, so its singular values are the two scalars chosen, and the two
constructed rank-1 pieces are orthogonal to **1e-15**.

Structure: `orthonormal` builds the frames; `rank_two` assembles the delta with
known singular values; `residuals` runs the lesson's own trainer across ranks.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "08-controlnet-lora-conditioning"
DIM, RANKS, FIRST, SECOND = 6, (1, 2, 3, 4), 3.0, 2.0
BIG_FIRST, BIG_SECOND, SAFE_RATE = 8.96, 4.99, 0.001


def orthonormal(rng, count, dim=DIM):
    """`count` orthonormal vectors, by Gram-Schmidt on random draws."""
    frame = []
    for _ in range(count):
        vec = [rng.gauss(0, 1) for _ in range(dim)]
        for prior in frame:
            overlap = sum(a * b for a, b in zip(vec, prior))
            vec = [a - overlap * b for a, b in zip(vec, prior)]
        norm = math.sqrt(sum(a * a for a in vec))
        frame.append([a / norm for a in vec])
    return frame


def rank_two(rng, first, second):
    """A delta whose singular values are exactly `first` and `second`."""
    left, right = orthonormal(rng, 2), orthonormal(rng, 2)
    return [[first * left[0][i] * right[0][j] + second * left[1][i] * right[1][j]
             for j in range(DIM)] for i in range(DIM)], left, right


def residuals(ref, random, frozen, target, rate):
    """{rank: residual mse} from the lesson's own train_lora."""
    return {r: ref.train_lora(frozen, target, r, random.Random(7), lr=rate) for r in RANKS}


def solve():
    import random
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = random.Random(4)
    delta, left, right = rank_two(rng, FIRST, SECOND)
    frozen = ref.randn_matrix(DIM, DIM, rng, scale=0.5)
    target = [[frozen[i][j] + delta[i][j] for j in range(DIM)] for i in range(DIM)]
    big, _, _ = rank_two(random.Random(4), BIG_FIRST, BIG_SECOND)
    big_target = [[frozen[i][j] + big[i][j] for j in range(DIM)] for i in range(DIM)]
    return {
        "by_rank": residuals(ref, random, frozen, target, 0.01),
        "bound": SECOND ** 2,
        "big": ref.train_lora(frozen, big_target, 1, random.Random(7), lr=0.01),
        "big_safe": ref.train_lora(frozen, big_target, 1, random.Random(7), lr=SAFE_RATE),
        "big_bound": BIG_SECOND ** 2,
        "orthogonal": max(abs(sum(a * b for a, b in zip(left[0], left[1]))),
                          abs(sum(a * b for a, b in zip(right[0], right[1])))),
    }


def verify(result):
    by_rank, bound = result["by_rank"], result["bound"]
    return [
        practice.Check(
            "ANSWER: rank 2, exactly, and more rank buys nothing",
            by_rank[2] < 1e-20 and by_rank[3] < 1e-20 and by_rank[1] > 1.0,
            "residual mse by rank -- "
            + ", ".join(f"r={r}: {by_rank[r]:.3e}" for r in RANKS)
            + ". Rank 2 matches the rank-2 delta to machine precision, and ranks 3 and 4 buy "
              "nothing further because there is nothing left to fit",
        ),
        practice.Check(
            "FINDING: the r=1 row is the Eckart-Young optimum, not a failure",
            abs(by_rank[1] - bound) / bound < 0.05,
            f"the delta is built with singular values {FIRST} and {SECOND}, so the best possible "
            f"rank-1 approximation leaves exactly {SECOND}^2 = {bound} of squared error, and with "
            f"isotropic probes that is the mean squared error per input. The trainer reaches "
            f"{by_rank[1]:.3f}, within {abs(by_rank[1] - bound) / bound:.1%} of a bound it cannot "
            "beat: the sweep measures a theorem rather than finding where the optimiser starts",
        ),
        practice.Check(
            "FINDING: whether the sweep is readable depends on the delta's size",
            result["big"] != result["big"] and result["big_safe"] < 2 * result["big_bound"],
            f"the same train_lora at the same lr=0.01, given singular values {BIG_FIRST} and "
            f"{BIG_SECOND} instead, returns {result['big']} at r=1 -- the update diverges rather "
            f"than settling at that delta's own bound of {result['big_bound']:.1f}. Dropping the "
            f"rate to {SAFE_RATE} recovers {result['big_safe']:.1f}. The question has a clean "
            "answer only while the target stays small enough for a rate the lesson hard-codes",
        ),
        practice.Check(
            "CONTROL: the delta really is rank 2, by construction",
            result["orthogonal"] < 1e-12,
            f"the delta is assembled from orthonormal u and v pairs, so its singular values are "
            f"the two scalars chosen rather than something to be measured afterwards. The two "
            f"frames are orthogonal to {result['orthogonal']:.0e}, which is what makes the "
            "Eckart-Young number above a prediction and not a postdiction",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
