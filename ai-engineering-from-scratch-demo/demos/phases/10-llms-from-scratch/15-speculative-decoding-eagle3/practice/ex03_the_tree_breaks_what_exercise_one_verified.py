"""Exercise 3 — the naive tree raises alpha per leaf and destroys the distribution Exercise 1 checked.

    Modify the code to simulate EAGLE-2 tree search: at each step, the draft
    proposes a tree of shape `[2, 2, 2]` (eight candidate paths). The verifier
    runs once, and the highest-probability accepted path wins. Compute `α` per
    leaf and total tokens per verifier call. Compare to linear-chain
    spec-decoding at equivalent compute.

Reading of the exercise: "the highest-probability accepted path wins" is
implemented literally, because that is the natural reading and it is exactly what
breaks. The result is then put through the same `chi_square` test Exercise 1
uses, since a speculative scheme that changes the output distribution is not a
faster decoder -- it is a different model.

**ANSWER: per-node acceptance rises from 0.933 to 0.996, and the output
distribution fails its own check by a factor of 600.**

    branches   per-node alpha   chi^2 vs q   tokens/call   drafted nodes
        1           0.933             3.3        3.62            3
        2           0.996          8,790.5       3.97           14
        4           1.000         38,669.6       4.00           84

Two branches per node is a genuine improvement in acceptance and a catastrophic
one in correctness. The chi-square is scored against the exact `q`, which is
what Exercise 1 establishes the check should have done all along.

**MECHANISM: "highest-probability accepted path wins" is a selection step, and
selection is a bias.** Leviathan's guarantee is that *one* draft, accepted with
`min(1, q/p)` and corrected from the residual otherwise, is distributed as `q`.
Taking `b` independent drafts and keeping the survivor with the largest `q/p`
re-weights the output by the very ratio that was supposed to cancel: tokens the
draft under-produces relative to `q` are over-represented at exactly the rate
that made them likely to win.

**FINDING: acceptance per node is the wrong thing to maximise.** At four
branches the chain survives all three levels almost surely, so tokens per
verifier call reaches **4.00** against the linear chain's **3.62** -- a 10%
gain for **84 drafted nodes** against 3, and the tokens it emits are not samples
from the verifier.

**FINDING: the correct fix is known and is not a tuning change.** SpecInfer and
EAGLE-2 verify a tree by walking one path with the residual renormalised over
the candidates already rejected at that node -- an accept rule per *node*, not a
selection over leaves. That keeps the chi-square at Exercise 1's level, and the
distinction is invisible in the metric the exercise asks for, which is tokens
per verifier call.

Structure: `best_of` is the exercise's rule at one node; `tree_step` walks the
`[2, 2, 2]` tree; `linear_step` is the lesson's own `spec_step` at equal depth.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "15-speculative-decoding-eagle3"
Q = [0.30, 0.22, 0.15, 0.10, 0.08, 0.07, 0.05, 0.03]
TRIALS, LEVELS, CRITICAL = 50_000, 3, 14.07
BRANCHES = (1, 2, 4)


def best_of(ref, draft, branches, rng):
    """The exercise's rule: propose `branches` candidates, keep the accepted one with the
    highest q/p, and fall back to the residual when none is accepted."""
    winner = None
    for _ in range(branches):
        token = ref.sample(draft, rng)
        ratio = Q[token] / draft[token] if draft[token] > 0 else float("inf")
        if rng.random() < min(1.0, ratio) and (winner is None or ratio > winner[1]):
            winner = (token, ratio)
    if winner is None:
        return ref.sample(ref.residual(Q, draft), rng), False
    return winner[0], True


def arm(ref, draft, branches, trials=TRIALS):
    """One tree width: the first emitted token's histogram, per-level alpha, and depth."""
    rng = random.Random(11)
    observed, accepted, depths = [0] * len(Q), 0, []
    expected = [trials * qi for qi in Q]      # exact q, per Exercise 1
    for _ in range(trials):
        token, ok = best_of(ref, draft, branches, rng)
        observed[token] += 1
        accepted += ok
        depth = 0
        for _ in range(LEVELS):
            if not best_of(ref, draft, branches, rng)[1]:
                break
            depth += 1
        depths.append(depth + 1)
    return {"chi": ref.chi_square(observed, expected), "alpha": accepted / trials,
            "tokens": statistics.fmean(depths), "nodes": sum(branches ** d
                                                             for d in range(1, LEVELS + 1))}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    draft = ref.perturb(Q, 0.02, random.Random(2))
    return {"arms": {b: arm(ref, draft, b) for b in BRANCHES}, "levels": LEVELS}


def column(arms, field, fmt):
    return ", ".join(f"b={b} {format(row[field], fmt)}" for b, row in arms.items())


def verify(result):
    arms = result["arms"]
    one, two, four = arms[1], arms[2], arms[4]
    return [
        practice.Check(
            "ANSWER: alpha rises 0.933 -> 0.996 and the distribution fails its own check by 600x",
            two["alpha"] > one["alpha"] and two["chi"] > 100 * CRITICAL > one["chi"],
            "per-node acceptance is " + column(arms, "alpha", ".3f")
            + " and the chi-square of the first emitted token against q is "
            + column(arms, "chi", ",.1f")
            + f" against a {CRITICAL} critical value. Two branches per level is a genuine "
            f"improvement in acceptance -- {two['alpha'] - one['alpha']:+.3f} -- and a "
            f"catastrophic one in correctness, {two['chi'] / CRITICAL:.0f}x the threshold "
            "Exercise 1 confirms the single-draft scheme stays under",
        ),
        practice.Check(
            "MECHANISM: 'the highest-probability accepted path wins' is a selection, and that biases",
            four["chi"] > two["chi"] > one["chi"],
            "Leviathan's guarantee is that one draft, accepted with min(1, q/p) and corrected "
            "from the residual otherwise, is distributed as q. Taking b independent drafts and "
            "keeping the survivor with the largest q/p re-weights the output by the very ratio "
            f"that was supposed to cancel, and the damage grows with b: "
            + column(arms, "chi", ",.0f")
            + ". Tokens the draft under-produces relative to q are over-represented at exactly "
            "the rate that made them likely to win",
        ),
        practice.Check(
            "FINDING: acceptance per node is the wrong thing to maximise",
            four["tokens"] > two["tokens"] > one["tokens"] and four["nodes"] > 20 * one["nodes"],
            f"at four branches the chain survives all {result['levels']} levels almost surely, so "
            f"tokens per verifier call reaches {four['tokens']:.2f} against the linear chain's "
            f"{one['tokens']:.2f} at the same depth. It costs {four['nodes']} drafted nodes "
            f"against {one['nodes']}, and the tokens it emits are not samples from the verifier "
            "-- the metric the exercise asks for cannot see the difference",
        ),
        practice.Check(
            "FINDING: the correct fix is known and is not a tuning change",
            one["chi"] < CRITICAL,
            f"the single-draft arm scores {one['chi']:.2f}, below {CRITICAL}, because it is the "
            "lesson's own accept rule. SpecInfer and EAGLE-2 verify a tree by walking one path "
            "with the residual renormalised over the candidates already rejected at that node -- "
            "an accept rule per node, not a selection over leaves. No setting of b fixes the "
            "selection; the selection is the bug",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
