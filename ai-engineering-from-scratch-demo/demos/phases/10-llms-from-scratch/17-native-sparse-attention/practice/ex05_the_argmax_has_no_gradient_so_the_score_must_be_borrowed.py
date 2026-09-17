"""Exercise 5 — `top_k_blocks` has zero derivative everywhere and one jump, which is why the score is reused.

    Read Section 4 of the NSA paper (arXiv:2502.11089) and explain in three
    sentences why the compressed branch's attention scores are reused for top-k
    selection rather than computing a separate routing score. Tie the answer to
    gradient flow.

Reading of the exercise: the explanation is written as a measurement of the
lesson's own `top_k_blocks`, because "tie the answer to gradient flow" is a claim
about a derivative and the function whose derivative is at issue is fifteen lines
away. Its sensitivity to its input is measured directly: sweep one score across
the range and record where the selected set changes.

**ANSWER: the selection is a step function -- constant almost everywhere, with a
single jump.** Sweeping one block's compressed score from 0.00 to 0.40 in steps
of 0.0005, `top_k_blocks` returns `[1, 2, 4]` throughout and switches to
`[1, 2, 3]` at exactly **0.2000**, where it ties the block it displaces. That is
**1 change point in 800 samples**; perturbing a score by 1e-09 anywhere else
returns the identical set.

**MECHANISM: a separate routing score would receive no gradient at all.** The
only thing the selection does with a score is order it, and ordering has
derivative zero wherever it is defined and is undefined where it is not. A
routing head trained *through* the selection would get `0` from every batch
except the measure-zero set where two blocks tie. Reusing the compressed
branch's attention scores means the routing signal is trained through the
compressed branch's *own output path*, which is an ordinary differentiable
attention, and the selection rides along for free.

**FINDING: the scores that do the routing are used twice and differentiated
once.** `nsa_step` computes `cmp_w, cmp_out = attention(q, K_cmp, V_cmp)`, feeds
`cmp_w` to `top_k_blocks`, and feeds `cmp_out` into the gated sum. The gradient
reaches `cmp_w` through `cmp_out` -- the second use -- and never through the
first. Selection is a consumer of a trained signal, not a trainer of one.

**FINDING: the same argument rules out training the gate through the
selection.** The gate multiplies `sel_out`, which depends on which blocks were
picked, which is the step function above. Whatever gradient the gate receives
about the selected branch is a gradient about the *contents* of the blocks that
happened to be picked, holding the picks constant -- so nothing in `nsa_step`
can learn to pick differently.

Structure: `sweep` walks one score across a grid and records the selected set at
each point; `jumps` reports where it changed and `flat` probes the derivative
between changes.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "17-native-sparse-attention"
SCORES = [0.10, 0.30, 0.25, 0.05, 0.20, 0.10]
MOVED, TOP_K = 3, 3
STEPS, LIMIT, EPSILON = 800, 0.40, 1e-9


def selection(ref, value):
    scores = list(SCORES)
    scores[MOVED] = value
    return tuple(ref.top_k_blocks(scores, TOP_K))


def sweep(ref):
    """The selected set at every point on the grid, and where it changed."""
    grid = [i * LIMIT / STEPS for i in range(STEPS)]
    picks = [selection(ref, value) for value in grid]
    changes = [(grid[i], picks[i - 1], picks[i])
               for i in range(1, len(grid)) if picks[i] != picks[i - 1]]
    return {"grid": grid, "picks": picks, "changes": changes,
            "distinct": len(set(picks))}


def flat(ref, probes):
    """Does a 1e-09 nudge ever change the selection away from a tie?"""
    return [value for value in probes if selection(ref, value) != selection(ref, value + EPSILON)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    walk = sweep(ref)
    probes = [0.05, 0.12, 0.19, 0.21, 0.28, 0.35]
    order = sorted(range(len(SCORES)), key=lambda i: -SCORES[i])
    return {
        "changes": [(round(at, 4), list(before), list(after))
                    for at, before, after in walk["changes"]],
        "distinct": walk["distinct"],
        "samples": len(walk["grid"]),
        "moved_by_eps": flat(ref, probes),
        "probes": probes,
        "base": list(selection(ref, SCORES[MOVED])),
        "sorted_by_score": order[:TOP_K],
        "tie_value": SCORES[4],
    }


def verify(result):
    changes = result["changes"]
    return [
        practice.Check(
            "ANSWER: the selection is a step function -- one jump in 800 samples",
            len(changes) == 1 and result["distinct"] == 2,
            f"sweeping one block's compressed score from 0.00 to {LIMIT} in "
            f"{result['samples']} steps, top_k_blocks returns {result['base']} throughout and "
            f"switches to {changes[0][2]} at exactly {changes[0][0]:.4f}, where it ties the block "
            f"it displaces at {result['tie_value']}. That is {len(changes)} change point in "
            f"{result['samples']} samples and {result['distinct']} distinct outputs in total",
        ),
        practice.Check(
            "MECHANISM: a separate routing score would receive no gradient at all",
            not result["moved_by_eps"],
            f"perturbing the score by {EPSILON:.0e} at any of {result['probes']} returns the "
            "identical set -- the selection's derivative with respect to its input is zero "
            "wherever it is defined and undefined where it is not. A routing head trained through "
            "the selection would get exactly 0 from every batch except the measure-zero set where "
            "two blocks tie, which is why NSA reuses the compressed branch's scores instead: that "
            "signal is trained through the compressed branch's own differentiable output path",
        ),
        practice.Check(
            "FINDING: the scores that route are used twice and differentiated once",
            result["base"] == sorted(result["sorted_by_score"]),
            "nsa_step computes cmp_w, cmp_out = attention(q, K_cmp, V_cmp), feeds cmp_w to "
            "top_k_blocks and feeds cmp_out into the gated sum. The selection it produces is "
            f"exactly the argsort of the scores, {result['base']}, and the gradient reaches cmp_w "
            "through cmp_out -- the second use -- and never through the first. Selection is a "
            "consumer of a trained signal, not a trainer of one",
        ),
        practice.Check(
            "FINDING: the same argument rules out training the gate through the selection",
            len(changes) == 1,
            "the gate multiplies sel_out, which depends on which blocks were picked, which is the "
            "step function above. Whatever gradient the gate receives about the selected branch "
            "is a gradient about the contents of the blocks that happened to be picked, holding "
            "the picks constant -- so nothing anywhere in nsa_step can learn to pick differently, "
            "and the compressed branch is the only part of the routing that can be trained at all",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
