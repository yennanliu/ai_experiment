"""Exercise 3 — the sentence never enters the computation.

    Take two different sentences of the same length, feed them through the same
    SelfAttention instance, and compare their attention patterns. What changes?
    What stays the same?

Reading of the exercise: `SelfAttention.forward(self, X)` takes an array and no
tokens, and `main()` builds `X` from `rng.normal` while using `sentence` only for
row and column labels. So "two different sentences" can only mean two different
`X`. They are built here from two seeds, the same way the lesson builds its one,
and then the question is asked a second way -- by reordering a single sentence,
which is the comparison that has an exact answer.

**WHAT CHANGES: the numbers, and nothing else.** Mean `|W1 - W2|` is 0.104 and
the largest single disagreement is 0.378, on a matrix whose entries average
0.167. The rows stay high-entropy in both -- 0.95 and 0.88 of `log 6` -- because
these projections are random rather than trained, so neither pattern means
anything; they differ the way two draws differ.

**WHAT STAYS THE SAME: the instance, the shape, and the row sums.** `forward`
touches no state, so `Wq`, `Wk`, `Wv` are bit-identical afterwards, both matrices
are `(6, 6)`, and every row sums to 1 within 2.2e-16.

**FINDING: the token strings are decorative.** Two different word lists with the
same `X` give bit-identical attention. Nothing that makes one sentence differ
from another reaches the computation, so the exercise's "two different sentences"
is two different random seeds wearing labels.

**FINDING: the real invariant is that word order does not exist.** Permute the
rows of `X` and the attention matrix is the *same matrix* with rows and columns
permuted, to **5.6e-17**, and every output row is the old one relabelled. Feed in
"the cat sat on the mat" and "mat the on sat cat the" and self-attention cannot
tell them apart -- which is the entire reason Lesson 04 exists.

**CONTROL: the diagonal is not privileged.** Over 2,000 random sentences the row
argmax lands on the diagonal 17.6% of the time against 16.7% by chance. `Wq` and
`Wk` are independent draws, so `q_i . k_i` has no special status; the fraction of
a point of excess is `q_i` and `k_i` sharing `x_i`, not a "token looks at itself"
mechanism.

Structure: `patterns` runs one X; `entropy_fraction` normalises by `log n`;
`diagonal_rate` is the 2,000-sentence control.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "02-self-attention-from-scratch"
D_MODEL, DK, DV, TOKENS = 16, 8, 8, 6
SEEDS, TRIALS = (42, 7), 2_000
ORDER = (3, 0, 5, 1, 4, 2)


def embed(np, seed):
    """main()'s own 'fake embeddings' line, at a chosen seed."""
    return np.random.default_rng(seed).normal(0, 1, (TOKENS, D_MODEL))


def entropy_fraction(np, weights):
    """Mean row entropy as a fraction of log n -- 1.0 is a uniform lookup."""
    return float(-(weights * np.log(weights)).sum(axis=-1).mean() / np.log(TOKENS))


def diagonal_rate(np, layer, trials):
    """How often a row's largest weight is its own position, over random sentences."""
    rng = np.random.default_rng(0)
    hits = sum(int(row.argmax() == i)
               for _ in range(trials)
               for i, row in enumerate(layer.forward(rng.normal(0, 1, (TOKENS, D_MODEL)))[1]))
    return hits / (trials * TOKENS)


def solve():
    import numpy as np
    ref = parity.load_reference(PHASE, LESSON, "self_attention")
    layer = ref.SelfAttention(D_MODEL, DK, DV, seed=42)
    before = layer.Wq.copy()
    first, second = (embed(np, seed) for seed in SEEDS)
    (out, weights), (_, other) = layer.forward(first), layer.forward(second)
    order = np.array(ORDER)
    shuffled_out, shuffled = layer.forward(first[order])
    return {
        "mean_gap": float(np.abs(weights - other).mean()),
        "max_gap": float(np.abs(weights - other).max()),
        "entropies": (entropy_fraction(np, weights), entropy_fraction(np, other)),
        "shapes": (weights.shape, other.shape),
        "row_sums": float(max(np.abs(w.sum(axis=-1) - 1).max() for w in (weights, other))),
        "stateless": np.array_equal(layer.Wq, before),
        "relabelled": np.array_equal(layer.forward(first)[1], weights),
        "permuted_w": float(np.abs(weights[np.ix_(order, order)] - shuffled).max()),
        "permuted_out": float(np.abs(out[order] - shuffled_out).max()),
        "diagonal": diagonal_rate(np, layer, TRIALS),
    }


def verify(result):
    low, high = sorted(result["entropies"])
    return [
        practice.Check(
            "WHAT CHANGES: the numbers, on a matrix that means nothing either way",
            result["mean_gap"] > 0.05 and low > 0.8,
            f"mean |W1 - W2| is {result['mean_gap']:.3f} and the largest single disagreement "
            f"{result['max_gap']:.3f}, on entries averaging {1 / TOKENS:.3f}. Both stay "
            f"high-entropy -- {low:.2f} and {high:.2f} of log {TOKENS} -- because these "
            "projections are random, not trained: the patterns differ the way two draws differ",
        ),
        practice.Check(
            "WHAT STAYS THE SAME: the instance, the shape, and the row sums",
            result["stateless"] and result["shapes"] == ((TOKENS, TOKENS),) * 2
            and result["row_sums"] < 1e-15,
            f"forward touches no state, so Wq is bit-identical afterwards; both matrices are "
            f"{result['shapes'][0]}; every row of both sums to 1 within "
            f"{result['row_sums']:.1e}. Those are properties of softmax and of the instance, and "
            "they would hold for any input at all",
        ),
        practice.Check(
            "FINDING: the token strings never enter the computation",
            result["relabelled"],
            "SelfAttention.forward(self, X) takes an array and no tokens, and main() uses "
            "`sentence` only for row and column labels. Re-running the same X returns a "
            "bit-identical matrix whatever you call the words, so 'two different sentences' is "
            "two different random seeds -- the exercise cannot be run the way it is worded",
        ),
        practice.Check(
            "FINDING: the real invariant is that word order does not exist here",
            result["permuted_w"] < 1e-15 and result["permuted_out"] < 1e-15,
            f"permuting the rows of X by {ORDER} returns the same matrix with rows and columns "
            f"permuted, to {result['permuted_w']:.1e}, and every output row relabelled, to "
            f"{result['permuted_out']:.1e}. 'the cat sat on the mat' and a shuffle of it are the "
            "same input to this module, which is the entire reason Lesson 04 exists",
        ),
        practice.Check(
            "CONTROL: the diagonal is not privileged, so 'the token looks at itself' is not here",
            abs(result["diagonal"] - 1 / TOKENS) < 0.03,
            f"over {TRIALS:,} random sentences the row argmax is the diagonal "
            f"{result['diagonal'] * 100:.1f}% of the time against {100 / TOKENS:.1f}% by chance. "
            "Wq and Wk are independent draws, so q_i . k_i has no special status; the fraction of "
            "a point of excess is q_i and k_i sharing x_i, not a self-reference mechanism",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
