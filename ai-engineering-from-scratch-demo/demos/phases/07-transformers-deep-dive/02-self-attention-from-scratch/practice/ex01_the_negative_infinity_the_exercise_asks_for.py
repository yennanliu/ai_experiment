"""Exercise 1 — the negative infinity the exercise asks for is the unsafe fill.

    Modify `scaled_dot_product_attention` to accept an optional mask matrix
    that sets certain positions to negative infinity before softmax (this is
    how causal/decoder masking works)

Reading of the exercise: "optional" is read strictly -- with no mask the new
function must be bit-identical to the lesson's, or it is a different function
wearing the same name -- and "negative infinity" is read literally, then tested
against the `-1e9` every real implementation uses instead. Both fills run
through the lesson's own `softmax`, because the trap is in that function and not
in the mask.

**ANSWER: it works, and it is a strict extension.** With `mask=None` the result
is bit-identical to `scaled_dot_product_attention` on the lesson's own X. With a
causal mask, row 0 becomes exactly `[1, 0, 0, 0, 0, 0]` and the output for the
first token is bit-identical to `v_0` -- a decoder's first position has nothing
but itself, which is the whole reason generation has to start from a prompt.

**FINDING: `-inf` returns `nan` on a fully-masked row.** The lesson's `softmax`
subtracts the row max for stability; if the whole row is `-inf` the max is `-inf`
too, and `-inf - -inf` is `nan`. All 6 entries come back `nan`, silently, and the
`nan` then spreads through `weights @ V` into every downstream token. Causal
masking never triggers it because the diagonal is always kept -- but a padding
mask, the other half of "certain positions", hits it on the first fully-padded
row.

**FINDING: `-1e9` is bit-identical where it matters and safe where it is not.**
Across the causal mask the two fills agree to **0.0** -- not approximately, the
same floats -- because `exp(-1e9 - max)` underflows to exactly 0. On the
fully-masked row it returns a uniform distribution instead of `nan`. The safer
fill costs nothing, and the exercise prescribes the other one.

**CONTROL: the last row is untouched.** Causal masking changes n-1 of n rows
exactly; row n-1 already attends to everything, and its weights come back
bit-identical. That is how you know the mask is upper-triangular and not
something else that happens to look plausible.

Structure: `masked_attention` is the modified function; `entropies` reads how
far each row collapsed; `fills` runs the same mask under both fill values.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "02-self-attention-from-scratch"
D_MODEL, DK, DV, TOKENS = 16, 8, 8, 6
BIG_NEGATIVE = -1e9


def masked_attention(ref, np, Q, K, V, mask=None, fill=None):
    """The lesson's own function plus one optional argument, and nothing else."""
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -np.inf if fill is None else fill)
    weights = ref.softmax(scores)
    return weights @ V, weights


def entropies(np, weights):
    """Row entropy in nats; a row pinned to one position reads exactly 0."""
    safe = np.where(weights > 0, weights, 1.0)
    return -(weights * np.log(safe)).sum(axis=-1)


def shown(values, places=3):
    """A row of floats as the check prints it, so verify() stays a list of facts."""
    return [round(float(v), places) for v in values]


def projections(ref, np):
    """The lesson's own X and its own SelfAttention weights, as main() builds them."""
    x = np.random.default_rng(42).normal(0, 1, (TOKENS, D_MODEL))
    layer = ref.SelfAttention(D_MODEL, DK, DV, seed=42)
    return x @ layer.Wq, x @ layer.Wk, x @ layer.Wv


def solve():
    import numpy as np
    ref = parity.load_reference(PHASE, LESSON, "self_attention")
    q, k, v = projections(ref, np)
    causal = np.tril(np.ones((TOKENS, TOKENS), dtype=bool))
    padded = causal.copy()
    padded[2, :] = False
    with np.errstate(invalid="ignore"):
        fills = {f: masked_attention(ref, np, q, k, v, causal, f)[1]
                 for f in (None, BIG_NEGATIVE)}
        starved = {f: masked_attention(ref, np, q, k, v, padded, f)[1]
                   for f in (None, BIG_NEGATIVE)}
    plain, reference = masked_attention(ref, np, q, k, v), ref.scaled_dot_product_attention(q, k, v)
    causal_weights, row = fills[None], fills[None][0]
    entropy, bound = entropies(np, causal_weights), np.log(np.arange(1, TOKENS + 1))
    return {
        "extends": np.array_equal(plain[1], reference[1]) and np.array_equal(plain[0], reference[0]),
        "first_row": shown(row), "one_hot": bool(row[0] == 1.0 and row[1:].sum() == 0.0),
        "v0_exact": bool(((causal_weights @ v)[0] == v[0]).all()),
        "gap": float(np.abs(causal_weights - fills[BIG_NEGATIVE]).max()),
        "nans": {f: int(np.isnan(w).sum()) for f, w in starved.items()},
        "starved_row": shown(starved[BIG_NEGATIVE][2][:3], 4),
        "entropy": shown(entropy), "plain_entropy": shown(entropies(np, reference[1])),
        "bound": shown(bound), "pinned": bool(entropy[0] == 0.0),
        "within_bound": bool((entropy <= bound + 1e-12).all()),
        "last_row_same": np.array_equal(causal_weights[-1], reference[1][-1]),
        "changed": int((np.abs(causal_weights - reference[1]) > 0).any(axis=1).sum()),
    }


def verify(result):
    collapsed = result["pinned"] and result["within_bound"] and result["last_row_same"]
    return [
        practice.Check(
            "ANSWER: optional means optional -- with no mask it is the lesson's own function",
            result["extends"],
            "masked_attention(Q, K, V) returns weights and output bit-identical to "
            "scaled_dot_product_attention(Q, K, V) on the lesson's own X and SelfAttention "
            "weights, so the modification is a strict extension rather than a replacement",
        ),
        practice.Check(
            "ANSWER: under a causal mask token 0 is pinned to itself, exactly",
            result["one_hot"] and result["v0_exact"],
            f"row 0 comes back {result['first_row']} -- one-hot to the bit, not to a "
            "tolerance -- and the output for the first token is bit-identical to v_0. A decoder's "
            "first position has no information but itself, which is why generation needs a prompt",
        ),
        practice.Check(
            "FINDING: the prescribed -inf returns nan on a fully-masked row",
            result["nans"][None] == TOKENS and result["nans"][BIG_NEGATIVE] == 0,
            f"the lesson's softmax subtracts the row max; if the whole row is -inf the max is -inf "
            f"and -inf - -inf is nan. All {result['nans'][None]} entries come back nan and spread "
            f"through weights @ V, where {BIG_NEGATIVE:.0e} gives "
            f"{[round(float(w), 4) for w in result['starved_row'][:3]]}... instead. Causal masking "
            "never trips it -- padding, the other half of 'certain positions', does",
        ),
        practice.Check(
            "FINDING: -1e9 is bit-identical where -inf works, so the safe fill costs nothing",
            result["gap"] == 0.0,
            f"across the whole causal mask the two fills differ by {result['gap']} -- the same "
            f"floats, not merely close -- because exp({BIG_NEGATIVE:.0e} - max) underflows to "
            "exactly 0. There is no numerical argument for the fill the exercise prescribes",
        ),
        practice.Check(
            "CONTROL: the mask collapses n-1 rows to <= log(i+1) and leaves the last alone",
            collapsed and result["changed"] == TOKENS - 1,
            f"row entropies fall to {result['entropy']} against the "
            f"{result['bound']} ceiling of log(i+1), from "
            f"{result['plain_entropy']} unmasked. Row {TOKENS - 1} "
            f"is bit-identical to unmasked and exactly {result['changed']} rows moved, which is "
            "how you know the mask is triangular and not merely plausible",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
