"""Exercise 4 — the exact shape at every step of multi-head attention.

    **Hard -- Attention shape tracker.** Write a function that takes
    `batch_size`, `seq_len`, `embed_dim`, and `num_heads` as inputs and prints
    the exact shape at every step of multi-head attention: input, Q/K/V
    projection, head split, attention scores, softmax weights, weighted sum,
    head merge, output projection. Verify against the `demo_attention_einsum()`
    output.

Reading of the exercise: a tracker that only *prints* cannot be verified against
anything, so it returns its table, a real forward pass runs alongside it, and
check 2 compares the two step by step. Check 4 covers the constraint the
signature does not enforce — embed_dim must divide num_heads.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "12-tensor-operations"
B, T, E, H = 2, 8, 64, 4       # the demo's own configuration
STEPS = ("input", "q_projection", "head_split", "scores", "weights",
         "weighted_sum", "head_merge", "output")


def track(batch_size, seq_len, embed_dim, num_heads):
    """Predicted shapes, derived from the four inputs alone."""
    if embed_dim % num_heads:
        raise ValueError(f"embed_dim {embed_dim} is not divisible by "
                         f"num_heads {num_heads}")
    head_dim = embed_dim // num_heads
    return {
        "input": (batch_size, seq_len, embed_dim),
        "q_projection": (batch_size, seq_len, embed_dim),
        "head_split": (batch_size, num_heads, seq_len, head_dim),
        "scores": (batch_size, num_heads, seq_len, seq_len),
        "weights": (batch_size, num_heads, seq_len, seq_len),
        "weighted_sum": (batch_size, num_heads, seq_len, head_dim),
        "head_merge": (batch_size, seq_len, embed_dim),
        "output": (batch_size, seq_len, embed_dim),
    }


def forward(numpy, rng):
    """A real pass, mirroring demo_attention_einsum's einsum formulation."""
    head_dim = E // H
    X = rng.normal(size=(B, T, E))
    weights = {name: rng.normal(size=(E, E)) * 0.02 for name in "qkvo"}
    Q = numpy.einsum("bte,ek->btk", X, weights["q"])
    K = numpy.einsum("bte,ek->btk", X, weights["k"])
    V = numpy.einsum("bte,ek->btk", X, weights["v"])
    split = [t.reshape(B, T, H, head_dim).transpose(0, 2, 1, 3) for t in (Q, K, V)]
    scores = numpy.einsum("bhtd,bhsd->bhts", split[0], split[1]) / head_dim ** 0.5
    shifted = scores - scores.max(axis=-1, keepdims=True)
    attention = numpy.exp(shifted)
    attention /= attention.sum(axis=-1, keepdims=True)
    summed = numpy.einsum("bhts,bhsd->bhtd", attention, split[2])
    merged = summed.transpose(0, 2, 1, 3).reshape(B, T, E)
    out = numpy.einsum("bte,ek->btk", merged, weights["o"])
    return {"input": X.shape, "q_projection": Q.shape, "head_split": split[0].shape,
            "scores": scores.shape, "weights": attention.shape,
            "weighted_sum": summed.shape, "head_merge": merged.shape,
            "output": out.shape}, attention


def solve():
    numpy = parity.try_numpy()
    if numpy is None:
        raise practice.Skip("needs numpy — uv sync --extra math")
    predicted = track(B, T, E, H)
    actual, attention = forward(numpy, numpy.random.default_rng(42))
    try:
        track(B, T, 64, 5)
        refusal = "accepted embed_dim=64 with num_heads=5"
    except ValueError as exc:
        refusal = f"ValueError: {exc}"
    return {"predicted": predicted, "actual": {k: tuple(v) for k, v in actual.items()},
            "refusal": refusal,
            "rows_sum_to_one": float(numpy.abs(attention.sum(axis=-1) - 1).max()),
            "elements": {k: B * H * T * T if k in ("scores", "weights")
                         else B * T * E for k in predicted}}


def verify(result):
    predicted, actual = result["predicted"], result["actual"]
    mismatched = {k: (predicted[k], actual[k]) for k in STEPS if predicted[k] != actual[k]}
    return [
        practice.Check(f"all {len(STEPS)} steps predicted from the four inputs alone",
                       set(predicted) == set(STEPS),
                       "; ".join(f"{k}{predicted[k]}" for k in STEPS)),
        practice.Check("every predicted shape matches a real forward pass",
                       not mismatched,
                       f"B={B}, T={T}, E={E}, H={H}, head_dim={E // H}: "
                       f"{len(STEPS)} of {len(STEPS)} agree"),
        practice.Check("the softmax rows are a distribution, so 'weights' is earned",
                       result["rows_sum_to_one"] < 1e-12,
                       f"worst |Σ weights − 1| = {result['rows_sum_to_one']:.3g} over "
                       f"all {B * H * T} attention rows"),
        practice.Check("a non-divisible embed_dim is refused, not truncated",
                       "ValueError" in result["refusal"],
                       f"embed_dim=64, num_heads=5 -> {result['refusal']} — the signature "
                       f"does not enforce this and 64 // 5 = 12 would silently drop 4 "
                       f"dimensions per head"),
        practice.Check("the scores step is the only one that grows with seq_len²",
                       result["elements"]["scores"] == B * H * T * T
                       and result["elements"]["head_split"] == B * T * E,
                       f"scores holds {result['elements']['scores']} elements against "
                       f"{result['elements']['head_split']} for the split — head_split is "
                       f"B·T·E regardless of H, since the heads only repartition E. "
                       f"Scores is B·H·T², which is where long context gets expensive"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
