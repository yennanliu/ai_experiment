"""Exercise 2 — main() prints the same attention matrix twice.

    Implement multi-head attention from scratch: split Q, K, V into `n_heads`
    chunks, run attention on each, concatenate, and project through a final
    weight matrix Wo

Reading of the exercise: "split Q, K, V into `n_heads` chunks" describes a
different construction from the one the lesson ships. `MultiHeadSelfAttention`
builds `n_heads` whole `SelfAttention` modules, each projecting the full input
through its own `(d_model, dk)` matrices; the exercise projects **once** through
a fused `(d_model, d_model)` matrix and slices the result. So the solution builds
the exercise's version and asks whether it is the same function.

**ANSWER: bit-identical, given the same weights.** Horizontally stacking the
lesson's per-head `Wq`, `Wk`, `Wv` into one `(16, 16)` matrix, projecting once,
slicing into `dk`-wide chunks and running the lesson's own
`scaled_dot_product_attention` on each reproduces the lesson's output to **0.0**
and every per-head weight matrix exactly. The two formulations differ in memory
layout, not in arithmetic: one matmul instead of `n_heads`.

**FINDING: the parameter count is identical too.** `n_heads * (d_model * dk)` =
`d_model^2` = 256 per projection, because `dk = d_model / n_heads`. Splitting is
not a saving; it is the same weights read in a different order, which is why
frameworks do it -- one GEMM beats `n_heads` small ones -- and not because heads
are cheaper than attention.

**FINDING: `main()` prints the same matrix twice.** `MultiHeadSelfAttention`
gives head `i` `seed=seed + i`, so head 0 is `SelfAttention(16, 8, 8, seed=42)`
-- the very object built twenty lines earlier for the single-head demo. Under
"Head 1 attention weights" the lesson reprints, bit for bit, the matrix already
shown under "Attention weights", and nothing in the output says so.

**FINDING: the offset scheme aliases across configurations.** `seed + i` means
`MultiHeadSelfAttention(16, 2, seed=42).heads[1]` and `...(16, 2, seed=43)
.heads[0]` are the same draw. Two "independent" models share a head whenever
their seeds differ by less than `n_heads`.

**CONTROL: `Wo` is the only thing that mixes the heads.** Concatenation leaves
head 0 in columns 0-7 and head 1 in columns 8-15, disjoint. `Wo` is dense -- all
256 entries nonzero -- so zeroing head 1 moves all 16 output columns. Drop `Wo`
and "multi-head attention" is two attentions stapled together.

Structure: `fuse` stacks the per-head weights; `chunked` is the exercise's
construction; `mixing` counts how far one head reaches after `Wo`.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "02-self-attention-from-scratch"
D_MODEL, N_HEADS, TOKENS, SEED = 16, 2, 6, 42


def fuse(np, heads, name):
    """The exercise's single projection, assembled from the lesson's per-head ones."""
    return np.hstack([getattr(head, name) for head in heads])


def chunked(ref, np, mha, x):
    """Project once, slice into n_heads chunks, attend, concatenate, project by Wo."""
    dk = mha.dk
    q, k, v = (x @ fuse(np, mha.heads, name) for name in ("Wq", "Wk", "Wv"))
    pairs = [ref.scaled_dot_product_attention(q[:, c:c + dk], k[:, c:c + dk], v[:, c:c + dk])
             for c in range(0, dk * mha.n_heads, dk)]
    return np.concatenate([out for out, _ in pairs], axis=-1) @ mha.Wo, [w for _, w in pairs]


def mixing(np, mha, outputs, reference):
    """Output columns disturbed when the last head's contribution is zeroed."""
    silenced = list(outputs[:-1]) + [np.zeros_like(outputs[-1])]
    moved = np.abs(np.concatenate(silenced, axis=-1) @ mha.Wo - reference) > 1e-12
    return int(moved.any(axis=0).sum())


def solve():
    import numpy as np
    ref = parity.load_reference(PHASE, LESSON, "self_attention")
    x = np.random.default_rng(SEED).normal(0, 1, (TOKENS, D_MODEL))
    mha = ref.MultiHeadSelfAttention(D_MODEL, N_HEADS, seed=SEED)
    reference, head_weights = mha.forward(x)
    mine, mine_weights = chunked(ref, np, mha, x)
    single = ref.SelfAttention(D_MODEL, D_MODEL // N_HEADS, D_MODEL // N_HEADS, seed=SEED)
    shifted = ref.MultiHeadSelfAttention(D_MODEL, N_HEADS, seed=SEED + 1)
    return {
        "gap": float(np.abs(mine - reference).max()),
        "weights_match": all(np.array_equal(a, b) for a, b in zip(mine_weights, head_weights)),
        "fused_shape": fuse(np, mha.heads, "Wq").shape,
        "params": fuse(np, mha.heads, "Wq").size,
        "reprint": np.array_equal(head_weights[0], single.forward(x)[1]),
        "aliased": np.array_equal(mha.heads[1].Wq, shifted.heads[0].Wq),
        "dense": int(np.count_nonzero(mha.Wo)), "wo_shape": mha.Wo.shape,
        "moved": mixing(np, mha, [head.forward(x)[0] for head in mha.heads], reference),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: splitting one fused projection reproduces the lesson bit for bit",
            result["gap"] == 0.0 and result["weights_match"],
            f"stacking the per-head Wq/Wk/Wv into one {result['fused_shape']} matrix, projecting "
            f"once, slicing into dk-wide chunks and running the lesson's own "
            f"scaled_dot_product_attention gives output differing by {result['gap']} and every "
            "per-head weight matrix identical. Different memory layout, same arithmetic",
        ),
        practice.Check(
            "FINDING: the parameter count is identical, so splitting saves nothing but GEMMs",
            result["params"] == D_MODEL ** 2,
            f"n_heads * (d_model * dk) = {result['params']} = d_model^2, because dk = d_model / "
            f"n_heads. The fused form is one matmul instead of {N_HEADS}, which is why frameworks "
            "prefer it -- not because heads cost less than attention",
        ),
        practice.Check(
            "FINDING: main() prints the same attention matrix twice",
            result["reprint"],
            f"MultiHeadSelfAttention gives head i seed={SEED} + i, so head 0 is "
            f"SelfAttention({D_MODEL}, {D_MODEL // N_HEADS}, {D_MODEL // N_HEADS}, seed={SEED}) "
            "-- the object built for the single-head demo twenty lines earlier. 'Head 1 attention "
            "weights' reprints 'Attention weights' bit for bit, and the output never says so",
        ),
        practice.Check(
            "FINDING: seed + i aliases heads across configurations",
            result["aliased"],
            f"MultiHeadSelfAttention(..., {N_HEADS}, seed={SEED}).heads[1] and the same call at "
            f"seed={SEED + 1}.heads[0] are the same draw. Two models whose seeds differ by less "
            f"than n_heads share heads, so a seed sweep of step 1 re-measures {N_HEADS - 1} of "
            f"every {N_HEADS} heads it thinks it replaced",
        ),
        practice.Check(
            "CONTROL: Wo is the only thing that mixes the heads",
            result["moved"] == D_MODEL and result["dense"] == D_MODEL ** 2,
            f"concatenation leaves head 0 in columns 0-{D_MODEL // N_HEADS - 1} and head 1 in the "
            f"rest, disjoint. Wo is {result['wo_shape']} with all {result['dense']} entries "
            f"nonzero, so zeroing the last head moves {result['moved']} of {D_MODEL} output "
            "columns. Without Wo this is two attentions stapled together, not multi-head",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
