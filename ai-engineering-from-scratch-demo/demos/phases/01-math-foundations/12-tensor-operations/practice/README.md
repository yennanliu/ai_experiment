<!-- generated:start -->
# 01-math-foundations / 12-tensor-operations

Solutions to all 4 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/01-math-foundations/12-tensor-operations/) · upstream spec
`phases/01-math-foundations/12-tensor-operations/docs/en.md`

```bash
uv run demo practice run 12-tensor-operations --ex 1
uv run demo explain 12-tensor-operations --ex 1
uv run pytest demos/phases/01-math-foundations/12-tensor-operations
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy -- Reshape round-trip. Take a tensor of shape `(2, 3, 4)`. Reshape it to `(6, 4)`, then… | code | T0 | `ex01_reshape_round_trip.py` |
| 2 | Medium -- Implement broadcasting. Extend the `Tensor` class with a `broadcast_to(shape)` meth… | code | T0 | `ex02_broadcasting.py` |
| 3 | Hard -- Build einsum from scratch. Implement a basic `einsum(subscripts, *tensors)` function… | code | T0 | `ex03_einsum_from_scratch.py` |
| 4 | Hard -- Attention shape tracker. Write a function that takes `batch_size`, `seq_len`, `embed_… | code | T0 | `ex04_attention_shape_tracker.py` |
<!-- generated:end -->

## Notes

**1 — reshape reinterprets, it does not copy.** The flat buffer is byte-identical
across all four snapshots; only the strides change, each being the suffix product
of its shape: `(12,4,1) → (4,1) → (1,) → (12,4,1)`. The check that distinguishes
this from a copy is the refusal: `reshape((5,5))` on 24 elements raises rather
than padding or truncating.

The exercise says to verify by *printing* the flat data, which requires reaching
for `Tensor._data` — the class exposes `shape`, `rank`, `size` and `strides` as
properties but no public accessor for the buffer itself.

**2 — the shape pair that catches a naive implementation** is `(3,2)` against
`(4,2)`. Both are rank 2, so a check that only compares ranks lets it through and
mangles the result. Right-aligned per-axis comparison is what makes broadcasting
correct, and the solution asserts the refusal rather than only the success. For
reference, the unmodified `Tensor` rejects `(3,1)+(1,4)` with
*"Shape mismatch. Use broadcast() first"* — which is the behaviour the exercise
asks to replace.

**3 — four named patterns are a floor, not a spec.** An implementation that
special-cased `i,i->`, `ij,jk->ik`, `i,j->ij` and `ij->ji` would pass the exercise
and nothing else. The solution writes one generic loop — free indices outer,
everything else summed — and then runs five patterns it was never told about:
`ij->` (empty output), `ii->i` (repeated index within one operand), `bij,bjk->bik`
(batched), `ij,j->i`, and `ij,jk,kl->il` (three operands). All nine match
`np.einsum` to 6.7e-16. Batch indices need no special handling because they are
just free indices appearing in both operands, and a chain needs none because every
non-output index is summed.

**4 — a shape tracker that only prints cannot be verified.** The exercise says to
check against `demo_attention_einsum()`, so the tracker returns its table and a
real einsum forward pass runs alongside it; all 8 steps agree at B=2, T=8, E=64,
H=4.

Two things fall out of the table. `embed_dim` must be divisible by `num_heads` and
nothing in the signature enforces it — 64 // 5 = 12 would silently drop 4
dimensions per head, so the tracker raises. And `head_split` holds B·T·E elements
*regardless of H*, because heads only repartition the embedding, while `scores` is
B·H·T² — the one step that grows quadratically in sequence length, and the reason
long context is expensive.
