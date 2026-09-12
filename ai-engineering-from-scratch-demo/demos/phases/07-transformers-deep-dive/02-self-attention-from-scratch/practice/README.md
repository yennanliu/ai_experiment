<!-- generated:start -->
# 07-transformers-deep-dive / 02-self-attention-from-scratch

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/07-transformers-deep-dive/02-self-attention-from-scratch/) · upstream spec
`phases/07-transformers-deep-dive/02-self-attention-from-scratch/docs/en.md`

```bash
uv run demo practice run 02-self-attention-from-scratch --ex 1
uv run demo explain 02-self-attention-from-scratch --ex 1
uv run pytest demos/phases/07-transformers-deep-dive/02-self-attention-from-scratch
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Modify `scaled_dot_product_attention` to accept an optional mask matrix that sets certain pos… | code | T1 | `ex01_the_negative_infinity_the_exercise_asks_for.py` |
| 2 | Implement multi-head attention from scratch: split Q, K, V into `n_heads` chunks, run attenti… | code | T1 | `ex02_main_prints_the_same_matrix_twice.py` |
| 3 | Take two different sentences of the same length, feed them through the same SelfAttention ins… | code | T1 | `ex03_the_sentence_never_enters_the_computation.py` |
<!-- generated:end -->

## Answers

Three exercises about a 90-line NumPy file, and each one lands on something the
file does that its own output does not admit. Exercise 1's prescribed fill value
is the unsafe one. Exercise 2 asks for a construction the lesson already has —
and finding that out means noticing that `main()` prints the same attention
matrix twice under two different headings. Exercise 3 asks what changes between
two sentences, and the honest answer is that the sentence never reaches the
computation at all.

All three are **T1** — `numpy` and nothing else, as the lesson is.

### 1 — the negative infinity the exercise asks for

The modified function, with `mask=None`, is bit-identical to
`scaled_dot_product_attention`; a strict extension, not a replacement. Under a
causal mask on the lesson's own `X`:

| row | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| entropy, unmasked | 1.732 | 1.719 | 1.735 | 1.661 | 1.647 | 1.712 |
| entropy, causal | **0.000** | 0.686 | 1.089 | 1.221 | 1.465 | **1.712** |
| ceiling `log(i+1)` | 0.000 | 0.693 | 1.099 | 1.386 | 1.609 | 1.792 |

**ANSWER: it works, and token 0 is pinned to itself exactly.** Row 0 comes back
`[1, 0, 0, 0, 0, 0]` — one-hot to the bit, not to a tolerance — and the output
for the first token is bit-identical to `v_0`. A decoder's first position has no
information but itself, which is why generation has to start from a prompt.

**FINDING: `-inf` returns `nan` on a fully-masked row.** The lesson's `softmax`
subtracts the row max for stability. If a whole row is `-inf` then the max is
`-inf` too, and `-inf − (−inf)` is `nan` — all 6 entries, silently, and the `nan`
then spreads through `weights @ V` into every token downstream. Causal masking
never trips it, because the diagonal is always kept. A **padding** mask — the
other half of "certain positions" — hits it on the first fully-padded row.

**FINDING: `-1e9` is bit-identical where `-inf` works.** Across the entire causal
mask the two fills differ by **0.0** — the same floats, not merely close —
because `exp(-1e9 − max)` underflows to exactly zero. It also returns a uniform
row instead of `nan` where `-inf` fails. There is no numerical argument for the
fill the exercise prescribes; every production implementation uses the other one.

**CONTROL: exactly `n−1` rows move.** Row 5 already attends to everything and
comes back bit-identical to unmasked. That is how you know the mask is
upper-triangular rather than something that merely looks plausible.

### 2 — `main()` prints the same matrix twice

The exercise says "split Q, K, V into `n_heads` chunks". The lesson's
`MultiHeadSelfAttention` does not split anything — it builds `n_heads` whole
`SelfAttention` modules, each projecting the full input through its own
`(d_model, dk)` matrices. So: build the exercise's version and ask whether it is
the same function.

**ANSWER: bit-identical, given the same weights.** `hstack` the per-head `Wq`,
`Wk`, `Wv` into one `(16, 16)`, project once, slice into 8-wide chunks, run the
lesson's own `scaled_dot_product_attention` on each: output differs by **0.0**
and every per-head weight matrix is identical. Different memory layout, same
arithmetic.

**FINDING: the parameter count is identical too.** `n_heads · (d_model · dk)` =
`d_model²` = **256** per projection, because `dk = d_model / n_heads`. Splitting
is not a saving. It buys one GEMM instead of `n_heads` — which is why frameworks
do it, and not because heads cost less than attention.

**FINDING: `main()` prints the same attention matrix twice.** Head `i` gets
`seed = seed + i`, so head 0 is `SelfAttention(16, 8, 8, seed=42)` — the object
built twenty lines earlier for the single-head demo. Everything printed under
**"Head 1 attention weights"** is, bit for bit, what was already printed under
**"Attention weights"**. Nothing in the output says so, and the ASCII heatmap
above it invites you to read the two as different.

**FINDING: `seed + i` aliases across configurations.** `MultiHeadSelfAttention(
16, 2, seed=42).heads[1]` and the same call at `seed=43`'s `heads[0]` are the
same draw. Two "independent" models share a head whenever their seeds differ by
less than `n_heads`, so a seed sweep of step 1 re-measures half its heads.

**CONTROL: `Wo` is the only thing that mixes the heads.** Concatenation leaves
head 0 in columns 0–7 and head 1 in 8–15, disjoint. `Wo` is `(16, 16)` with all
**256** entries nonzero, so zeroing head 1 moves **all 16** output columns. Drop
`Wo` and this is two attentions stapled together, not multi-head attention.

### 3 — the sentence never enters the computation

`SelfAttention.forward(self, X)` takes an array and no tokens. `main()` builds
`X` from `rng.normal` and uses its `sentence` list only for row and column
labels. So "two different sentences" can only mean two different `X`.

**WHAT CHANGES: the numbers, on a matrix that means nothing either way.** Mean
`|W1 − W2|` = **0.104**, largest single disagreement **0.378**, on entries
averaging 0.167. Both stay high-entropy — **0.95** and **0.88** of `log 6` —
because these projections are random rather than trained. The two patterns differ
the way two draws differ.

**WHAT STAYS THE SAME: the instance, the shape, the row sums.** `forward` touches
no state, so `Wq` is bit-identical afterwards; both matrices are `(6, 6)`; every
row sums to 1 within **2.2e-16**. Those are properties of `softmax` and of the
instance, and would hold for any input at all.

**FINDING: the token strings are decorative.** Feed the same `X` and call the
words anything you like — the matrix is bit-identical. The exercise cannot be run
the way it is worded; what it is really comparing is two seeds.

**FINDING: the invariant that answers the question is that word order does not
exist here.** Permute the rows of `X` and the attention matrix comes back as the
*same matrix* with rows and columns permuted — to **5.6e-17** — and every output
row is the old one relabelled, to **2.2e-16**. "The cat sat on the mat" and "mat
the on sat cat the" are the same input to this module. That is the entire reason
Lesson 04 exists, and it is the one thing that stays the same between *any* two
sentences that are permutations of each other.

**CONTROL: the diagonal is not privileged.** Over 2,000 random sentences the row
argmax lands on the diagonal **17.6%** of the time, against **16.7%** by chance.
`Wq` and `Wk` are independent draws, so `q_i · k_i` has no special status; the
fraction of a point of excess comes from `q_i` and `k_i` sharing `x_i`, not from
any "the token looks at itself" mechanism the Key Terms table describes.
