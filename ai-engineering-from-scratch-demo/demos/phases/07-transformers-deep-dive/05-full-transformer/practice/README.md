<!-- generated:start -->
# 07-transformers-deep-dive / 05-full-transformer

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/07-transformers-deep-dive/05-full-transformer/) · upstream spec
`phases/07-transformers-deep-dive/05-full-transformer/docs/en.md`

```bash
uv run demo practice run 05-full-transformer --ex 1
uv run demo explain 05-full-transformer --ex 1
uv run pytest demos/phases/07-transformers-deep-dive/05-full-transformer
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Count the parameters in your encoder_block at `d_model=512, n_heads=8, ffn_expansion=4,… | code | T0 | `ex01_a_fifth_of_the_block_is_never_read.py` |
| 2 | Medium. Switch from post-norm to pre-norm. Initialize both and measure the activation norm af… | code | T0 | `ex02_the_exercise_has_it_backwards.py` |
| 3 | Hard. Implement a 4-layer encoder-decoder on a toy copy task (copy `x` reversed). Train 100 s… | code | T0 | `ex03_reversing_the_source_changes_nothing.py` |
<!-- generated:end -->

## Answers

The lesson is a pure-stdlib encoder-decoder in 254 lines, and all three exercises
turn out to be about what it already contains. Exercise 1's prescribed validation
counts weights the block never reads. Exercise 2 predicts the wrong sign, on an
arm whose behaviour is fixed by construction. Exercise 3 asks you to train a task
the stack cannot represent and then to "swap in" two things it already has. All
three are **T0** — pure stdlib, as the lesson is.

### 1 — a fifth of the block is never read

**ANSWER: 4,194,304 — exactly 2²².**

| | count |
|---|---:|
| attention, `4·d²` | 1,048,576 |
| SwiGLU FFN, `3·d·h` at `h = 4d = 2048` | 3,145,728 |
| **total read by `encoder_block`** | **4,194,304** |
| also held by `BlockParams` (cross-attention) | 1,048,576 |
| **total held** | **5,242,880** |

The set of weights that count is decided mechanically — by parsing
`encoder_block`'s source for the attributes it reads — rather than assumed.

**FINDING: `BlockParams` allocates decoder cross-attention for every block.**
`Wq_x, Wk_x, Wv_x, Wo_x` are constructed unconditionally and `encoder_block`
never mentions them. `sum(p.numel() for p in block.parameters())` — the
validation the exercise prescribes — counts what the object *holds*, so it would
"validate" the answer against **1.25×** the right number, and agree with itself.

**FINDING: SwiGLU at `ffn_expansion=4` costs 1.5× the FFN it replaces.** Three
matrices where ReLU has two: an extra **1,048,576** parameters, which is the
exact size of the whole attention sublayer. The convention is `8/3` expansion
with SwiGLU so the two match; these settings add 50% to the FFN silently.

**FINDING: zero normalisation parameters.** `rms_norm(X, eps)` and
`layer_norm(X, eps)` both take an input and a constant — no learnable scale — so
a standard block's gains are simply absent from this count.

**CONTROL: the one-liner cannot run.** `torch` is absent and `BlockParams` is a
plain object with no `parameters()`. Summing `len(Matrix.data)` over its
attributes is the equivalent — and doing that is exactly what makes the dead
weights visible.

### 2 — the exercise has it backwards

Both stacks are driven by the *same* twelve `BlockParams` and the same input, so
the only difference between the traces is where the norm sits.

| layer | 1 | 3 | 6 | 12 |
|---|---:|---:|---:|---:|
| pre-norm row RMS | 1.180 | 1.630 | 2.142 | **2.791** |
| post-norm row RMS | 1.000000 | 1.000000 | 1.000000 | **1.000000** |

(input row RMS 0.982)

**FINDING: post-norm's activations cannot explode — at any depth.** `rms_norm` is
the *last* operation of every post-norm block, so the output's row RMS is 1 by
construction, for any initialisation and any input. Its twelve values span
**1.4e-07**, which is `rms_norm`'s own `eps=1e-6` and not dynamics. The
prescribed measurement has zero dynamic range. What explodes in a post-norm
transformer is the **gradient**, and the forward activation norm is precisely the
quantity that cannot see it.

**FINDING: pre-norm is the arm that grows.** The residual stream is never
normalised — only each branch's *input* is — so roughly independent contributions
add in variance. Growth against depth fits a log-log slope of **0.36**:
sub-linear, near the 0.5 of variance accumulation and far from the 0 a normalised
output gives.

**FINDING: the lesson is already pre-norm.** `encoder_block` is
`x + sublayer(rms_norm(x))`. The switch the exercise names runs the other way,
and the arm it then predicts wrongly is the one you have to write.

### 3 — reversing the source changes nothing

**ANSWER: the decoder output is invariant to any permutation of the source, to
5.3e-15.** Invariant, not equivariant. The encoder is equivariant — permuting the
source permutes `enc_out`'s rows, to **2.7e-15** — and cross-attention then sums
over those rows with weights that depend only on their content, so the
permutation cancels on the way out.

Feed the source **reversed** and every number the decoder produces moves by at
most **5.3e-15**. So do all **719** other orderings of six tokens. "Copy `x`
reversed" and "copy `x`" are the same target to this model: 100 steps of training
report a floor, and it is the same floor for both arms, because the task is
outside the function class rather than hard.

**FINDING: two of the three swaps are no-ops.** `encoder_block` and
`decoder_block` both call `rms_norm` — `layer_norm` is defined in the module and
called by neither — and `BlockParams` defaults `use_swiglu=True`. The baseline
the exercise asks you to improve already *is* RMSNorm + SwiGLU.

**FINDING: RoPE is not an improvement here, it is the enabling condition.**
`apply_rope` does not exist in this module, and neither does any other positional
encoding — which is exactly why the permutation symmetry above holds. RoPE is the
only one of the three swaps that changes what the model can *represent*: it takes
the loss from a floor to zero, and the other two contribute nothing to that.
