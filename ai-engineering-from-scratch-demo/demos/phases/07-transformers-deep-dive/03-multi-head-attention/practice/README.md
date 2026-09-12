<!-- generated:start -->
# 07-transformers-deep-dive / 03-multi-head-attention

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/07-transformers-deep-dive/03-multi-head-attention/) · upstream spec
`phases/07-transformers-deep-dive/03-multi-head-attention/docs/en.md`

```bash
uv run demo practice run 03-multi-head-attention --ex 1
uv run demo explain 03-multi-head-attention --ex 1
uv run pytest demos/phases/07-transformers-deep-dive/03-multi-head-attention
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Take the MHA from `code/main.py` and change `n_heads` from 1 to 16 with `d_model=64` fi… | code | T1 | `ex01_eleven_of_the_sixteen_head_counts_are_illegal.py` |
| 2 | Medium. Implement MQA (one KV head shared across all query heads). Measure how much parameter… | code | T0 | `ex02_the_cache_saving_is_unbounded_the_parameter_one_is_not.py` |
| 3 | Hard. Implement a tiny version of Multi-head Latent Attention: compress K,V to a rank-`r` lat… | code | T1 | `ex03_the_two_conditions_never_overlap.py` |
<!-- generated:end -->

## Answers

The lesson is a pure-stdlib `Matrix` class and five attention functions. It ships
no loss, no gradients and no optimizer — and two of its three exercises ask for a
loss curve. So the tiny one-layer model gets built here: the lesson's own MHA
forward, a hand-written backward pass and Adam, with the numpy forward checked
against `multi_head_attention` at `d_model=64` (agreeing to **9e-16**) before any
of its numbers are trusted.

The task both training exercises use is **two simultaneous lookups per
position**: each token carries two query keys and must emit the values held at
the two positions those keys match. It is the smallest task a single attention
distribution provably cannot do, which is what makes the head sweep mean
something.

Exercises 1 and 3 are **T1** (numpy); Exercise 2 is **T0** — it is arithmetic
plus the lesson's own `grouped_query_attention`.

### 1 — eleven of the sixteen head counts do not exist

**FINDING: only 5 of 16 legal.** `split_heads` asserts `d_model % n_heads == 0`.
At `d_model=64` the sweep "from 1 to 16" raises
`AssertionError("d_model not divisible by n_heads")` for **11 of its 16 values**.
The legal set is `{1, 2, 4, 8, 16}`, so the plot has five points.

**ANSWER: all three — help, plateau, and hurt — in that order.**

| n_heads | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| `d_head` | 64 | 32 | 16 | 8 | 4 |
| variance explained | 51.4% | **98.9%** | 98.6% | 92.8% | 71.8% |

One head **fails**: one attention distribution cannot be one-hot at two positions
at once, so it recovers about half the target — which is what averaging two
lookups gets you. Two heads **help**, by 47 points. Four **plateau** — the task
needs two lookups and buying more than two only changes how wide each one is.
Eight and sixteen **hurt**.

**MECHANISM: the turn is `d_head`, not `n_heads`.** The task's keys are
12-dimensional. Degradation begins exactly where `d_model / n_heads` drops below
12 — at `d_head = 8` and `4`, a head can no longer represent the key it is
matching on. Reported across three seeds the ordering is identical to within 1
point, so this is the shape of the curve and not a run.

### 2 — the cache saving is unbounded, the parameter saving is capped at half

MQA is `grouped_query_attention(..., n_kv_heads=1)` — the lesson already has it.
`repeat = n_heads // 1`, so every query head indexes `Kh_small[0]` and all 16
read **one** K head of width `d_model / n_heads = 4`.

| at `d_model=64`, `n_heads=16` | MHA | MQA | ratio |
|---|---:|---:|---:|
| parameters / layer | 16,384 | **8,704** | 1.88× (−46.875%) |
| KV cache / layer at N=2048 | 262,144 | **16,384** | **16×** (−93.75%) |

**FINDING: these are not the same kind of number.** MQA leaves `Wq` and `Wo` at
`d_model²` each and shrinks only `Wk` and `Wv`, so the parameter ratio is
`1/2 + 1/(2h)`. Measured at `d_model = n_heads = 4096` — one scalar per KV head,
as extreme as sharing gets — the drop is **49.9878%** and it never reaches 50%.
No amount of KV sharing removes more than half an attention block's weights.

The cache ratio is `1/h` with **no floor and no dependence on N or `d_model`**:
16× at 16 heads, 64× at 64. That asymmetry is the whole reason every decoder
since 2023 shares KV and none of them shares `Wq` — and it is only visible
because the exercise asks for both numbers side by side.

**CONTROL: `n_kv_heads = n_heads` is MHA, to 0.0.** The same function reproduces
`multi_head_attention` bit for bit when nothing is shared, so the group count is
the only thing MQA changes.

### 3 — the two conditions never overlap

"Within 1 bit of validation ppl" needs a conversion the exercise does not give.
Taken honestly: for a Gaussian residual, one bit per dimension is a factor of
**4 in MSE**, since `0.5·log₂(mse_a/mse_b) = 1`. The MLA is `c = X·Wd` at width
`r`, then `K = c·Wuk`, `V = c·Wuv`, with only `c` cached.

| r | cache vs MHA | below 1/8? | bits worse than full MHA |
|---:|---:|---|---:|
| 16 | 0.125 | no — *equal* | 2.81 |
| 32 | 0.250 | no | 1.82 |
| **36** | 0.281 | no | **0.71** |
| 40 | 0.312 | no | −0.19 |
| 64 | 0.500 | no | **−1.04** |

**ANSWER: no such `r` exists here, and the miss is 2.25×.** The cache condition
is arithmetic — MLA caches `r` numbers per position where MHA caches
`2·d_model = 128`, so `r/128 < 1/8` means **r < 16**. The quality condition first
holds at **r = 36**, a cache ratio of 0.281. They never meet.

**FINDING: the floor is information, not optimisation.** Every position's K and V
must carry its key and both of its values — `12 + 14 + 14 = 40` dimensions — and
the curve breaks exactly there: 1.82 bits worse at 32, 0.71 at 36, and *better
than full MHA* from 40 up. No learning rate pushes a rank-`r` bottleneck below
the rank of what it has to carry, so the exercise's 1/8 target sits under the
task's own floor before a single step is taken.

**FINDING: the free win is 2×, and it is an improvement.** At `r = d_model = 64`
the cache is already halved — one latent instead of a K *and* a V — and the model
is **1.04 bits better** than full MHA, because the shared low-rank factorisation
ties K and V together. The exercise asks where MLA starts to hurt; the
measurement says it starts by helping.
