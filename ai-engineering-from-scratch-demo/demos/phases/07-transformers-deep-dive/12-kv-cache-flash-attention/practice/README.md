<!-- generated:start -->
# 07-transformers-deep-dive / 12-kv-cache-flash-attention

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/07-transformers-deep-dive/12-kv-cache-flash-attention/) · upstream spec
`phases/07-transformers-deep-dive/12-kv-cache-flash-attention/docs/en.md`

```bash
uv run demo practice run 12-kv-cache-flash-attention --ex 1
uv run demo explain 12-kv-cache-flash-attention --ex 1
uv run pytest demos/phases/07-transformers-deep-dive/12-kv-cache-flash-attention
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py`. Confirm the naive and cached decoders produce the same output; note… | code | T0 | `ex01_the_two_op_counts_are_the_same_number.py` |
| 2 | Medium. Implement prefix caching: given a prompt P and several completions, run one forward p… | code | T0 | `ex02_the_speedup_is_one_when_the_prompt_is_short.py` |
| 3 | Hard. Implement a toy PagedAttention: KV cache in fixed 16-token blocks with a free-list. Whe… | code | T0 | `ex03_paging_beats_reserving_not_packing.py` |
<!-- generated:end -->

## Answers

The lesson is 157 lines of KV caching and Flash-style tiled softmax. Exercise 1
asks you to note a difference between two numbers that are equal. Exercise 2 asks
for "the" speedup of a quantity that spans 1.00× to the branch count. Exercise 3
names a comparison that has two different answers depending on what "contiguous"
means. All three are **T0**.

### 1 — the two op counts are the same number

**ANSWER: the outputs are identical to 0.0, and so are the op counts.**
`decode_naive` adds `t + 1` per step; `decode_cached` adds `len(cache)`, which is
`t + 1`. Both reach **55** at N=10 — `N(N+1)/2` exactly — and `main()` prints
them on two lines labelled `O(N²)` and `O(N)`.

**FINDING: the saving is in a quantity neither counter counts.**

| N | attention ops, both | K,V projections naive | cached | saving |
|---:|---:|---:|---:|---:|
| 10 | 55 | 55 | 10 | 5.5× |
| 100 | 5,050 | 5,050 | 100 | 50.5× |
| 1,000 | 500,500 | 500,500 | 1,000 | **500.5×** |

The lesson's own comment says so two lines further down — "counting K,V
recomputes would make naive O(N²) in matmuls" — and then prints the counter that
does not.

**FINDING: the tiled softmax is one ULP off, not bit-identical.** The module
docstring promises "bit-identical output tile-by-tile". Measured: **one ULP** at
tiles 2, 4, 7, 10 and 16, **two** at tile 1. Even a single tile covering the
whole sequence disagrees, and the gap does not grow with the number of tiles — so
it is the formulation, not the accumulation.

Everything here is quoted in units in the last place rather than in absolutes,
because `math.exp` is not bit-identical across libms: the same code lands on
1.1e-16 on one platform where it lands on 2.2e-16 on another. The ULP count is
the part that is a property of the algorithm.

### 2 — the speedup is 1.01× when the prompt is short

| prompt | completion | branches | re-encode | prefix-cached | speedup |
|---:|---:|---:|---:|---:|---:|
| 512 | 64 | 8 | 1,329,408 | 410,112 | **3.24×** |
| 2,000 | 64 | 8 | 17,048,640 | 3,041,640 | 5.61× |
| 512 | 512 | 8 | 4,198,400 | 3,279,104 | 1.28× |
| 64 | 512 | 8 | 1,329,408 | 1,314,848 | **1.01×** |
| 512 | 64 | 1 | 166,176 | 166,176 | **1.00×** |

**ANSWER: prefix caching buys exactly the share of the work that is prompt.**
With one branch it is 1.00× by definition; with completions four times the prompt
it is 1.01×, because every decode step is unshared. The lever is `P / L`, and the
ceiling is the branch count.

**FINDING: the lesson's `KVCache` cannot be branched.** No `copy()`, and
`cache.K` is a plain list — two branches taken from one cache alias it, and the
second sees the first's appended token. Branching needs an `O(P)` copy per
completion, which is why real implementations page the cache instead.

**CONTROL: the measured counts match the closed form to the token** (145,860 vs
45,360 at P=200, L=20, C=6), which is what lets the sweep reach sizes pure Python
could not run.

### 3 — paging beats reserving by 8×, and beats packing by nothing

1,000 completions, lengths log-normal around 180 tokens (mean 262, median 173,
capped at a 2,048 context).

**ANSWER: against reserve-at-max_len, paging is 8.1×.** A 32,768-token pool holds
**16** sequences reserved at 2,048 each and **129** paged. Internal waste falls
from **87.2%** to **2.82%**, and the paged overhead is **7.61 tokens per
sequence** — bounded by `BLOCK − 1 = 15` and independent of length. Reserving
costs **1,786** tokens per sequence on the same corpus, and that number grows
with the context window while the other does not.

**FINDING: against exact-size packing, paging is a wash.**

| pool | concurrent | contiguous failures | of those, fragmentation | paged failures |
|---:|---:|---:|---:|---:|
| 6,144 | 20 | 58 | 46 | **82** |
| 8,192 | 28 | 60 | 49 | 48 |
| 12,288 | 40 | 26 | 21 | 32 |

External fragmentation is real — **81%** of the contiguous failures had enough
total free space and no single hole big enough — and block rounding costs about
what removing it saves. At the smallest pool, paging loses.

**FINDING: exact-size packing is not available anyway.** It needs the
completion's length at allocation time, which is the one number an autoregressive
decoder does not have. That is why the real alternative is reserve-at-max_len,
and why 8.1× is the number that matters.
