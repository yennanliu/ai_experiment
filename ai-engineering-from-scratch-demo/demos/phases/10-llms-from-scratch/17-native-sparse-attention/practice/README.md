<!-- generated:start -->
# 10-llms-from-scratch / 17-native-sparse-attention

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/10-llms-from-scratch/17-native-sparse-attention/) · upstream spec
`phases/10-llms-from-scratch/17-native-sparse-attention/docs/en.md`

```bash
uv run demo practice run 17-native-sparse-attention --ex 1
uv run demo explain 17-native-sparse-attention --ex 1
uv run pytest demos/phases/10-llms-from-scratch/17-native-sparse-attention
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py` on a 1024-token synthetic. Sweep `(l, k, w)` across three presets and prin… | code | T0 | `ex01_the_papers_own_preset_costs_more_than_dense.py` |
| 2 | Replace the mean-pool compressor with a tiny learned MLP (2-layer, hidden 32). Train it on a… | code | T0 | `ex02_the_baseline_is_exact_and_cannot_be_beaten.py` |
| 3 | Implement the gate MLP. It takes the query as input and outputs three scalars. Show that the… | code | T0 | `ex03_the_gate_cannot_see_which_branch_found_anything.py` |
| 4 | Compute the KV cache memory budget for an NSA-enabled 70B model at 128k context. KV heads are… | code | T0 | `ex04_the_fine_branch_equals_full_attention_at_1024.py` |
| 5 | Read Section 4 of the NSA paper (arXiv:2502.11089) and explain in three sentences why the com… | code | T0 | `ex05_the_argmax_has_no_gradient_so_the_score_must_be_borrowed.py` |
<!-- generated:end -->

## Answers

`code/main.py` implements NSA's three branches -- compressed, selected,
sliding-window -- with a mean-pool compressor, a top-k selector and a gate, plus
two key-counting functions. The five exercises find that the sequence the lesson
runs it on is shorter than the method's break-even point, and that two of the
components the exercises ask you to improve cannot be improved as specified.

All five are **T0** on **no** dependency group (stdlib only).

> Exercise 5 is scaffolded as a prose item ("read the paper, explain in three
> sentences"). It is built here as a **measurement**, because "tie the answer to
> gradient flow" is a claim about a derivative and the function in question is
> fifteen lines away.

### 1 — the paper's own preset costs more than dense

| preset | keys | share of full | needle recall |
|---|---:|---:|---:|
| l=32, k=8, W=512 | 800 | 78.1% | 100% |
| **l=64, k=16, W=512** | **1552** | **151.6%** | 100% |
| l=64, k=4, W=256 | **528** | 51.6% | 100% |
| l=128, k=4, W=128 | 648 | 63.3% | 100% |

**ANSWER: the cheapest preset is `l=64, k=4, W=256`, and the recall constraint
never binds.**

**FINDING: `l=64, k=16, W=512` reads 152% of full attention at 1024 tokens.**
`count_nsa` is `N/l + k·l + W`, and two of its three terms do not depend on `N`.
That preset breaks even at **N = 1560**; the exercise runs it at 1024.

**FINDING: 95% recall is not a constraint here.** `synthesize_sequence` fills
the signal block with copies of one unit pattern, so the block's *mean* is that
pattern and it ranks first however coarse the compression.

**MECHANISM: the branches trade against each other.** Doubling `l` cuts the
compressed term and doubles the selected one — which is why the coarsest preset
is not the cheapest.

### 2 — the baseline is exact and cannot be beaten

```text
target = the block average
compress_mean = the block average          →  MSE = 0.000e+00, by construction
2-layer MLP, hidden 32, 80 epochs          →  MSE = 1.19e-05   (from 8.94e-02)
```

**ANSWER: mean-pool is not a baseline on this task — it is the closed-form
solution.** The gap the exercise asks to measure has its sign fixed before the
experiment runs.

**FINDING: the MLP does learn the function, and learning it is the problem.**
A 7,519× improvement — the network discovering that the answer is its own input.

**MECHANISM: NSA's learned compressor exists for a task this synthetic does not
contain.** A real compressor preserves what the *router* needs, which is a
learned projection into the query's space, not the mean.

**FINDING: with the block's *maximum* as the target, mean-pool scores 5.60e+00**
and the MLP would have something to win.

### 3 — the gate cannot see which branch found anything

```text
needle in block 1   →  gates 0.719, 0.767, 0.733   picks [1, 7, 11, 12]
needle in block 15  →  gates 0.719, 0.767, 0.733   picks [7, 11, 12, 15]
```

**ANSWER: the picks move and the gates do not change in any decimal place.**

**MECHANISM: `gate(q, Wg)` takes the query and nothing else.** It never sees K,
V, the compressed scores, the picks, or the branch outputs. "The query hits a
far-back block" is a property of the *sequence*.

**FINDING: the three gates are not a distribution.** Three independent sigmoids
sum to 1.643 on average and range 0.313 to 2.968 — a **9.5×** swing in output
magnitude driven only by which query asked. A softmax is one line away.

**FINDING: "near-uniform on random queries" is a property of the
initialisation.** `Wg` is never trained, so each logit is `dot(q, w)` with both
zero-mean — the claim holds for *any* `Wg`, and cannot fail.

### 4 — the fine branch equals full attention at 1024

| at 128k context | keys | GB |
|---|---:|---:|
| full attention | 131,072 | **40.00** |
| NSA compressed (N/64) | 2,048 | 0.62 |
| NSA selected (k·l) | **1,024** | 0.31 |
| NSA window (W) | 512 | 0.16 |
| NSA total | 3,584 | **1.09** |
| MLA latent (512/layer) | — | 10.00 |

**ANSWER: the selected branch is `k·l` keys whatever N is**, so it equals full
attention at exactly **N = 1024** — and below that it reads more than the whole
sequence.

**FINDING: NSA is 9.2× smaller than MLA at 128k and 1.6× *larger* at 4k**,
crossing at N = 6,554. The exercise compares them at one context length.

**MECHANISM: NSA caches the whole sequence and reads part of it.** Stored, NSA
is **40.62 GB** — more than full attention. The saving is bandwidth, not
capacity, and the exercise asks for a memory budget.

**FINDING: only the compressed branch grows**, overtaking the two fixed terms at
N = 98,304. NSA's asymptote is `N/l` — a 64× smaller slope, not a flat one.

### 5 — the argmax has no gradient, so the score must be borrowed

```text
score[3]: 0.0000 ──────────────── 0.2000 ──────────────── 0.4000
picks:    [1, 2, 4] ············· ↑ jump ··············· [1, 2, 3]
          1 change point in 800 samples; a 1e-09 nudge anywhere else changes nothing
```

**ANSWER: the selection is a step function.** Its derivative is zero wherever it
is defined and undefined where it is not.

**MECHANISM: a separate routing score would receive no gradient at all.** A
routing head trained *through* the selection gets exactly 0 from every batch
except the measure-zero set where two blocks tie. Reusing the compressed
branch's scores trains the routing signal through that branch's own
differentiable output path.

**FINDING: the scores are used twice and differentiated once.** `cmp_w` goes to
`top_k_blocks`; `cmp_out` goes to the gated sum. The gradient reaches `cmp_w`
only through the second.

**FINDING: the same argument rules out training the gate through the
selection** — its gradient about `sel_out` is about the *contents* of blocks
already picked.
