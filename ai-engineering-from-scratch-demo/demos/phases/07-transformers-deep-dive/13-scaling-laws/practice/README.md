<!-- generated:start -->
# 07-transformers-deep-dive / 13-scaling-laws

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/07-transformers-deep-dive/13-scaling-laws/) · upstream spec
`phases/07-transformers-deep-dive/13-scaling-laws/docs/en.md`

```bash
uv run demo practice run 13-scaling-laws --ex 1
uv run demo explain 13-scaling-laws --ex 1
uv run pytest demos/phases/07-transformers-deep-dive/13-scaling-laws
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py`. Print Chinchilla-optimal `(N, D)` for compute budgets `1e20`, `1e22… | code | T0 | `ex01_the_headline_ratio_is_never_twenty_here.py` |
| 2 | Medium. Implement the Hoffmann loss-as-function-of-compute curve. Plot loss vs `log10(C)` for… | code | T0 | `ex02_the_next_tenth_stops_being_expensive_and_becomes_impossible.py` |
| 3 | Hard. Fit your own scaling law on 5 tiny models (100K to 10M params) trained on the same data… | code | T0 | `ex03_one_dataset_cannot_separate_e_from_b_over_d.py` |
<!-- generated:end -->

## Answers

The lesson is 94 lines: the Chinchilla loss equation, a grid search for the
compute-optimal `(N, D)`, and a table of real models. Every exercise turns into a
question about whether the constants printed at the top agree with the sentences
printed at the bottom. All three are **T0**.

### 1 — with these constants D/N is never 20 in the range that matters

The optimum has a closed form — `N* = (Aα / Bβ)^(1/(α+β)) · (C/6)^(β/(α+β))` —
so the grid can be checked rather than trusted:

| C | grid N* | exact N* | error | grid D/N | exact D/N | loss |
|---:|---:|---:|---:|---:|---:|---:|
| 1e20 | 6.59e8 | 6.45e8 | 2.2% | 38.3 | 40.1 | 2.5998 |
| 1e22 | 5.05e9 | 5.16e9 | 2.1% | 65.3 | 62.6 | 2.1386 |
| 1e24 | 4.25e10 | 4.13e10 | 2.9% | 92.4 | 97.7 | 1.9112 |

**ANSWER: the grid is accurate to 3% in N and 2e-5 in loss.** 200 points over 8
decades is a **9.7%** step in `N`; the losses agree anyway because the objective
is flat at its minimum. That is why a coarse grid answers the loss question and
cannot answer the ratio question.

**FINDING: `D/N = 20` happens at `C = 7.6e16`.** The ratio scales as `C^0.0968`
exactly, so it passes 20 **five to six orders of magnitude below** the 1e22–1e23
the lesson says Chinchilla studied. `main()` prints "Hoffmann 2022 published
D/N ≈ 20 as the headline" directly beneath a table where its own constants give
40.1, 62.6 and 97.7.

**FINDING: half the real table is *under*-trained by this law.** At 1e24 the law
wants D/N = 98. GPT-3 sits at 1.7 and Chinchilla at 20.0 — both below the
optimum — while Llama 3 8B is at 1,875 and DeepSeek-V3 (active) at 400.

### 2 — the next tenth stops being expensive and becomes impossible

Substituting the optimum back in gives `L*(C) = E + k·C^(−αβ/(α+β))`, an exponent
of **0.15355**. The closed form agrees with the lesson's own `compute_optimal` to
2e-5.

| from C | L*(C) | C needed for L − 0.1 | multiple |
|---:|---:|---:|---:|
| 1e22 | 2.139 | 5.2e22 | 5× |
| 1e23 | 2.005 | 1.2e24 | 12× |
| 1e24 | 1.911 | 5.0e25 | 50× |
| 1e25 | 1.845 | 8.3e27 | **832×** |
| 1e26 | 1.799 | 1.1e33 | **1.1e7×** |

**ANSWER: from `C = 1.07e25` the next 0.1 first costs more than 1e28 FLOPs.**

**FINDING: shortly after, the answer stops being a number.** `E = 1.69` is a
floor the law never crosses, so once `L* < E + 0.1` **no finite compute** buys
another tenth. The frontier reaches 1.79 at **C = 1.76e26** — about 16× past the
point the exercise asks about. Between the two, the honest answer changes from
"1e28 FLOPs" to "there is no such budget".

**FINDING: the curve is not slowing down; the distance to the floor is
shrinking.** Near `L − E = 0.5` a tenth costs 4×; near `L − E = 0.15` it costs
1,280×. Same exponent, different remaining headroom.

### 3 — five models on one dataset cannot separate E from B/D^β

The published law is used as ground truth and the exercise's *design* is run
against it: 5 sizes from 1e5 to 1e7, `D` fixed. Every loss is exactly right by
construction, so anything the fit gets wrong belongs to the design.

**ANSWER: α comes back exactly and E is wrong by 73%** — `α = 0.3400` against
0.34, `A = 406.6` against 406.4, residual 1.4e-9, and `E = 2.9307` against
**1.69**.

**FINDING: the error is exactly `B / D^β`.** With `D` fixed at 1e9 the data term
is `410.7 / (1e9)^0.28 = 1.2403`, a constant over the whole design, so least
squares folds it into the intercept: `1.69 + 1.240 = 2.930`, matching the fitted
value to four digits. **"Trained on the same dataset" is the phrase that makes E
unidentifiable** — a constant cannot be separated from a constant by better
fitting. More sizes, less noise and a perfect optimiser all leave the same 1.24
offset; only varying `D` removes it.

**FINDING: α survives noise, barely.**

| loss noise | α range (200 refits) | median |
|---|---|---:|
| 0.2% | 0.33 – 0.35 | 0.340 |
| 0.5% | 0.31 – 0.37 | 0.339 |
| 1.0% | **0.28 – 0.40** | 0.335 |

Matching a published exponent to two digits needs the five losses measured to
better than a percent.

**CONTROL: the design never gets near the floor.** Loss runs 11.039 at N=1e5 to
4.624 at N=1e7 — still **2.93 above** the true E. A parameter that only reveals
itself where the curve flattens cannot be read off a stretch that has not begun
to flatten.
