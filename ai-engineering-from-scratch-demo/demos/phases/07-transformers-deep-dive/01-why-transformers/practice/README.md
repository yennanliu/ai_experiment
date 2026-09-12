<!-- generated:start -->
# 07-transformers-deep-dive / 01-why-transformers

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/07-transformers-deep-dive/01-why-transformers/) · upstream spec
`phases/07-transformers-deep-dive/01-why-transformers/docs/en.md`

```bash
uv run demo practice run 01-why-transformers --ex 1
uv run demo explain 01-why-transformers --ex 1
uv run pytest demos/phases/07-transformers-deep-dive/01-why-transformers
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Take `rnn_style` from `code/main.py` and replace the scalar hidden state with a length-… | code | T0 | `ex01_sixty_four_lanes_of_the_same_number.py` |
| 2 | Medium. Implement a parallel prefix-sum (Hillis-Steele scan) in pure Python. Verify it produc… | code | T0 | `ex02_the_equivalence_check_runs_on_the_safe_input.py` |
| 3 | Hard. Port the attention-style reduction to PyTorch on GPU. Time both as you sweep sequence l… | code | T1 | `ex03_the_sweep_ends_where_attention_stops_fitting.py` |
<!-- generated:end -->

## Answers

The lesson is five functions of pure stdlib and every exercise turns out to be a
question about the gap between what one of them is called and what it does. The
**Easy** exercise asks how the serial overhead grows with hidden dimension, and
the answer is that it does not grow at all — what grows is the work per step,
which is the half a GPU gets for free. The **Medium** one asks you to verify an
equality that is false. The **Hard** one sweeps to exactly the length where the
thing it is a proxy for stops fitting in memory, holding a proxy that cannot
notice.

Exercises 1 and 2 are **T0** — pure stdlib, as the lesson is. Exercise 3 is
**T1** because `torch` is absent and `numpy` stands in for the GPU arm.

### 1 — sixty-four lanes holding one number

`h = [decay * hi + x for hi in h]`, the sentence read literally, against the
lesson's own `rnn_style` on the same input at N = 20,000:

| d | wall-clock | ×scalar | per lane |
|---:|---:|---:|---:|
| 1 | 0.75 ms | 3.9× | 3.94 |
| 8 | 2.31 ms | 12.1× | 1.51 |
| 64 | **16.9 ms** | **85×** | **1.31** |

**ANSWER: roughly linearly, ≈1.3× per lane.** 64 lanes cost ~85× the scalar, not
64×: the comprehension adds a per-step allocation the scalar loop does not have.
Per lane that overhead *falls* from 3.9× at d=1 to 1.3× at d=64, so the total
grows slightly faster than `d` while the cost per lane improves — interpreter
overhead amortising over a wider step.

**FINDING: the serial overhead grows by exactly zero.** The lesson's own
`depth()` takes `n` and no `d`; `rnn_depth = n` at every hidden size. Step 2 says
in as many words that "depth, not op count, decides GPU time" — so the quantity
this exercise names is invariant under the change this exercise asks for. What
grows is work *per step*, which is the part a GPU parallelises for free. The
exercise measures the one dimension that does not matter to its own thesis.

**FINDING: the 64 lanes hold one distinct value.** Every lane starts at `0.0`
and receives the same `decay * hi + x`, so they stay equal forever:
`len(set(h)) == 1`, and the value is **bit-identical** to the scalar
`rnn_style`. 64× the work for zero extra information, because nothing in an
elementwise update mixes the lanes. It is not a 64-dimensional RNN; it is 64
copies of a 1-dimensional one.

**FINDING: the growth the question wants is quadratic.** A real recurrence is
`h_t = W h_{t-1} + x_t` — `d² = 4096` multiply-adds per step. Measured at
N = 2,000: **~11,000× the scalar**, about **120×** the elementwise reading. That
is the number the question is reaching for, and "a length-64 vector of hidden
states" cannot reach it.

### 2 — the equivalence check runs on the one safe input

The lesson already ships `parallel_scan`. Re-derived here from the recurrence, it
comes out bit-identical to the lesson's, so every number below is the lesson's
own:

| | serial | Hillis-Steele |
|---|---:|---:|
| depth at n=1024 | 1023 | **10** |
| additions | 1023 | **9217** (9.01×) |
| wall-clock at n=2²⁰ | 17 ms | **548 ms** (31×) |

**ANSWER: depth 10, bought with 9.0× the additions.** A pass at stride `s`
updates `n − s` positions, so the work is `Σ (n − s) = 9217`. Hillis-Steele is
not work-efficient, and 9.01× is exactly what the 102× depth reduction costs.

**FINDING: the outputs are not the same.** At length 1024 on the lesson's own
`benchmark()` data, only **23 of 1024** entries are bit-identical — worst gap
4.1e-14 absolute, **45 ULP**. Float addition is not associative and the two scans
bracket the same sum differently. "The same numerical output" is true to about
1e-14 and false as written.

**FINDING: `main()`'s check is run on the one input that cannot disagree.** Its
fixture is `[float(i) for i in range(16)]` — small exact integers, every partial
sum exactly representable. That same generator at length 1024 is still
**1024/1024** identical. The fixture is what passes the check, not the length;
swap in `benchmark()`'s own `0.001 * (i % 17)` and 1001 of 1024 entries move.

**FINDING: the 1e-9 tolerance is the wrong shape.** It is *absolute*, applied to
a prefix sum whose magnitude grows with `n`. At n = 1,048,576 — the top of
`benchmark()`'s own sweep, in the same file — the worst gap is **1.9e-8**, over
tolerance, while the relative error has not moved. A relative bound would have
held at every length in the lesson.

**CONTROL: it is 31× slower, not "the same wall-clock".** `parallel_scan`'s
docstring says "on a CPU it's the same wall-clock but the graph shape is what
matters". The shape claim is right; the wall-clock claim is off by 31×, because
log-depth costs 9× the adds *plus* a full `list(out)` copy per pass.

### 3 — the sweep ends where attention stops fitting

`torch` and `jax` both return `None` from `find_spec`, and the host is arm64 with
no CUDA device — "PyTorch on GPU" is unbuildable twice over. `numpy.mean` is the
substitute: one C-level reduction with no Python work per element, which is the
same dependency-graph argument as a GPU kernel at a smaller constant. Both arms
swept over exactly the range asked for, 2⁶ → 2¹⁶:

```text
          R
         R
        R
       R
      R

     R
    R     n
   R     n
        n
  R    n
nR   nn
 nnnn
67890123456   (x = log2 N, y = log10 s; R = rnn_style, n = numpy mean)
```

`R` is a straight diagonal across eleven doublings. `n` is flat for the first
four and then turns.

**ANSWER: a hockey stick, and the flat part is not parallelism.** The log-log
slope makes the shape a number:

| region | numpy mean | rnn_style |
|---|---:|---:|
| N = 64 … 1024 | **−0.03** | 0.93 |
| N = 8192 … 65536 | **+0.77** | 1.07 |
| whole sweep | **+0.32** | **1.01** |

16× the data for the same microsecond, then linear. The plateau is fixed
per-call dispatch cost, so the left half of the plot times *the call* and only
the right half times the reduction. A GPU has the identical shape with the knee
further right — a kernel launch is slower than a numpy dispatch — so reading the
flat part as "the GPU is parallel" is reading overhead.

**FINDING: nothing in the swept quantity is quadratic.** `attention_style` is
`sum(xs) / len(xs)`. A mean. `O(N)` time, `O(1)` extra memory, no N×N anything.
The doc says the gap widens "until you hit the `O(N²)` memory wall of attention"
— and at this sweep's own endpoint a real N×N attention matrix is
**65536² × 4 = 16 GiB per head per layer**, against **512 KiB** for the array
being swept. The exercise stops precisely where the wall it names begins.

**FINDING: the recurrent arm has no curve at all.** Slope **1.01** across the
same 1024× range — a ruler. Every bend in the plot belongs to the reduction's
plateau; recurrence contributes none of it, which means the picture the exercise
asks you to explain is a picture of numpy's calling convention.

**CONTROL: "beats it at length ≥ 1,000" is off by 16×.** Step 1 says the
attention-style reduction overtakes the RNN at N ≥ 1000. In the same pure Python
it already wins at **N = 64 by 3–4×**, and at all 11 lengths swept. The reason is
that `sum()` is implemented in C — which the doc does say — and that has nothing
to do with the dependency graph the lesson is selling.
