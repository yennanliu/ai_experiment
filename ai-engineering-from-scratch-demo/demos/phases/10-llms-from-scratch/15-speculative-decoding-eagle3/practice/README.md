<!-- generated:start -->
# 10-llms-from-scratch / 15-speculative-decoding-eagle3

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/10-llms-from-scratch/15-speculative-decoding-eagle3/) · upstream spec
`phases/10-llms-from-scratch/15-speculative-decoding-eagle3/docs/en.md`

```bash
uv run demo practice run 15-speculative-decoding-eagle3 --ex 1
uv run demo explain 15-speculative-decoding-eagle3 --ex 1
uv run pytest demos/phases/10-llms-from-scratch/15-speculative-decoding-eagle3
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Confirm the chi-square statistic on the Leviathan distribution check stay… | code | T0 | `ex01_the_check_fails_on_a_third_of_seeds.py` |
| 2 | Sweep `N` from 1 to 10 with `α` held at 0.9 and `c` held at 0.04. Plot expected tokens per ve… | code | T0 | `ex02_the_minimum_is_outside_the_sweep.py` |
| 3 | Modify the code to simulate EAGLE-2 tree search: at each step, the draft proposes a tree of s… | code | T0 | `ex03_the_tree_breaks_what_exercise_one_verified.py` |
| 4 | Implement a batched KV rollback simulator for two concurrent sequences. Sequence A has all dr… | code | T0 | `ex04_truncate_to_never_truncates.py` |
| 5 | Read the EAGLE-3 paper's Section 4 (Training-Time Test). Explain in two sentences why naive d… | code | T0 | `ex05_teacher_forced_alpha_over_predicts_by_28_percent.py` |
<!-- generated:end -->

## Answers

`code/main.py` is a correct implementation of Leviathan rejection sampling: the
accept rule, the residual distribution, a KV-length tracker, a chi-square
equivalence check, and the throughput algebra. It is the most rigorous reference
in the phase, and four of the five exercises find that the *questions* around it
are looser than the code is.

All five are **T0** on **no** dependency group (stdlib only).

> Exercise 5 is scaffolded as a prose item ("read the paper, explain in two
> sentences"). It is built here as a **measurement** instead, because the
> assumption EAGLE-3's Section 4 is about is written into `spec_step`'s own
> docstring, and a measured answer is checkable where a paraphrase is not.

### 1 — the check fails on a third of seeds

| scored against | mean χ² | above 14.07 |
|---|---:|---:|
| a second random sample (as the lesson does) | **12.34** | **29%** |
| the exact `q` | 7.01 | 4% |

**ANSWER: the check passes on seed 42 (11.60) and fails on 7 of 24 seeds.** A
chi-square with 7 degrees of freedom should average 7.0.

**MECHANISM: the test compares two random samples and calls one the
expectation.** `distribution_check` draws 50,000 speculative tokens *and* 50,000
direct ones; `chi_square` rescales the second and treats it as `E`. Both sides
carry noise, roughly doubling the variance, so a nominal 5% test rejects 29% of
the time.

**FINDING: scored against `q`, the same runs behave** — mean 7.01, 4% above the
threshold. `q` is a list of eight floats at the top of `main`.

**FINDING: Leviathan's guarantee really is exact.** With a draft at KL 0.0198
from the verifier, the worst of 24 seeds is 18.37.

### 2 — the minimum is outside the sweep

```text
N=1  0.5474   N=4  0.2833   N=7  0.2247   N=10 0.2040   ← sweep ends here
N=12 0.1984   N=14 0.1964   N=15 0.1964 ← minimum      N=20 0.2021   N=30 0.2287
```

**ANSWER: wall time falls at every step from 1 to 10**, so "the N that minimizes
wall time" over that range is 10 — the last point tried. The minimum is
**0.1964 at N=15**.

**MECHANISM: cost is `1 + Nc`, linear forever; tokens saturate at `1/(1−α) =
10`.** At N=10 the chain has reached 6.86 and is gaining 0.35 tokens per extra
draft against 0.04 of extra cost. The curve turns where the marginal token stops
paying for the marginal draft.

**FINDING: the ceiling ends the curve, not the cost.** At N=30 the chain
averages 9.62 — 96% of the ceiling — and wall time is back to 0.2287.

**FINDING: the shape is flat where the answer is.** N = 13–17 are within 1% of
each other; the 1-to-10 window is entirely slope, falling 63%.

### 3 — the tree breaks what exercise 1 verified

| branches | per-node α | χ² vs `q` | tokens/call | drafted nodes |
|---:|---:|---:|---:|---:|
| 1 | 0.933 | **3.3** | 3.62 | 3 |
| 2 | 0.996 | **8,790.5** | 3.97 | 14 |
| 4 | 1.000 | **38,669.6** | 4.00 | 84 |

**ANSWER: two branches per node improves acceptance and destroys correctness** —
600× the critical value Exercise 1 confirms the single-draft scheme stays under.

**MECHANISM: "the highest-probability accepted path wins" is a selection, and
selection is a bias.** Keeping the survivor with the largest `q/p` re-weights the
output by the very ratio that was supposed to cancel.

**FINDING: acceptance per node is the wrong thing to maximise.** 4.00 tokens
against 3.62 — a 10% gain for 84 drafted nodes against 3, and the tokens are no
longer samples from the verifier. The metric the exercise asks for cannot see it.

**FINDING: the fix is known and is not a tuning change.** SpecInfer and EAGLE-2
renormalise the residual over the candidates already rejected *at each node* —
an accept rule, not a selection over leaves.

### 4 — `truncate_to` never truncates

```text
prefix 10 ──> A: all accepted   emits 6   kv 16   truncate_to called 0×
          └─> B: rejects at 2   emits 3   kv 13   truncate_to called 1×, and it GREW the buffer
```

**ANSWER: the per-sequence lengths are correct, and nothing ever shrinks.**
`spec_step` calls `extend(1)` per accepted token and then
`truncate_to(prefix + len(emitted))` — one *more* than the current length,
because `emitted` already holds the correction token.

**FINDING: "no work is wasted" is false, and the amount is the point.** B drafted
5 and used 2. Three draft forwards are discarded — which is what speculative
decoding pays for the verifier call it saves.

**MECHANISM: a batched rollback has nothing to synchronise.** `KVBuffer`'s state
is one scalar; the "batched" simulator is two independent simulators.

### 5 — teacher-forced α over-predicts by 28%

| position | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| teacher-forced α | 0.871 | 0.849 | 0.848 | 0.849 | 0.848 | **0.849** |
| free-running α | 0.869 | 0.755 | 0.683 | 0.630 | 0.595 | **0.568** |
| P(draft is in the bad state) | 0.000 | 0.162 | 0.275 | 0.347 | 0.400 | **0.435** |

**ANSWER: that table is exposure bias.** The draft is *measured* on prefixes the
verifier produced and *deployed* on prefixes it produced itself, and the two
drift apart with every token it emits.

**MECHANISM: the draft walks into the states where it is worst.** It
over-produces the one token the two models disagree on and, once there, its own
transition keeps it there. Feeding the draft its own predictions during training
puts those prefixes into the training distribution — scheduled sampling's answer
to the same problem in seq2seq.

**FINDING: `measure_alpha` is itself a teacher-forced measurement.** One token,
no chain, no conditioning → 0.871. Fed to `expected_tokens_per_verify` it
predicts **4.52** tokens per call against the **3.52** the chain delivers.

**FINDING: the formula assumes α is *constant*, not merely known.** Even given
the correct mean of the free-running αs it predicts 2.94 against 3.52 — a
decaying sequence and a constant one with the same mean are not the same product.
So `spec_step`'s aside about extending to position-dependent `q_i, p_i` "without
changing the loop" is right about the loop and wrong about the throughput model
built on it.
