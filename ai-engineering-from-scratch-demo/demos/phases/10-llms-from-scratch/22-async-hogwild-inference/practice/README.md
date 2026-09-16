<!-- generated:start -->
# 10-llms-from-scratch / 22-async-hogwild-inference

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/10-llms-from-scratch/22-async-hogwild-inference/) · upstream spec
`phases/10-llms-from-scratch/22-async-hogwild-inference/docs/en.md`

```bash
uv run demo practice run 22-async-hogwild-inference --ex 1
uv run demo explain 22-async-hogwild-inference --ex 1
uv run pytest demos/phases/10-llms-from-scratch/22-async-hogwild-inference
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py` with the default settings. Confirm the N=2 Hogwild! configuration produces… | code | T0 | `ex01_progress_saturates_at_two_whatever_n_is.py` |
| 2 | Reduce the coordination heuristic's strength (set `coordination_weight=0.1`). Re-run. Show th… | code | T0 | `ex02_coordination_helps_the_baseline_too.py` |
| 3 | Compute the expected Hogwild! speedup for a 50k-token reasoning task with `p=0.8, c=500` and… | code | T0 | `ex03_the_chat_task_is_slower_at_every_n.py` |
| 4 | Read the Hogwild! paper's Section 4 (preliminary evaluation). Identify the two failure modes… | code | T0 | `ex04_perfect_coordination_emits_no_coordination_tokens.py` |
| 5 | Combine Hogwild! with speculative decoding in the toy: each worker uses a 2-token spec-decode… | code | T0 | `ex05_there_is_one_list_and_no_prefix_to_share.py` |
<!-- generated:end -->

## Answers

`code/main.py` simulates N workers appending to one shared cache, choosing
between two work categories under a coordination weight, plus an Amdahl-style
speedup formula. Every exercise lands on the same place: the simulator computes
redundancy carefully and then reports a headline number that cannot see it.

All five are **T0** on **no** dependency group (stdlib only).

### 1 — progress saturates at two, whatever N is

| N | tokens | work tokens | unique progress | progress/step | useful |
|---:|---:|---:|---:|---:|---:|
| 1 | 200 | 193 | 193 | 0.96 | 100% |
| 2 | 400 | 385 | 385 | 1.93 | 100% |
| 4 | 800 | 761 | **400** | **2.00** | 53% |
| 8 | 1600 | 1511 | **400** | **2.00** | **26%** |

**ANSWER: confirmed at N=2 — and unique progress is identical at N=4 and N=8.**

**MECHANISM: there are two work categories**, so at most two tokens per step can
be unique. At N=8, six of every eight are redundant by construction.

**FINDING: the metric the exercise names is the one that cannot saturate.** Work
tokens grow 7.83×; unique progress grows 2.07×.

**FINDING: the useful fraction falls to 26%** — the simulator reports exactly the
phenomenon Hogwild! exists to avoid, on a metric that does not see it.

### 2 — coordination helps the baseline too

| weight | N=1 progress | N=2 | N=4 | N=8 |
|---:|---:|---:|---:|---:|
| 1.0 | 193 | **1.99×** | 2.07× | 2.07× |
| 0.5 | 181 | 1.61× | 1.97× | **2.18×** |
| 0.1 | 172 | **1.27×** | 1.53× | 1.81× |
| 0.0 | 171 | 1.15× | 1.17× | 1.17× |

**ANSWER: 1.99× collapses to 1.27× at weight 0.1.**

**MECHANISM: at weight 0 every worker stays on `A`** — unique progress is 197,
200, 200 whatever N is, against 1,352 work tokens at N=8.

**FINDING: part of the collapse is the baseline improving** — one worker gains
13% from a heuristic meant to avoid workers that are not there, because at
weight 1.0 it never spends a step on a `coord` token.

**FINDING: at weight 0.5, N=8 beats perfect coordination** (2.18× vs 2.07×),
because the denominator moved.

### 3 — the chat task is slower at every N

| task | N=2 | N=4 | N=8 | best N | Amdahl ceiling |
|---|---:|---:|---:|---:|---:|
| 50k, p=0.8, c=500 | 1.613× | **2.273×** | 2.632× | 9 | 5.00× |
| 1k, p=0.3, c=200 | 0.800× | **0.635×** | 0.428× | **1** | 1.43× |

**ANSWER: 2.273× and 0.635×** — and at N=1 the formula still charges `c`,
reporting 0.833× for a run with nobody to coordinate with.

**MECHANISM: the chat task's `c·N` at N=4 is 800 against a 1,000-unit job** —
80% before any work is done.

**FINDING: neither `p` nor `c` alone decides it.** Swap `T` and the verdicts
swap: 1.23× and 0.83×.

### 4 — perfect coordination emits no coordination tokens

| | weight 0.0 | weight 1.0 |
|---|---:|---:|
| redundancy at N=2 | 146 of 343 | **0 of 385** |
| redundancy at N=8 | 1,152 of 1,352 | **1,111 of 1,511** |
| coordination tokens at N=8 | 163 | **0** |

**ANSWER: two failure modes, and only one is a coordination problem.** At N=2 a
better prompt removes every redundant token; at N=8 the two-category ceiling has
taken over and it removes almost none.

**FINDING: coordinating costs nothing here, which inverts the trade.**
`decide_next_category` reaches its `coord` branch only when the coordination
branch did *not* fire — so the better a worker coordinates, the less it spends
saying so.

**FINDING: noise is the only cost the weight cannot touch** (5.6%).

**MECHANISM: `target_per_category` never appears in the function's body.** A
prompt saying "stop at N of each" has nowhere to land.

### 5 — there is one list and no prefix to share

| N | Hogwild! | × spec (K=2) | combined |
|---:|---:|---:|---:|
| 2 | 1.99× | 2.26× | 4.51× |
| 4 | 2.07× | 2.26× | **4.68×** |
| 8 | 2.07× | 2.26× | 4.68× |

**ANSWER: 4.68×, and only the first factor has a ceiling.** The speculative
factor `(1 + a + a²) / (1 + 2c)` does not depend on the worker count.

**MECHANISM: the two act on different quantities** — tokens per *step* against
tokens per *verifier call*, and `run_hogwild` has no notion of a call.

**ANSWER to the bookkeeping question:** `SharedCache` is one field holding
`(worker_id, category)` pairs. Two workers extending the same prefix would need
positions, branches and per-worker lengths; a rejected speculation would have to
truncate its own tokens out of a list others have appended to, and no index says
which those are.

**FINDING: the simulator already has the race and does not model it as one.**
`decide_next_category` reads `cache.counts()` *before* the step's appends, so no
worker sees the others' current tokens — the Hogwild! premise written as a loop.
