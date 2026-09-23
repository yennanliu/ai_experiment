<!-- generated:start -->
# 15-autonomous-systems / 03-alphaevolve-evolutionary-coding

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/15-autonomous-systems/03-alphaevolve-evolutionary-coding/) · upstream spec
`phases/15-autonomous-systems/03-alphaevolve-evolutionary-coding/docs/en.md`

```bash
uv run demo practice run 03-alphaevolve-evolutionary-coding --ex 1
uv run demo explain 03-alphaevolve-evolutionary-coding --ex 1
uv run pytest demos/phases/15-autonomous-systems/03-alphaevolve-evolutionary-coding
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Note the best score trajectory. Disable the held-out evaluator (flag `--n… | code | T0 | `ex01_the_flag_keeps_the_evaluator_and_drops_it_from_the_signal.py` |
| 2 | Read Section 3 of the AlphaEvolve paper on the MAP-elites grid. Design a feature-vector descr… | code | T0 | `ex02_the_grid_axis_is_a_quality_ladder.py` |
| 3 | The 48-multiplication 4x4 result improved on Strassen's 49-mul bound after 56 years. Read App… | explain | T0 | prose, below |
| 4 | Propose one domain where AlphaEvolve would fail. Identify exactly where the evaluator breaks… | code | T0 | `ex04_a_measured_evaluator_breaks_at_the_acceptance_test.py` |
| 5 | For a domain you know, write the evaluator signature you would use. Include (a) correctness c… | code | T0 | `ex05_the_anti_hacking_clause_is_declared_and_unenforced.py` |
<!-- generated:end -->

## Answers

### 1 — the flag keeps the evaluator and drops it from the signal

At the shipped seed the two runs look exactly as the lesson promises. With the
held-out signal the loop rediscovers the target exactly — train and test MSE
**0.0000** at generation **1229**. With `--no-holdout` it lands on train
**2.2917**, test **4.3906**, a **+2.0990** gap. The train trajectory runs
13.83 → 3.17 → 0.00 across generations 100/500/1500 with the signal and
22.50 → 8.00 → 2.29 without it.

**The flag does not disable the held-out evaluator.** `run_loop` computes
`te = mse(child_expr, test_xs)` for every child whichever way the flag is set;
only `signal_of` changes. The evaluator the exercise says to disable still runs
1500 times per loop and its output is still printed. It is excluded from
selection, not from execution — which matters, because the cost story of a
held-out evaluator is the evaluation, not the comparison.

**One seed is not a measurement.** The two runs consume the same random stream
and differ only in which candidates enter the archive, so they can be paired.
Over seeds 0–19 the held-out signal produces a better final test MSE **6**
times, a worse one **4** times, and an identical one **10**:

| | mean final test MSE | wins | losses | ties |
|---|---:|---:|---:|---:|
| held-out signal | 1.90 | 6 | — | — |
| train only | 3.58 | — | 4 | 10 |

The means separate by a factor of 1.9, and ten decided pairs splitting 6–4 is
what a coin does. The lesson's headline is true of the seed `main()` passes
and is not established by it.

**And the split measures interpolation.** The training points are the integers
−2…3 and the test points are their midpoints, so five of the seven held-out
points sit *inside* the training hull and the other two half a unit outside.
Symbolic regression fails by extrapolating; this evaluator is built where it
cannot see that.

### 2 — the grid axis is a quality ladder

**The descriptor, for compiler optimization passes:** two compositional ratio
axes — the fraction of the sequence that transforms rather than analyses, and
the fraction that operates on loops rather than straight-line code — five
buckets each. Both are ratios, so they are bounded without a cap; neither is a
size proxy, so neither grows monotonically as the search runs.

Ported to this lesson's grammar as operator mix and leaf mix, and used to key
the same archive, the difference is visible in where each axis puts its best
row:

| axis | median elite error by row | best row |
|---|---|---:|
| `cell_key` depth | 99.6, 57.3, 21.0, 22.3, 14.2, 6.7 | 5 of 6 — the capped end |
| operator mix | 28.7, 6.3, 6.5, 25.1, 42.4 | 1 of 5 — interior |
| leaf mix | 72.9, 26.2, 7.3, 6.6, 19.7 | 3 of 5 — interior |

That is the test worth applying to any MAP-elites descriptor: **an axis whose
best row is at an end is ranking; an axis whose best row is in the interior is
describing.** The target expression has a particular shape, not a maximal one,
so a descriptor that captures shape has its optimum in the middle. Depth does
not, and the diversity the shipped grid preserves is largely diversity in how
bad a candidate is allowed to be.

The archive this is read off is a replay — `run_loop` builds a grid and returns
only the winner — so before anything is measured it is checked against
`run_loop`'s own returned best, which it reproduces in 10 of 10 seeds.

**Both grids saturate.** `cell_key` caps at 30 cells and a 1500-generation run
fills 28.4 on average, all 30 in 5 of 10 seeds; the proposed descriptor's 25
cells fill 24.3. Past saturation MAP-elites is not a diversity mechanism, it is
N parallel hill climbs — and the fix for that is more cells, not better axes.
Honest scope for the proposal: it improves the *shape* of the search, not its
breadth.

**The capped axis is also nearly one-way.** Of `mutate`'s four branches, two
grow depth by wrapping the parent, one leaves it and one resets to a leaf. So
depth drifts up into the capped row, which is the single cell where
arbitrarily different programs compete as if they were neighbours.

### 3 — why this evaluator is easy and most are not

*Draws on "What makes the evaluator non-negotiable".*

The 4×4 matrix-multiplication evaluator is easy to get right because
correctness is a finite identity: multiply two matrices with the candidate
scheme, compare against the definition, and the answer is bit-exact — there is
no sampling, no distribution, no held-out set to design, and the metric
(scalar multiplications used) is a property of the program text rather than of
its execution. That combination means the evaluator cannot be gamed and cannot
be noisy: a candidate either computes the product for *every* input or it fails
on one, and the count of multiplications is read off, not measured. Most
domains have neither half — the correctness condition is a sample of inputs
rather than a proof, so a candidate can pass by overfitting the sample, and the
metric is a measurement on shared hardware, so repeating it returns a different
number and the loop can optimise the noise instead of the program.

### 4 — a measured evaluator breaks at the acceptance test

**The domain: kernel and query latency optimisation benchmarked on shared
hardware** — one AlphaEvolve is actually pointed at, and one where the
evaluator is a measurement rather than a proof.

**Exactly where it breaks:** `signal_of(child) < signal_of(incumbent)`. That
comparison treats a single measurement as the truth. Replaying the archive with
the score perturbed by σ = 1.0:

| | cells | cells held by a displaced program | median regret | worst |
|---|---:|---:|---:|---:|
| exact evaluator | 28.4 | 0 | 0.00 | 0.00 |
| measured, σ = 1.0 | 29.3 | 5 | 0.33 | 3.46 |

Nothing in the loop is wrong about arithmetic. Five cells out of twenty-nine
are simply held by a program that is not the best one that entered them.

**The loop never re-measures.** `run_loop` calls `mse` twice per child, both on
the child, and zero times on an incumbent. A score is frozen into the
`Candidate` at first sight, so a lucky draw is not corrected by the thousand
generations that follow — it is defended by them, because the cell it holds is
the cell every challenger must beat.

**The data model cannot express a measurement.** `Candidate` has four fields,
two of which are scores, and none of which is a sample count, a variance or a
confidence interval. There is no way to write "0.42 ± 0.15 over 5 runs" into
this archive, so there is no way for the acceptance test to demand a margin
before it swaps. That is the actual repair — re-evaluate the incumbent and
require a margin — and it needs a field before it needs a policy.

**And the final selection inherits the frozen numbers.** `min(archive.values(),
key=signal_of)` re-reads what was stored, so at σ = 1.0 the program the run
returns is worse than the best program in its own archive in 2 of 10 seeds, by
up to 1.00. With an exact evaluator, never.

### 5 — the anti-hacking clause is declared and unenforced

**The domain:** this repository's own solution gate, which scores a candidate
file the way `mse` scores a candidate expression. The signature:

```python
def evaluate(path: Path, exercise: Exercise, upstream: str) -> Verdict: ...
```

| clause | where it lives here |
|---|---|
| (a) correctness conditions | `practice.grade_file` — every `Check` in `verify()` must pass |
| (b) performance metric | `audit_practice` — complexity ≤ 8, and a line count whose enforced number is 150 with 120 reported as a target |
| (c) held-out input generation rule | `tests/test_practice.py` — the exercise text is re-read from upstream at scoring time and hashed, so a stale spec fails |
| (d) anti-reward-hacking check | the solution must name a symbol from the lesson's own `code/` — **does not exist** |

**Clause (d) is declared and never enforced.** `uses_reference` is a manifest
field: `harness/manifest.py` names it four times, and it appears zero times in
`audit_practice.py` and zero times in `check_deps.py`. A solution that forked
the lesson's implementation and asserted against its own copy would pass every
gate in this repository — which is precisely the failure `DESIGN D5` exists to
prevent.

**The lesson's own evaluator has one of the four.** `mse` is a performance
metric. Its correctness handling is a single `except` returning `inf` — an
error guard, not a condition a candidate must satisfy. Its held-out inputs are
seven literal floats, not a generation rule. And it has no anti-hacking clause
at all, which is exactly why the 48-multiplication result is checkable and most
domains are not.

**The clauses are cheap.** Applied statically to the four code files in this
directory, four define `PRACTICE_IMPL`, four load the reference rather than
copying it, and four sit inside both ceilings. That is what clause (d) would
look like if the gate ran it, and it is one function.
