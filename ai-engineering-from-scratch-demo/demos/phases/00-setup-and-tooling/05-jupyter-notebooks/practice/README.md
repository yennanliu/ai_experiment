<!-- generated:start -->
# 00-setup-and-tooling / 05-jupyter-notebooks

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/00-setup-and-tooling/05-jupyter-notebooks/) · upstream spec
`phases/00-setup-and-tooling/05-jupyter-notebooks/docs/en.md`

```bash
uv run demo practice run 05-jupyter-notebooks --ex 1
uv run demo explain 05-jupyter-notebooks --ex 1
uv run pytest demos/phases/00-setup-and-tooling/05-jupyter-notebooks
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Open JupyterLab, create a notebook, and use `%timeit` to compare list comprehension vs numpy… | code | T1 | `ex01_the_lessons_own_timer_squares_a_range_and_draws_no_random_number.py` |
| 2 | Create a notebook with both markdown and code cells that loads a CSV, displays a dataframe, a… | code | T0 | `ex02_a_deleted_cell_passes_every_click_fails_run_all_and_its_gap_is_already_in_the_json.py` |
| 3 | Take the code from `code/notebook_tips.py`, paste it into a Colab notebook, and run it with a… | code | T0 | `ex03_a_free_gpu_changes_nothing_because_the_code_never_touches_one.py` |
<!-- generated:end -->

## Answers

The lesson's code is one script, `notebook_tips.py`, that imports numpy,
matplotlib and pandas at module top. This repo's `math` group has numpy only,
so exercises 1 and 3 import the lesson through `parity` with matplotlib and
pandas stubbed **only when absent** and removed from `sys.modules` straight
after, then call only its numpy functions. Jupyter itself is not a dependency:
a `.ipynb` is nbformat JSON and a kernel is a namespace, so exercise 2 builds
both with the stdlib. Exercise 1 measures wall-clock time and is **T1**; 2 and 3
are **T0**.

### 1 — the lesson's own timer squares a range and draws no random number

`%timeit` is the stdlib `timeit` with `autorange` and repeats, so it runs here
without a kernel. n = 100,000, best of 5:

| route | time (this machine) |
|---|---:|
| `[random.random() for _ in range(n)]` | ~2.4–2.9 ms |
| `np.random.rand(n)` | ~0.18–0.25 ms |
| list, then `np.array(...)` | ~3.7–4.7 ms |

**ANSWER: numpy is ~11–14x faster.** The ratio is host-dependent; the check
asks only for 2x.

**FINDING: the lesson's `timing_comparison` answers a different question.** It
squares `range(1_000_000)` against `np.arange(size) ** 2` — no random draw, ten
times the n, one `perf_counter` shot per side. It printed **13.0–13.6x**, while
`timeit` on that same squaring gives **~22x**.

**FINDING: the two sides do not build the same object.** 100,000 boxed floats
take **3.20 MB**, exactly **4.00x** the array's **0.80 MB**; if the next cell
needs an array, the list route also pays the conversion.

### 2 — a deleted cell passes every click, fails Run All, and its gap is already in the JSON

The CSV is the lesson's own Step 5 table, parsed from `docs/en.md`; the chart is
an SVG `display_data` output.

**ANSWER:** the clean notebook (2 markdown, 4 code cells) runs top to bottom
with zero errors, counts **[1, 2, 3, 4]**, 3 rows, 3 bars.

**FINDING: the lesson's hidden-state trap, measured.** Run the cells, delete the
helper that builds `accuracy`, save: every output survives and the file records
no error, yet Restart & Run All stops with **NameError: name 'accuracy' is not
defined**.

**FINDING: you do not need a kernel to see it.** The saved counts read
**[1, 3, 4]** — count 2 ran and is gone. A scan for counts that are not exactly
1..k flags that and an out-of-order notebook (**[1, 2, 4, 3]**) and passes the
clean one.

### 3 — a free GPU changes nothing, because the code never touches one

**ANSWER:** the source references none of `torch`, `cuda`, `cupy`, `jax`,
`tensorflow`, `device`; all compute is numpy on the CPU, so a T4 runtime prints
the same numbers.

**FINDING: pasted into a cell, the code never asks for an inline plot.** It
calls `matplotlib.use("Agg")` at import, and `inline_plotting` calls `savefig`,
never `show`, while printing that `plt.show()` displays the plot inline.

**FINDING: the "Python process memory" line measures the array.** It is
`sys.getsizeof(large)`: **80,000,112** bytes against `nbytes` 80,000,000, and
**112 bytes** for a view of the same 80 MB — the wrong tool for the memory leaks
the lesson warns about.
