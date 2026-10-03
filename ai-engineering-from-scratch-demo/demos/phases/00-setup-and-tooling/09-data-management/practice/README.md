<!-- generated:start -->
# 00-setup-and-tooling / 09-data-management

Solutions to all 4 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/00-setup-and-tooling/09-data-management/) · upstream spec
`phases/00-setup-and-tooling/09-data-management/docs/en.md`

```bash
uv run demo practice run 09-data-management --ex 1
uv run demo explain 09-data-management --ex 1
uv run pytest demos/phases/00-setup-and-tooling/09-data-management
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Load the `glue` dataset with the `mrpc` config and inspect the first 5 examples | code | T0 | `ex01_load_and_inspect_shows_one_row_and_breaks_without_a_split.py` |
| 2 | Stream the `c4` dataset and count how many examples you can process in 10 seconds | code | T0 | `ex02_stream_dataset_keeps_every_row_it_counts.py` |
| 3 | Convert a dataset to Parquet and compare the file size to CSV | code | T0 | `ex03_parquet_win_is_compression_and_tiny_files_lose.py` |
| 4 | Create a 70/15/15 train/val/test split with a fixed seed and verify the sizes | code | T0 | `ex04_70_15_15_never_comes_out_exact.py` |
<!-- generated:end -->
## Answers

The lesson's code is one Python module, `data_utils.py`, that wraps Hugging Face
`datasets`. CI has no network, no `datasets` and no `pyarrow`, and the module
calls `sys.exit(1)` at import when `datasets` is missing — so every solution
imports it with `datasets` and `huggingface_hub` stubbed, and runs the lesson's
own helpers on labelled synthetic tables. All four are **T0**; exercise 4 uses
scikit-learn's `train_test_split` for the split sizes, the rule HF copies.

### 1 — `load_and_inspect` shows one row; with no split it counts splits

**ANSWER:** `load_and_inspect("glue", "mrpc")` calls
`load_dataset(path="glue", name="mrpc", split="train")` and prints
`Rows: 3668`; the first five examples are `ds.select(range(5))`.

**FINDING: the helper shows 1 example, not 5** — it prints `ds[0]` only.

**FINDING: `split=None` prints `Rows: 3`, then raises.** The guard is
`if split:`, so a `DatasetDict` comes back, `len()` counts its three splits
(of 5801 rows), and `.features` raises AttributeError before any row.

**FINDING: without `datasets` the import is `SystemExit(1)`**, not an
`ImportError` a caller could catch.

### 2 — `stream_dataset` keeps every row it reads

**ANSWER: 2,500 examples in 10 s at 4 ms each**, counted by a loop that stops
on an injected clock — the real number is your link's. The helper takes
`max_rows`, not a time budget.

| rows streamed | memory retained by `stream_dataset` | by a counting loop |
|---|---:|---:|
| 200 | 0.46 MB | 0 B |
| 2000 | 4.64 MB | 0 B |

**FINDING:** the doc says streaming memory "stays constant regardless of
dataset size"; the helper returns a list, so it grows **10.0x** with the count.

**FINDING: `max_rows=0` and `max_rows=-5` each return one row** — the break
test runs after the append. **CONTROL:** `max_rows=5` pulls exactly 5.

### 3 — the size win is compression, not columns

The lesson's `convert_format` writes CSV, JSON Lines and a Parquet stand-in
(columns stored contiguously, gzip-compressed; no Parquet encodings or footer).

| table | CSV / stand-in | CSV / gzip(CSV) | JSON / CSV |
|---|---:|---:|---:|
| 500 free-text reviews | **2.4x** | 2.4x | 1.16x |
| 500 low-cardinality rows | **6.9x** | 6.6x | 4.0x |
| 3 low-cardinality rows | **0.6x** | — | — |

**FINDING:** gzip of the row-major CSV is within **1.01x** (reviews) and
**1.05x** (codes) of the columnar file: without Parquet's encodings,
"Parquet: Small" is "compressed".

**FINDING: on 3 rows the lesson prints "Parquet is 0.6x smaller than CSV"** —
75 B became 118 B, and the fixed wording calls the growth "smaller".

### 4 — 70/15/15 never comes out exact

**ANSWER: 699 / 150 / 151 at n = 1000**, disjoint, complete and reproducible.

**FINDING: every n is one row off.** `1 - 0.7 - 0.15` is
`0.15000000000000005`, so `ceil()` takes an extra test row; **100 of 100**
sizes n = 20…2000 miss, while the default 80/10/10 misses on **0**.

**FINDING: the doc's own split snippet is 70/10/20** (700 / 100 / 200), beside
bullets that say 80/10/10.

**CONTROL:** integer test sizes (300, then 150) give exactly 700 / 150 / 150.
