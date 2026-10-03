"""Exercise 1 — load_and_inspect shows one row; with no split it counts splits, then crashes.

    Load the `glue` dataset with the `mrpc` config and inspect the first 5 examples

Reading of the exercise: there is no network and no `datasets` here, so the
lesson's own `load_and_inspect("glue", config="mrpc")` runs against a stubbed
`load_dataset` that records its arguments and returns an MRPC-shaped table
(columns `sentence1, sentence2, label, idx`; split sizes 3668 / 408 / 1725 from
the GLUE MRPC card). The five rows are synthetic placeholders, labelled as such
-- the point is what the lesson's code does with a dataset, not MRPC's text.

**ANSWER: the call is `load_dataset(path="glue", name="mrpc", split="train")`,
and the first five examples are `ds.select(range(5))`.** The lesson's helper
routes `config` to `name` correctly and reports 3668 rows and four columns.

**FINDING: the helper inspects one example, not five.** It prints `ds[0]` and
nothing else; the transcript holds 1 of the 5 rows, so the exercise needs a
`select(range(5))` the lesson never shows.

**FINDING: `split=None` reports "Rows: 3", then raises before showing a row.**
The guard is `if split:`, so a falsy split is dropped and `load_dataset` returns
a `DatasetDict` -- a dict of the three splits. `len()` then counts splits, so
the helper prints "Rows: 3" for a 5801-row dataset; `DatasetDict` has
`column_names` but no `features`, so the next line raises AttributeError (and
`ds[0]` would be `KeyError: 0` after it).

**FINDING: without `datasets` the module cannot be imported at all.** Its import
guard calls `sys.exit(1)`, so a library user gets `SystemExit(1)`, not an
`ImportError` they could catch as one.

**CONTROL: the stub is what the helper saw.** It recorded exactly one call per
inspection, with the arguments above.

Structure: `load_lesson` stubs the HF imports; `Dataset` is the MRPC-shaped table.
"""

from __future__ import annotations

import contextlib
import io
import sys
import types
from unittest import mock

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "09-data-management"
SIZES = {"train": 3668, "validation": 408, "test": 1725}      # GLUE MRPC card
COLUMNS = ["sentence1", "sentence2", "label", "idx"]


class Dataset:
    """An MRPC-shaped split: synthetic rows, real column names and row count."""

    def __init__(self, split, n):
        self.split, self.n, self.column_names = split, n, COLUMNS
        self.features = {c: "string" if c.startswith("s") else "int" for c in COLUMNS}

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"sentence1": f"SYNTHETIC {self.split} {i} a", "sentence2":
                f"SYNTHETIC {self.split} {i} b", "label": i % 2, "idx": i}

    def select(self, indices):
        return [self[i] for i in indices]


class DatasetDict(dict):
    """HF's DatasetDict is a dict of splits with column_names, and no features."""

    @property
    def column_names(self):
        return {k: v.column_names for k, v in self.items()}


def fake_loader(calls):
    def load_dataset(path, name=None, split=None):
        calls.append({"path": path, "name": name, "split": split})
        splits = DatasetDict({s: Dataset(s, n) for s, n in SIZES.items()})
        return splits if split is None else splits[split]
    return load_dataset


def load_lesson(hf=True):
    stub = types.SimpleNamespace(load_dataset=None, Dataset=Dataset)
    mods = {"datasets": stub if hf else None,
            "huggingface_hub": types.SimpleNamespace(hf_hub_download=None)}
    with mock.patch.dict(sys.modules, mods):
        return parity.load_reference(PHASE, LESSON, "data_utils")


def inspect(ref, **kwargs):
    out, error, ds = io.StringIO(), None, None
    with contextlib.redirect_stdout(out):
        try:
            ds = ref.load_and_inspect("glue", "mrpc", **kwargs)
        except (KeyError, AttributeError) as exc:
            error = type(exc).__name__
    text = out.getvalue()
    rows = [line.strip() for line in text.splitlines() if "Rows" in line]
    return ds, rows, text.count("'sentence1': 'SYNTHETIC"), error


def solve():
    ref, calls = load_lesson(), []
    ref.load_dataset = fake_loader(calls)
    ds, rows, shown, _ = inspect(ref)
    _, broken, _, error = inspect(ref, split=None)
    try:
        exit_code = load_lesson(hf=False) and None
    except SystemExit as exc:
        exit_code = exc.code
    return {"calls": calls, "first_five": [row["idx"] for row in ds.select(range(5))],
            "rows_line": rows, "rows_shown": shown, "broken_rows": broken, "error": error,
            "exit_code": exit_code}


def verify(result):
    calls = result["calls"]
    return [
        practice.Check(
            "ANSWER: load_dataset(path='glue', name='mrpc', split='train'), then select(range(5))",
            calls[0] == {"path": "glue", "name": "mrpc", "split": "train"}
            and result["rows_line"] == ["Rows: 3668"] and result["first_five"] == [0, 1, 2, 3, 4],
            f"called {calls[0]}, printed {result['rows_line']}; idx {result['first_five']}",
        ),
        practice.Check(
            "FINDING: the helper shows 1 example, not 5", result["rows_shown"] == 1,
            f"its transcript contains {result['rows_shown']} row: it prints ds[0] only",
        ),
        practice.Check(
            "FINDING: split=None prints 'Rows: 3', then raises before any row",
            result["broken_rows"] == ["Rows: 3"] and result["error"] == "AttributeError",
            f"'if split:' drops the split, a DatasetDict comes back, it prints "
            f"{result['broken_rows']} for {sum(SIZES.values())} rows, then "
            f"{result['error']} on .features",
        ),
        practice.Check(
            "FINDING: importing the module without datasets is SystemExit(1)",
            result["exit_code"] == 1,
            f"without datasets, loading data_utils raised SystemExit({result['exit_code']})",
        ),
        practice.Check(
            "CONTROL: the stub recorded exactly the two inspections",
            len(calls) == 2 and calls[1]["split"] is None, f"recorded calls: {calls}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
