"""Exercise 4 — make_splits never returns an exact 70/15/15: float error moves one row.

    Create a 70/15/15 train/val/test split with a fixed seed and verify the sizes

Reading of the exercise: the split is made by the lesson's own
`make_splits(ds, train_ratio=0.7, val_ratio=0.15, seed=42)`. `datasets` is not
installed (and the lesson's module calls `sys.exit(1)` at import without it), so
`datasets` is stubbed with a minimal `Dataset` whose `train_test_split` delegates
to scikit-learn's `train_test_split`: Hugging Face copied its size rule from
scikit-learn -- `ceil(test_size * n)` for the test side, the rest for train --
so the sizes are the library's, not a guess. "Verify the sizes" is read as
checking them against `0.70 n / 0.15 n / 0.15 n` for every n divisible by 20.

**ANSWER: 699 / 150 / 151 at n = 1000, not 700 / 150 / 150.** The three parts
are disjoint, cover all 1000 rows, and the same seed reproduces them exactly.

**FINDING: the off-by-one is every n, and it is floating point.**
`1 - 0.7 - 0.15` is `0.15000000000000005`, so the first `test_size` is
`0.30000000000000004` and `ceil(0.30000000000000004 * 1000) = 301`; the second
split's `1 - val_fraction` is `0.5000000000000001`, so the 301 split 150 / 151.
All 100 sizes n = 20, 40, ..., 2000 come out one row short in train and one row
long in test. The lesson's own default 80/10/10 is exact on all of them, which
is why the lesson's demo never shows it.

**FINDING: the doc's own split snippet is 70/10/20, not the 80/10/10 it
describes.** `test_size=0.2` then `0.125` of the remaining 80% gives
700 / 100 / 200 at n = 1000, while the bullets above it say 80 / 10 / 10.

**CONTROL: integer sizes fix it.** Passing row counts (300, then 150) through
the same stubbed split gives exactly 700 / 150 / 150.

Structure: `load_lesson` stubs the HF imports; `sizes` runs `make_splits`.
"""

from __future__ import annotations

import contextlib
import io
import sys
import types
from unittest import mock

from sklearn.model_selection import train_test_split

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "09-data-management"
N, SEED, RATIOS = 1000, 42, (0.7, 0.15)


class Dataset:
    """Just enough of datasets.Dataset: len and a sklearn-backed train_test_split."""

    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def train_test_split(self, test_size, seed):
        train, test = train_test_split(self.rows, test_size=test_size, random_state=seed)
        return {"train": Dataset(train), "test": Dataset(test)}


def load_lesson():
    """Import data_utils with `datasets` and `huggingface_hub` stubbed out."""
    hf = types.ModuleType("datasets")
    hf.load_dataset, hf.Dataset = None, Dataset
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = None
    with mock.patch.dict(sys.modules, {"datasets": hf, "huggingface_hub": hub}):
        return parity.load_reference(PHASE, LESSON, "data_utils")


def sizes(ref, n, ratios=RATIOS, seed=SEED):
    with contextlib.redirect_stdout(io.StringIO()):
        parts = ref.make_splits(Dataset(range(n)), *ratios, seed=seed)
    return parts, [len(parts[k]) for k in ("train", "val", "test")]


def solve():
    ref = load_lesson()
    parts, got = sizes(ref, N)
    again, _ = sizes(ref, N)
    rows = [set(parts[k].rows) for k in ("train", "val", "test")]
    exact = {n: [n * 7 // 10, n * 3 // 20, n * 3 // 20] for n in range(20, 2001, 20)}
    first = Dataset(range(N)).train_test_split(test_size=300, seed=SEED)
    second = first["test"].train_test_split(test_size=150, seed=SEED)
    doc_a = Dataset(range(N)).train_test_split(test_size=0.2, seed=SEED)
    doc_b = doc_a["train"].train_test_split(test_size=0.125, seed=SEED)
    return {
        "sizes": got,
        "partition": sum(map(len, rows)) == N and set.union(*rows) == set(range(N)),
        "repeat": all(parts[k].rows == again[k].rows for k in parts),
        "off": sum(sizes(ref, n)[1] != want for n, want in exact.items()),
        "default_off": sum(sizes(ref, n, (0.8, 0.1))[1] != [n * 8 // 10, n // 10, n // 10]
                           for n in exact),
        "floats": (1 - 0.7 - 0.15, 1 - 0.15 / (0.15 + (1 - 0.7 - 0.15))),
        "integer": [len(first["train"]), len(second["train"]), len(second["test"])],
        "doc_snippet": [len(doc_b["train"]), len(doc_b["test"]), len(doc_a["test"])],
        "doc_says": "typically 80%" in parity.doc_text(PHASE, LESSON),
        "total": len(exact),
    }


def verify(result):
    test_ratio, second = result["floats"]
    return [
        practice.Check(
            "ANSWER: 699 / 150 / 151 at n = 1000, a reproducible partition",
            result["sizes"] == [699, 150, 151] and result["partition"] and result["repeat"],
            f"make_splits(0.7, 0.15, seed={SEED}) on {N} rows gives {result['sizes']}; the "
            "parts are disjoint, cover every row, and the same seed reproduces them",
        ),
        practice.Check(
            "FINDING: every n is off by one row, from floating point",
            result["off"] == result["total"] and result["default_off"] == 0,
            f"{result['off']} of {result['total']} sizes n = 20..2000 miss 70/15/15 by one "
            f"row: 1 - 0.7 - 0.15 = {test_ratio!r}, so ceil() takes an extra test row, and "
            f"the second split's test_size is {second!r}; the default 80/10/10 misses on "
            f"{result['default_off']}",
        ),
        practice.Check(
            "FINDING: the doc's own split snippet is 70/10/20, not its stated 80/10/10",
            result["doc_says"] and result["doc_snippet"] == [700, 100, 200],
            f"test_size=0.2 then 0.125 gives train/val/test {result['doc_snippet']} at "
            f"n = {N}, beside bullets that say 'typically 80%' / 10% / 10%",
        ),
        practice.Check(
            "CONTROL: integer test sizes give the exact split",
            result["integer"] == [700, 150, 150],
            f"test_size=300 then 150 through the same split gives {result['integer']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
