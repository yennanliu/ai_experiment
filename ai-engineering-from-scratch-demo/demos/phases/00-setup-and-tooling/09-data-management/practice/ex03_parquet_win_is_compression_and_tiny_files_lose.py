"""Exercise 3 — the Parquet-style win is compression, not columns; a tiny file "0.6x smaller".

    Convert a dataset to Parquet and compare the file size to CSV

Reading of the exercise: the conversion is the lesson's own `convert_format`,
which writes CSV, JSON and Parquet and prints the CSV/Parquet ratio. `datasets`
and `pyarrow` are absent, so the stubbed dataset writes CSV and JSON Lines
exactly (the formats `datasets` emits) and a Parquet stand-in: each column
stored contiguously and gzip-compressed (GZIP is a Parquet codec), without
Parquet's dictionary/RLE encodings or its metadata footer. Three synthetic,
labelled tables go through it: 500 reviews (free text + a 0/1 label, shaped like
the lesson's rotten_tomatoes sample), 500 rows of four low-cardinality columns,
and 3 rows of the latter.

**ANSWER: the stand-in is 2.4x smaller than CSV on the reviews and 6.9x on the
low-cardinality table.** JSON Lines is 1.16x larger than CSV on the reviews and
4.0x on the coded table, because it repeats every key on every row.

**FINDING: the columnar layout itself buys at most 1.05x.** gzip of the
row-major CSV is also 2.4x smaller than CSV on the reviews -- the columnar file
beats it by 1.01x -- and on the low-cardinality table by only 1.05x. Without
Parquet's encodings, "Parquet: Small" in the doc's table is "compressed", which
is what the glossary's "Compressed CSV" says people say.

**FINDING: on a tiny table the lesson prints "Parquet is 0.6x smaller than
CSV".** Three rows are 75 bytes of CSV and 118 bytes of stand-in (a gzip header
per column; real Parquet adds a metadata footer on top), and the helper's fixed
wording turns a 1.6x growth into "smaller".

**CONTROL: the helper measured the files that were written.** Its printed CSV
byte count matches `stat()` on the CSV the stub wrote.

Structure: `table` builds the fixtures; `Dataset` writes them; `measure` runs it.
"""

from __future__ import annotations

import contextlib
import csv
import gzip
import io
import json
import random
import sys
import tempfile
import types
from unittest import mock

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "09-data-management"


class Dataset:
    """Writes CSV and JSON Lines as datasets does, and a columnar-gzip Parquet stand-in."""

    def __init__(self, rows):
        self.rows = rows

    def to_csv(self, path):
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(self.rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(self.rows)

    def to_json(self, path):
        with open(path, "w") as fh:
            fh.writelines(json.dumps(row) + "\n" for row in self.rows)

    def to_parquet(self, path):
        with open(path, "wb") as fh:
            for col in self.rows[0]:
                data = "\n".join(str(r[col]) for r in self.rows).encode()
                fh.write(gzip.compress(data, mtime=0))


def table(kind, n, seed=9):
    """SYNTHETIC rows: 'reviews' = free text + label; 'codes' = low-cardinality columns."""
    rng = random.Random(seed)
    vocab = ["".join(rng.choices("etaoinshrdlucmfwyp", k=rng.randint(2, 9))) for _ in range(2000)]
    if kind == "reviews":
        return [{"text": " ".join(rng.choices(vocab, k=rng.randint(12, 30))),
                 "label": rng.randint(0, 1)} for _ in range(n)]
    return [{"split": rng.choice(["train", "val", "test"]), "label": rng.randint(0, 1),
             "source": rng.choice(["web", "book", "news", "wiki"]),
             "length": rng.randint(1, 9) * 100} for _ in range(n)]


def load_lesson():
    hf = types.SimpleNamespace(load_dataset=None, Dataset=Dataset)
    hub = types.SimpleNamespace(hf_hub_download=None)
    with mock.patch.dict(sys.modules, {"datasets": hf, "huggingface_hub": hub}):
        return parity.load_reference(PHASE, LESSON, "data_utils")


def measure(ref, rows, name):
    out = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(out):
        paths = ref.convert_format(Dataset(rows), tmp, name)
        size = {k: p.stat().st_size for k, p in paths.items()}
        size["csv_gz"] = len(gzip.compress(paths["csv"].read_bytes(), mtime=0))
    said = [ln.strip() for ln in out.getvalue().splitlines() if "smaller" in ln or "CSV:" in ln]
    return size, said


def solve():
    ref = load_lesson()
    return {name: measure(ref, table(kind, n), name)
            for name, kind, n in (("reviews", "reviews", 500), ("codes", "codes", 500),
                                  ("tiny", "codes", 3))}


def verify(result):
    (rev, _), (codes, _), (tiny, said) = result["reviews"], result["codes"], result["tiny"]

    def ratio(s, a="csv", b="parquet"):
        return s[a] / s[b]

    return [
        practice.Check(
            "ANSWER: the stand-in is 2.4x smaller than CSV on reviews, 6.9x on codes",
            2.2 < ratio(rev) < 2.6 and 6.5 < ratio(codes) < 7.5 and ratio(rev, "json", "csv") > 1.1,
            f"CSV/Parquet {ratio(rev):.1f}x on 500 reviews ({rev['csv']:,} vs "
            f"{rev['parquet']:,} B), {ratio(codes):.1f}x on 500 coded rows; JSON Lines is "
            f"{ratio(rev, 'json', 'csv'):.2f}x CSV on reviews, "
            f"{ratio(codes, 'json', 'csv'):.1f}x on codes",
        ),
        practice.Check(
            "FINDING: the columnar layout buys only ~1.05x over gzip of the CSV",
            max(ratio(rev, "csv_gz", "parquet"), ratio(codes, "csv_gz", "parquet")) < 1.10,
            f"row-major gzip is {ratio(rev, 'csv', 'csv_gz'):.1f}x smaller than CSV on reviews "
            f"(columnar: {ratio(rev):.1f}x); on codes columnar beats row gzip by only "
            f"{ratio(codes, 'csv_gz', 'parquet'):.2f}x (reviews: "
            f"{ratio(rev, 'csv_gz', 'parquet'):.2f}x)",
        ),
        practice.Check(
            "FINDING: on 3 rows the lesson prints 'Parquet is 0.6x smaller than CSV'",
            ratio(tiny) < 1 and said[-1] == f"Parquet is {ratio(tiny):.1f}x smaller than CSV",
            f"{tiny['csv']} B of CSV became {tiny['parquet']} B, a "
            f"{1 / ratio(tiny):.1f}x growth; the helper said: '{said[-1]}'",
        ),
        practice.Check(
            "CONTROL: the helper's printed CSV size is the file's real size",
            said[0] == f"CSV:     {tiny['csv']:>10,} bytes",
            f"helper printed '{said[0]}', stat() gives {tiny['csv']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
