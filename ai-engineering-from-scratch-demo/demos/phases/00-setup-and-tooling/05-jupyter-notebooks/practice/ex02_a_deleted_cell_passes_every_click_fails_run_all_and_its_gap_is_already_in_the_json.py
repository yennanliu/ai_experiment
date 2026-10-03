"""Exercise 2 — a deleted cell passes every click, fails Run All, and its gap is in the JSON.

    Create a notebook with both markdown and code cells that loads a CSV, displays a
    dataframe, and plots a chart. Then run Kernel > Restart & Run All to verify it
    works top to bottom

Reading of the exercise: Jupyter is not a dependency here, but a `.ipynb` is
plain nbformat-4 JSON and a kernel is a namespace that code cells are `exec`'d
into, so both are built with the stdlib. The CSV is the lesson's own Step 5
table (`model`, `accuracy`, `training_time`), parsed out of `docs/en.md`; the
"dataframe" is printed rows and the chart is an SVG bar chart stored as a
`display_data` output, which is how a notebook carries a plot. "Restart & Run
All" is a fresh namespace and the cells in document order, stopping at the first
error, as Jupyter does.

**ANSWER: the clean notebook runs top to bottom.** Two markdown and four code
cells, zero errors, execution counts **[1, 2, 3, 4]**, three table rows, three
bars, and the third row is Neural Net at 0.94.

**FINDING: the lesson's "hidden state" trap, measured.** Run the same four
cells interactively, then delete the helper cell that builds `accuracy`. Every
remaining cell still shows its output and the saved file records no error, yet
Restart & Run All stops at the chart cell with **NameError: name 'accuracy' is
not defined**. Nothing in the visible notebook defines the name it depends on.

**FINDING: the ghost is visible without running anything.** nbformat stores each
code cell's `execution_count`, and the saved ghost notebook reads **[1, 3, 4]**:
count 2 ran and is no longer in the file. A ten-line stdlib scan for counts that
are not exactly 1..k flags it, and flags a notebook whose cells were clicked out
of order the same way, while the clean notebook passes. Restart & Run All finds
this by running the whole notebook; the JSON already says it.

**CONTROL: Run All is deterministic here.** Two Run All passes of the clean
notebook serialise to byte-identical JSON, so the difference above is the
deletion and not run-to-run noise.

Structure: `table` reads the lesson's Step 5 data; `run` is the kernel;
`notebook` serialises cells to nbformat; `scan` reads one without a kernel.
"""

from __future__ import annotations

import ast
import contextlib
import csv
import io
import json
import re

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "05-jupyter-notebooks"
LOAD = "import csv\nwith open(CSV_PATH) as f:\n    rows = list(csv.DictReader(f))\n"
HELPER = "accuracy = {r['model']: float(r['accuracy']) for r in rows}\n"
SHOW = "for r in rows:\n    print(f\"{r['model']:<14}{r['accuracy']:>6}{r['training_time']:>7}\")\n"
CHART = ("bars = ''.join(f'<rect x=\"{40 * i}\" y=\"{100 - 100 * v:.0f}\" width=\"30\" "
         "height=\"{100 * v:.0f}\"/>' for i, v in enumerate(accuracy.values()))\n"
         "SVG = f'<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"120\">{bars}</svg>'\n")
TITLE, NOTE = "# Model comparison\n", "Accuracy by model, from the CSV above.\n"


def table():
    """The lesson's Step 5 DataFrame literal, read out of docs/en.md."""
    doc = parity.doc_text(PHASE, LESSON, "en")
    literal = re.search(r"pd\.DataFrame\((\{.*?\})\)", doc, re.S).group(1)
    return ast.literal_eval(literal)


def run(cells, namespace):
    """Execute code cells in the given order; returns (cell outputs, error or None)."""
    outputs = []
    for count, source in enumerate(cells, 1):
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                exec(source, namespace)
        except Exception as exc:  # a notebook records the error and stops Run All
            return outputs, f"{type(exc).__name__}: {exc}"
        shown = [{"output_type": "stream", "name": "stdout", "text": out.getvalue()}]
        if "SVG" in source:
            shown.append({"output_type": "display_data", "metadata": {},
                          "data": {"image/svg+xml": namespace["SVG"]}})
        outputs.append((source, count, shown))
    return outputs, None


def notebook(ran):
    """nbformat-4 JSON for markdown + the executed code cells, in document order."""
    cells = [{"cell_type": "code", "metadata": {}, "source": source,
              "execution_count": count, "outputs": shown} for source, count, shown in ran]
    cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source": TITLE})
    cells.insert(len(cells) - 1, {"cell_type": "markdown", "metadata": {}, "source": NOTE})
    return json.dumps({"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": cells},
                      indent=1, sort_keys=True)


def scan(text):
    """Read a saved notebook without running it: cell kinds, error outputs, counts, flag."""
    cells = json.loads(text)["cells"]
    code = [c for c in cells if c["cell_type"] == "code"]
    counts = [c["execution_count"] for c in code]
    errors = sum(o["output_type"] == "error" for c in code for o in c["outputs"])
    return {"markdown": len(cells) - len(code), "code": len(code), "errors": errors,
            "counts": counts, "flagged": counts != list(range(1, len(counts) + 1))}


def solve():
    import pathlib
    import tempfile
    data = table()
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "models.csv"
        with path.open("w", newline="") as handle:
            csv.writer(handle).writerows([list(data), *zip(*data.values())])
        fresh = lambda: {"CSV_PATH": str(path)}  # noqa: E731
        clean, error = run([LOAD, HELPER, SHOW, CHART], fresh())
        same = notebook(clean) == notebook(run([LOAD, HELPER, SHOW, CHART], fresh())[0])
        ghost = notebook([cell for cell in clean if cell[0] != HELPER])
        saved = [c["source"] for c in json.loads(ghost)["cells"] if c["cell_type"] == "code"]
        _, ghost_error = run(saved, fresh())
        clicked = {src: count for src, count, _ in run([LOAD, HELPER, CHART, SHOW], fresh())[0]}
    shuffled = [(src, clicked[src], []) for src in (LOAD, HELPER, SHOW, CHART)]
    rows = clean[2][2][0]["text"].splitlines()
    bars = clean[3][2][1]["data"]["image/svg+xml"].count("<rect")
    return {
        "clean": scan(notebook(clean)), "error": error, "ghost": scan(ghost),
        "ghost_error": ghost_error, "shuffled": scan(notebook(shuffled)), "same": same,
        "size": len(notebook(clean)), "rows": rows, "bars": bars,
        "drawn": len(rows) == 3 and rows[2].startswith("Neural Net") and bars == 3,
    }


def verify(result):
    clean, ghost, shuffled = result["clean"], result["ghost"], result["shuffled"]
    return [
        practice.Check(
            "ANSWER: the clean notebook runs top to bottom",
            result["error"] is None and clean["markdown"] == 2 and result["drawn"],
            f"{clean['markdown']} markdown and {clean['code']} code cells, error "
            f"{result['error']}, counts {clean['counts']}, rows {result['rows']}, "
            f"{result['bars']} bars",
        ),
        practice.Check(
            "FINDING: deleting the helper cell passes every click and fails Run All",
            result["ghost_error"] == "NameError: name 'accuracy' is not defined"
            and ghost["errors"] == 0,
            f"the saved ghost notebook records {ghost['errors']} errors and keeps every "
            f"output, yet Restart & Run All stops with {result['ghost_error']}",
        ),
        practice.Check(
            "FINDING: the ghost is visible in the JSON without running anything",
            ghost["counts"] == [1, 3, 4] and ghost["flagged"] and shuffled["flagged"]
            and not clean["flagged"],
            f"execution counts -- clean {clean['counts']}, ghost {ghost['counts']} (count 2 ran "
            f"and is gone), out of order {shuffled['counts']}: the 1..k scan flags the last two",
        ),
        practice.Check(
            "CONTROL: Run All is deterministic, so the difference is the deletion",
            result["same"],
            f"two Run All passes serialise to identical JSON, {result['size']} bytes each",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
