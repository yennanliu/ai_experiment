#!/usr/bin/env python3
"""Derive notebook.ipynb from run.py with jupytext. (D7)

The reference repo has 295 `notebook/` directories and zero notebooks in them --
a promise nobody kept. Hand-maintaining a notebook beside every `run.py` would
reproduce that failure at 511x scale, so notebooks here are *generated*: one
source of truth, and an empty notebook directory is structurally impossible.

Generated notebooks are gitignored. They are a build artefact, not content.

    python scripts/notebooks.py [--clean] [lesson-substring]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.manifest import discover  # noqa: E402
from harness.runner import DEMOS_ROOT  # noqa: E402


def build(demo) -> Path | None:
    source = demo.entrypoint_path
    if not source.exists():
        return None
    target = demo.path / "notebook.ipynb"
    subprocess.run(
        [sys.executable, "-m", "jupytext", "--to", "notebook",
         "--output", str(target), str(source)],
        check=True, capture_output=True, text=True,
    )
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", nargs="?", help="only build demos matching this")
    parser.add_argument("--clean", action="store_true", help="delete them instead")
    args = parser.parse_args()

    demos = discover(DEMOS_ROOT)
    if args.target:
        demos = [d for d in demos if args.target in d.lesson]

    for demo in demos:
        notebook = demo.path / "notebook.ipynb"
        if args.clean:
            if notebook.exists():
                notebook.unlink()
                print(f"removed  {notebook.relative_to(DEMOS_ROOT.parent)}")
            continue
        built = build(demo)
        if built:
            print(f"built    {built.relative_to(DEMOS_ROOT.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
