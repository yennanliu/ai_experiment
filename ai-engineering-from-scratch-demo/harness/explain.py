"""`run.py --explain`: what this demo proves, with nothing installed. (D6)

Every demo must be able to explain itself before a learner pays for a
`uv sync`, so this reads `demo.yaml` and prints -- no numpy, no torch, no
network. The lesson URL is derived by string substitution from the lesson path,
which is the concrete payoff of the path-identical mirror (D1).
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import quote

from .manifest import load
from .tiers import TIER_DESCRIPTION

SITE = "https://yennj12.js.org/ai-engineering-from-scratch"


def lesson_url(lesson: str) -> str:
    return f"{SITE}/lesson.html?path={quote(lesson, safe='')}"


def explain(demo_file: str | Path) -> bool:
    """If `--explain` is in argv, print the manifest and return True.

    Call it from a demo's `__main__` block:

        if explain(__file__):
            raise SystemExit(0)
    """
    if "--explain" not in sys.argv[1:]:
        return False

    demo = load(Path(demo_file).resolve().parent / "demo.yaml")
    width = 74
    print("=" * width)
    print(demo.title)
    print("=" * width)
    print(f"lesson    {demo.lesson}")
    print(f"read      {lesson_url(demo.lesson)}")
    print(f"tier      {demo.tier}  {TIER_DESCRIPTION[demo.tier]}")
    print(f"budget    {demo.runtime_seconds}s")
    print(f"install   uv sync --extra {demo.deps_group}")
    if demo.needs_env:
        print(f"env       {', '.join(demo.needs_env)} (live mode only)")
    print()
    print("proves")
    for line in _wrap(demo.proves, width - 4):
        print(f"    {line}")
    if demo.parity_with:
        print()
        print("parity against the lesson's own code")
        print(f"    {demo.parity_with}")
    return True


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines
