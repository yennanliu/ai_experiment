#!/usr/bin/env python3
"""Generate the derivable half of each lesson README, between markers.

The exercise table, the run commands and the source link all come from the
manifest, so they cannot drift from it. Prose answers and findings are written
by hand and live *outside* the markers, untouched by this script.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from harness import explain, manifest, runner  # noqa: E402

START, END = "<!-- generated:start -->", "<!-- generated:end -->"

TEMPLATE = """{start}
# {phase} / {lesson}

Solutions to all {n} exercises. Source: [lesson page]({url}) · upstream spec
`{source}`

```bash
uv run demo practice run {lesson} --ex 1
uv run demo explain {lesson} --ex 1
uv run pytest demos/phases/{phase}/{lesson}
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
{rows}
{end}"""


def render(pack) -> str:
    rows = []
    for ex in pack.exercises:
        text = ex.en.replace("|", "\\|").replace("**", "")
        if len(text) > 96:
            text = text[:93].rstrip() + "…"
        ships = f"`{ex.filename}`" if ex.kind != "explain" else "prose, below"
        rows.append(f"| {ex.index} | {text} | {ex.kind} | {ex.tier} | {ships} |")
    return TEMPLATE.format(
        start=START, end=END, phase=pack.phase, lesson=pack.lesson,
        n=len(pack.exercises), url=explain.lesson_url(pack.phase, pack.lesson),
        source=pack.source, rows="\n".join(rows))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("lesson", nargs="?")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    stale = []
    for man_path in runner._manifests(args.lesson):
        pack = manifest.load_practice(man_path)
        readme = man_path.parent / "README.md"
        block = render(pack)
        if readme.exists():
            text = readme.read_text(encoding="utf-8")
            if START in text and END in text:
                head, _, rest = text.partition(START)
                _, _, tail = rest.partition(END)
                updated = head + block + tail
            else:
                updated = block + "\n\n" + text
        else:
            updated = block + "\n"
        if args.check:
            if not readme.exists() or updated != readme.read_text(encoding="utf-8"):
                stale.append(str(readme.parent.parent.name))
            continue
        readme.write_text(updated, encoding="utf-8")
        print(f"wrote {readme.parent.parent.name}/practice/README.md")
    for name in stale:
        print(f"stale README: {name}")
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
