#!/usr/bin/env python3
r"""Patch a scaffolded `practice.yaml` with the values a solution actually asserts.

`scaffold_practice.py` proposes `kind`, `tier` and `deps_group` from keywords and
leaves `verifies`/`cites` as TODOs. Once the solutions are written those guesses
need correcting — most often the kind, since the classifier reads "explain" or
"describe" in an exercise that ships code.

Used as a library rather than a CLI, because the values are prose:

    from finalize_practice import finalize
    finalize("02-ml-fundamentals", "01-what-is-machine-learning", {
        1: {"slug": "three_way_split", "kind": "code", "verifies": "..."},
        2: {"kind": "explain", "cites": "The Three Types of Machine Learning"},
    })

Idempotent: re-running replaces rather than appends, so a second pass cannot
produce the duplicate keys the manifest parser rejects.

Two traps this file exists to avoid. `verifies` strings are prose about numerics
and routinely contain backslashes (`\kappa`, `\sigma`), which `re.sub` would read
as replacement escapes — so every substitution goes through a function, whose
return value is used literally. And a mistyped exercise index would otherwise
match no block and update nothing, while the function still wrote the file and
printed success; unknown indices are rejected before anything is written.
"""

from __future__ import annotations

import pathlib
import re
import textwrap

SIMPLE = ("slug", "tier", "deps_group", "kind")
DEMOS = pathlib.Path(__file__).resolve().parent.parent / "demos" / "phases"


def _literal(text: str):
    """A replacement *function*, so backslashes in `text` survive intact."""
    return lambda _match: text


FOLD_WIDTH = 96


def _fold(key: str, text: str) -> str:
    """`verifies` as a `>-` folded block scalar.

    Prose about thresholds routinely contains ": " -- "the destructive step: per-image
    min-max" -- and a plain YAML scalar may not. Written flat, 33 of these manifests
    parsed under `harness/yamlite` and failed every strict YAML reader with "mapping
    values are not allowed here". `>-` is the one form both agree on, and it drops the
    trailing newline `>` would add, so the value is byte-identical either way.
    """
    body = " ".join(str(text).split())
    lines = textwrap.wrap(body, width=FOLD_WIDTH, break_long_words=False, break_on_hyphens=False)
    return f"    {key}: >-\n" + "\n".join(f"      {line}" for line in lines) + "\n"


def _drop(body: str, key: str) -> str:
    """Remove `key` and its continuation lines, whether flat or a block scalar."""
    return re.sub(rf"(?m)^    {key}:(?: .*)?\n(?:      .*\n)*", "", body)


def _apply(body: str, update: dict) -> str:
    for key in SIMPLE:
        if key in update:
            body = re.sub(rf"(?m)^    {key}: .*$", _literal(f"    {key}: {update[key]}"), body)
    if update.get("kind") == "code" or "verifies" in update:
        body = _drop(body, "cites")
    if update.get("kind") == "explain":
        body = _drop(body, "verifies")
    for key in ("verifies", "cites"):
        if key in update:
            body = _drop(body, key).rstrip("\n") + "\n" + _fold(key, update[key])
    if "uses_reference" in update:
        body = re.sub(r"(?m)^    uses_reference:\n(?:      - .*\n)*", "", body)
        if update["uses_reference"]:
            block = "\n".join(f"      - {u}" for u in update["uses_reference"])
            body = body.rstrip("\n") + f"\n    uses_reference:\n{block}\n"
    return body


def finalize(phase: str, lesson: str, updates: dict) -> pathlib.Path:
    path = DEMOS / phase / lesson / "practice" / "practice.yaml"
    blocks = re.split(r"(?m)^(  - index: \d+$)", path.read_text(encoding="utf-8"))
    markers = [blocks[i] for i in range(1, len(blocks), 2)]
    known = {int(m.split(":")[1]) for m in markers}
    unknown = sorted(set(updates) - known)
    if unknown:
        raise KeyError(f"{phase}/{lesson}: no such exercise index {unknown} "
                       f"(manifest has {sorted(known)})")
    out = [blocks[0]]
    for i in range(1, len(blocks), 2):
        marker, body = blocks[i], blocks[i + 1]
        out.append(marker + _apply(body, updates.get(int(marker.split(":")[1]), {})))
    path.write_text("".join(out), encoding="utf-8")
    print(f"finalized {phase}/{lesson}")
    return path
