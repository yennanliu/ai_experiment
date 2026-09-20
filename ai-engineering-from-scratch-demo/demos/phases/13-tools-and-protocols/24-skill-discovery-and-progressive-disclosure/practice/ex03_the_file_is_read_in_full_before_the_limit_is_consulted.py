"""Exercise 3 — the file is read in full before the limit is consulted.

    Add a byte-size limit to `load_reference`. Test a file exactly at the
    limit and one byte above it.

Reading of the exercise: a byte limit next to an existing character limit is
only interesting if the two disagree, so the boundary cases are run in both
units on the same file and the multibyte case is the third test the exercise
implies. Writing the check then forces a decision the shipped code already
made in the other direction -- whether the limit is consulted before or
after the read.

**ANSWER: exactly at the limit passes and one byte over raises, and the
limit is read from `stat()` before the file is opened.** **1024** bytes
returns 1024 characters; **1025** raises `ReferencePathError` having read
**0** of them. The same two files pass the shipped character check, because
`max_chars` was never the same question.

**FINDING: the file is read in full before the limit is consulted.**
`load_reference` calls `read_text()` and then tests `len(content)`, so a
file 100x over the limit is decoded into memory before being rejected --
**524288** characters read to refuse a reference. `st_size` answers before
anything is opened, which is the only reason a byte limit is worth having.

**FINDING: characters and bytes are different limits, and the module already
uses both.** `FRONTMATTER_LIMIT` counts `len(raw_line.encode())` while
`load_reference` and `load_skill_body` count `len(content)`. A file of
**1024** CJK characters is **3072** bytes: it passes a 1024-character limit
and fails a 1024-byte one, so the two cannot be spelled the same way.

**FINDING: the character limit is not even a bound on memory.** The
strictest defensible reading of `max_chars=12_000` allows a 48KB file at
four bytes per character, and the shipped path has already allocated all of
it by the time it decides. A byte limit bounds the read; a character limit
bounds the result.

Structure: `bounded()` is the replacement -- stat first, then read -- and
`probe()` runs one file through both it and the shipped function.
"""

from __future__ import annotations

import pathlib
import tempfile

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "24-skill-discovery-and-progressive-disclosure"
LIMIT = 1_024


def bounded(ref, entry, reference, max_bytes=LIMIT):
    """The byte limit, answered from stat() before the file is opened."""
    target = ref.validate_reference(pathlib.Path(entry.directory), reference)
    size = target.stat().st_size
    if size > max_bytes:
        raise ref.ReferencePathError(f"reference is {size} bytes, over {max_bytes}")
    return target.read_text(encoding="utf-8")


def outcome(call):
    try:
        return len(call()), None
    except Exception as exc:
        return 0, type(exc).__name__


def probe(ref, entry, name, max_chars=12_000):
    """The same file through the new byte check and the shipped character check."""
    by_bytes = outcome(lambda: bounded(ref, entry, f"references/{name}"))
    by_chars = outcome(lambda: ref.load_reference(entry, f"references/{name}", max_chars))
    return {"bytes": by_bytes, "chars": by_chars}


def write(directory, name, text):
    (directory / "references" / name).write_text(text, encoding="utf-8")


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory() as temp:
        base = pathlib.Path(temp)
        ref._write_skill(base / "user", "evidence-report",
                         "Report evidence when an audit completes.", "# Evidence\n")
        skill = base / "user" / "evidence-report"
        write(skill, "at.md", "a" * LIMIT)
        write(skill, "over.md", "a" * (LIMIT + 1))
        write(skill, "wide.md", "中" * LIMIT)
        write(skill, "huge.md", "a" * (LIMIT * 512))
        candidates = ref.discover_scope(ref.Scope("user", base / "user"))
        catalog = ref.build_catalog(candidates, ("user",), ref.CatalogBudget())
        entry = catalog.entries[0]

        at, over = probe(ref, entry, "at.md"), probe(ref, entry, "over.md")
        wide = probe(ref, entry, "wide.md", max_chars=LIMIT)
        huge_chars = outcome(lambda: ref.load_reference(entry, "references/huge.md", LIMIT))
        huge_bytes = outcome(lambda: bounded(ref, entry, "references/huge.md"))
        source = pathlib.Path(ref.__file__).read_text(encoding="utf-8")
        body = source.split("def load_reference(")[1]
        return {
            "at": at, "over": over, "wide": wide,
            "wide_chars": LIMIT, "wide_bytes": (skill / "references" / "wide.md").stat().st_size,
            "huge_chars": huge_chars, "huge_bytes": huge_bytes,
            "huge_size": (skill / "references" / "huge.md").stat().st_size,
            "read_first": body.index("read_text") < body.index("> max_chars"),
            "frontmatter_unit": "encode" in source.split("def _frontmatter(")[1][:600],
            "reference_unit": "len(content)" in body,
            "max_chars_default": 12_000,
        }


def verify(result):
    return [
        practice.Check(
            "ANSWER: exactly at the limit passes, one byte over raises, and stat decides",
            all([result["at"]["bytes"] == (LIMIT, None),
                 result["over"]["bytes"] == (0, "ReferencePathError"),
                 result["at"]["chars"] == (LIMIT, None),
                 result["over"]["chars"] == (LIMIT + 1, None)]),
            f"{LIMIT} bytes returns {result['at']['bytes'][0]} characters and {LIMIT + 1} "
            f"raises {result['over']['bytes'][1]} having read {result['over']['bytes'][0]} "
            f"of them, while the shipped character check returns {result['over']['chars'][0]} "
            "for the same file -- max_chars was never the same question",
        ),
        practice.Check(
            "FINDING: the file is read in full before the limit is consulted",
            all([result["read_first"], result["huge_chars"] == (0, "ReferencePathError"),
                 result["huge_bytes"] == (0, "ReferencePathError"),
                 result["huge_size"] == LIMIT * 512]),
            f"load_reference calls read_text() and then tests len(content), so a "
            f"{result['huge_size']}-byte file is decoded into memory before being refused. "
            "stat() answers before anything is opened, which is the only reason a byte "
            "limit is worth adding rather than tightening the character one",
        ),
        practice.Check(
            "FINDING: characters and bytes are different limits, and the module uses both",
            all([result["frontmatter_unit"], result["reference_unit"],
                 result["wide_bytes"] == LIMIT * 3,
                 result["wide"]["chars"] == (LIMIT, None),
                 result["wide"]["bytes"] == (0, "ReferencePathError")]),
            f"FRONTMATTER_LIMIT counts encoded bytes while load_reference counts characters. "
            f"A file of {result['wide_chars']} CJK characters is {result['wide_bytes']} "
            f"bytes: it passes a {LIMIT}-character limit and fails a {LIMIT}-byte one, so "
            "the two cannot be spelled the same way",
        ),
        practice.Check(
            "FINDING: the character limit is not even a bound on memory",
            all([result["max_chars_default"] == 12_000, result["wide_bytes"] == LIMIT * 3]),
            f"max_chars={result['max_chars_default']} admits a 48KB file at four bytes per "
            "character, and the shipped path has already allocated it by the time it "
            "decides. A byte limit bounds the read; a character limit bounds the result, "
            "and only one of those is a resource control",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
