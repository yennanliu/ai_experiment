"""Exercise 4 — the anchor and its hash live in the same file.

    Design an alignment anchor for a coding agent. What text, stored where,
    checked how?

Reading of the exercise: three questions, so three answers, and each one is
checked against the anchor this lesson already ships -- because the shipped
one answers all three and gets the middle one wrong in a way that is easy to
miss and easy to fix.

**ANSWER: the agent's own contribution contract, stored in a checkout the
agent cannot write, checked by comparing a hash it does not hold.** For a
coding agent in this repository the text is the part of `DESIGN.md` that says
what a solution may not do -- answer a different question, fork the lesson's
implementation, improve the exercise. It is stored the way the exercise text
already is: in the reference checkout, resolved through `AIEFS_REFERENCE`,
outside the tree the agent edits. It is checked the way the spec already is:
re-read at grade time and hashed, with the expected value supplied by the
caller rather than read from a constant beside it.

**FINDING: the shipped anchor stores the text and its hash in one file.**
`OBJECTIVE` is **48** characters, `OBJECTIVE_HASH` is its sha256 truncated to
**16** hex digits, and `Agent.objective` defaults to the same constant -- all
**3** in the same module. `gate_anchor` compares a value the agent carries
against a constant the agent's file declares, so an edit that changes both
passes.

**FINDING: the property the design needs already exists next door.** The
spec-drift test re-reads the lesson's own `docs/en.md` through `parity` and
hashes it; that file lives in a different checkout and **0** of this
repository's gates can write to it. So the anchor design the exercise asks
for is a second application of a mechanism already running on the exercise
text -- which is the strongest argument for it.

**FINDING: truncation is the smaller problem.** **16** hex digits is **64**
bits, which is a birthday bound of **2^32** -- enough against accident and
irrelevant against an editor. The anchor's strength is set by where the
expected value lives, not by how many digits it has, and moving it costs one
parameter on `gate_anchor`.

Structure: `anchor()` reads the shipped constants; `outside()` resolves the
upstream file the proposed design would store the text in.
"""

from __future__ import annotations

import hashlib
import inspect
import pathlib

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "08-bounded-self-improvement"

HERE = pathlib.Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "harness").is_dir())
DIGITS = 16


def anchor(ref):
    """The shipped anchor: its text, its hash, and where each is declared."""
    module = inspect.getsourcefile(ref.gate_anchor)
    source = pathlib.Path(module).read_text(encoding="utf-8")
    return {
        "text": ref.OBJECTIVE,
        "length": len(ref.OBJECTIVE),
        "hash": ref.OBJECTIVE_HASH,
        "digits": len(ref.OBJECTIVE_HASH),
        "declared": sum(source.count(name) > 0
                        for name in ("OBJECTIVE =", "OBJECTIVE_HASH =", "objective:")),
        "recomputes": hashlib.sha256(ref.OBJECTIVE.encode()).hexdigest()[:DIGITS],
        "module": pathlib.Path(module).name,
    }


def outside():
    """The upstream file the proposed anchor would live in, and whether we can write it."""
    upstream = parity.lesson_dir(PHASE, LESSON) / "docs" / "en.md"
    inside = [p for p in (upstream,) if ROOT in p.parents]
    return upstream.is_file(), len(inside)


def drift_check():
    """How the exercise text is already anchored: re-read upstream, then hashed."""
    tests = (HERE / "tests" / "test_practice.py").read_text(encoding="utf-8")
    return "parity.doc_text" in tests and "spec_hash" in tests


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shipped = anchor(ref)
    reachable, inside = outside()
    agent = ref.Agent()
    return {
        **shipped,
        "default_is_constant": agent.objective == ref.OBJECTIVE,
        "gate_reads_constant": "OBJECTIVE_HASH" in inspect.getsource(ref.gate_anchor),
        "gate_parameters": len(inspect.signature(ref.gate_anchor).parameters),
        "upstream_reachable": reachable,
        "upstream_inside_repo": inside,
        "drift_check_exists": drift_check(),
        "bits": DIGITS * 4,
        "birthday": 2 ** (DIGITS * 4 // 2),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: text, an out-of-tree store, and a caller-supplied hash",
            all([result["upstream_reachable"], result["upstream_inside_repo"] == 0,
                 result["drift_check_exists"]]),
            "the text is the contribution contract, stored in the reference checkout "
            f"resolved through AIEFS_REFERENCE -- {result['upstream_inside_repo']} of "
            "its paths sit inside this repository -- and checked by the same re-read "
            "and hash the exercise text already gets",
        ),
        practice.Check(
            "FINDING: the shipped anchor stores the text and its hash in one file",
            all([result["length"] == 48, result["digits"] == DIGITS,
                 result["declared"] == 3, result["default_is_constant"],
                 result["gate_reads_constant"], result["gate_parameters"] == 1,
                 result["hash"] == result["recomputes"]]),
            f"OBJECTIVE is {result['length']} characters, OBJECTIVE_HASH its "
            f"{result['digits']}-digit truncation, and Agent.objective defaults to the "
            f"same constant -- all {result['declared']} declared in "
            f"{result['module']}, which gate_anchor reads with "
            f"{result['gate_parameters']} parameter",
        ),
        practice.Check(
            "FINDING: the property the design needs already exists next door",
            all([result["drift_check_exists"], result["upstream_inside_repo"] == 0]),
            "the spec-drift test re-reads the lesson's own docs/en.md through parity "
            "and hashes it, from a checkout no gate here can write -- the proposed "
            "anchor is that mechanism pointed at a second file",
        ),
        practice.Check(
            "FINDING: truncation is the smaller problem",
            all([result["bits"] == 64, result["birthday"] == 2 ** 32]),
            f"{result['digits']} hex digits is {result['bits']} bits, a birthday bound "
            f"of {result['birthday']:.0e} -- enough against accident and irrelevant "
            "against an editor, so the strength is set by where the expected value "
            "lives",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
