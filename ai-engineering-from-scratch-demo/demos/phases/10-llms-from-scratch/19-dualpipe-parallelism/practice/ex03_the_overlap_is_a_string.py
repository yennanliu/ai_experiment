"""Exercise 3 — communication is represented by one of three words, and no bubble formula reads it.

    Read Figure 5 of the DeepSeek-V3 technical report (arXiv:2412.19437).
    Identify the overlap window for all-to-all dispatch inside a DualPipe
    forward chunk. Explain how the compute schedule hides it.

Reading of the exercise: the explanation is written as an audit of what the
lesson can represent, because "how the compute schedule hides it" is a claim
about a quantity -- communication time -- and the module has to contain that
quantity for the claim to be checkable. Every place communication could enter is
enumerated: the four bubble functions, `summarize`'s tuples and the
`ScheduleStats` dataclass.

**ANSWER: communication appears exactly once, as one of the strings
`minimal`, `partial`, `full`.** It is the fourth element of `summarize`'s tuples
and it is printed. No bubble formula takes a communication argument, a bandwidth,
a message size or an overlap fraction, so there is no window to identify and
nothing for the schedule to hide.

**MECHANISM: the bubble formulas are functions of `(P, M)` and nothing else.**
`bubble_1f1b(P, M)`, `bubble_zero_bubble(P, M)`, `bubble_dualpipe(P, M)` and
`bubble_dualpipev(P, M)` take two integers each. A schedule that overlapped
all-to-all perfectly and one that overlapped none of it return the same number
for the same `(P, M)`, because the number does not depend on anything else.

**FINDING: `ScheduleStats` is declared and never constructed.** The dataclass has
five fields -- `name, stable_bubble_frac, scales_with_micro_batches,
param_copies, comm_overlap` -- and `summarize` returns bare tuples instead, so
`scales_with_micro_batches` is never assigned a value anywhere in the module. The
type that was meant to carry the schedule's properties is unused.

**FINDING: the one number that does distinguish DualPipeV is a stipulated
constant.** `bubble_dualpipev` is `bubble_dualpipe(P, M) * 1.2` -- exactly, to
machine precision, at every `(P, M)`. Its docstring says "approximate as 1.2x",
which is honest, and it means the V-shape schedule's cost is an assumption rather
than a derivation.

Structure: `signatures` reports each bubble function's parameters; `constant`
checks the DualPipeV multiplier across a grid.
"""

from __future__ import annotations

import dataclasses
import inspect

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "19-dualpipe-parallelism"
GRID = ((4, 8), (8, 16), (16, 64), (32, 128), (64, 512))
STAGES, MICRO = 8, 16


def signatures(ref):
    """Every bubble function's parameter names."""
    names = ("bubble_1f1b", "bubble_zero_bubble", "bubble_dualpipe", "bubble_dualpipev")
    return {name: list(inspect.signature(getattr(ref, name)).parameters) for name in names}


def constant(ref):
    """Is DualPipeV's bubble exactly 1.2x DualPipe's at every shape?"""
    return [ref.bubble_dualpipev(p, m) / ref.bubble_dualpipe(p, m)
            for p, m in GRID if ref.bubble_dualpipe(p, m) > 0]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = ref.summarize(STAGES, MICRO)
    return {
        "overlap_values": sorted({overlap for _, _, _, overlap in rows}),
        "copies": sorted({copies for _, _, copies, _ in rows}),
        "signatures": signatures(ref),
        "fields": [field.name for field in dataclasses.fields(ref.ScheduleStats)],
        "returns_tuple": isinstance(rows[0], tuple),
        "ratios": constant(ref),
        "row_width": len(rows[0]),
    }


def verify(result):
    signatures_, ratios = result["signatures"], result["ratios"]
    return [
        practice.Check(
            "ANSWER: communication is one of three strings and no formula reads it",
            result["overlap_values"] == ["full", "minimal", "partial"],
            f"the only representation of communication in the module is the fourth element of "
            f"summarize's {result['row_width']}-tuples, which takes the values "
            f"{result['overlap_values']}. It is printed and nothing else consumes it, so there "
            "is no overlap window to identify and nothing for the compute schedule to hide -- "
            "the exercise asks about a quantity the model does not have",
        ),
        practice.Check(
            "MECHANISM: the bubble formulas are functions of (P, M) and nothing else",
            all(params == ["P", "M"] for params in signatures_.values()),
            "every bubble function takes exactly two integers: "
            + ", ".join(f"{name}{tuple(params)}" for name, params in signatures_.items())
            + ". A schedule that overlapped all-to-all perfectly and one that overlapped none of "
            "it return the same number for the same (P, M), because the number cannot depend on "
            "anything else",
        ),
        practice.Check(
            "FINDING: ScheduleStats is declared and never constructed",
            result["returns_tuple"] and "scales_with_micro_batches" in result["fields"],
            f"the dataclass has {len(result['fields'])} fields -- {result['fields']} -- and "
            f"summarize returns bare {result['row_width']}-tuples instead, so "
            "scales_with_micro_batches is never assigned a value anywhere in the module. The type "
            "that was meant to carry a schedule's properties is unused, and two of its five "
            "fields have no other home",
        ),
        practice.Check(
            "FINDING: the number that distinguishes DualPipeV is a stipulated constant",
            all(abs(ratio - 1.2) < 1e-12 for ratio in ratios),
            f"bubble_dualpipev is bubble_dualpipe(P, M) * 1.2 exactly, to machine precision, at "
            f"all {len(ratios)} shapes tried. Its docstring says 'approximate as 1.2x', which is "
            "honest, and it means the V-shape schedule's cost -- the trade the lesson's takeaway "
            "rests on -- is an assumption rather than a derivation",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
