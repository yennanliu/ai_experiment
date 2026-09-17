"""Exercise 5 — of the two properties, the model records the one Chimera already had and labels the other.

    Compare DualPipe to Chimera (a competing bidirectional scheduler from 2021).
    Identify the two specific properties DualPipe added that Chimera did not
    have, using the paper's Section 3.4 as the reference.

Reading of the exercise: the comparison is made against what the lesson can
express, because a property DualPipe "added" has to be visible somewhere in the
model for the comparison to be checkable. Chimera is bidirectional and holds two
model copies -- that is its defining trick -- so `param_copies = 2` is the one
field DualPipe shares with it, and the two properties the exercise is asking
about must lie elsewhere.

**ANSWER: the model has two fields that could carry the answer, and one of them
is Chimera's own trick.**

    schedule       bubble    param_copies    comm_overlap
    1F1B           30.43%         1           minimal
    Zero Bubble    11.29%         1           partial
    DualPipe        5.45%         2           full
    DualPipeV       6.55%         1           partial

`param_copies = 2` is what Chimera did in 2021. `comm_overlap = "full"` is the
only field that separates DualPipe from everything else in the table -- and it
is a string, not a quantity. The second property DualPipe added, the fine-grained
forward/backward *chunk* split that makes the overlap possible, has no field at
all.

**FINDING: DualPipeV is the honest test of the comparison, and it fails it.**
DualPipeV has `param_copies = 1` -- it drops the very thing Chimera and DualPipe
share -- and the lesson still calls its overlap `partial` and its bubble 1.2x
DualPipe's. If dropping the second copy costs 20% of the bubble, then the second
copy was worth 20%, and the table says a schedule with Chimera's memory profile
loses a fifth of DualPipe's advantage.

**MECHANISM: only one number in the whole module separates the four schedules.**
Every schedule's bubble is `numerator(P) / denominator(P, M)`, and across the
four the numerators at P=8 are 14, 7, 3 and 3 with denominators 46, 62, 55 and
55. DualPipe and DualPipeV have *identical* formulas up to the stipulated 1.2,
so the comparison the exercise asks for reduces to one multiplier and two
labels.

**FINDING: the field that would have carried "scales with micro-batches" is
never set.** `ScheduleStats.scales_with_micro_batches` exists and is assigned
nowhere, and it is the field that would distinguish a schedule whose bubble is
constant in `M` from one whose bubble shrinks -- which, as Exercise 1 measures,
is none of them: all four shrink.

Structure: `table` reads the lesson's own `summarize`; `formula_parts`
recomputes each numerator and denominator so the four can be compared directly.
"""

from __future__ import annotations

import dataclasses

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "19-dualpipe-parallelism"
STAGES, MICRO = 8, 16


def formula_parts(stages, micro):
    """Each schedule's numerator and denominator, as its own source writes them."""
    dual = ((stages - 1) // 2, 3 * micro + (stages - 1))
    return {"1F1B": (2 * (stages - 1), 2 * micro + 2 * (stages - 1)),
            "Zero Bubble": (stages - 1, 3 * micro + 2 * (stages - 1)),
            "DualPipe": dual, "DualPipeV": dual}


def table(ref, stages, micro):
    return {name: {"bubble": bubble, "copies": copies, "overlap": overlap}
            for name, bubble, copies, overlap in ref.summarize(stages, micro)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = table(ref, STAGES, MICRO)
    parts = formula_parts(STAGES, MICRO)
    full = [name for name, row in rows.items() if row["overlap"] == "full"]
    return {
        "rows": rows,
        "parts": parts,
        "two_copies": [name for name, row in rows.items() if row["copies"] == 2],
        "full_overlap": full,
        "same_formula": parts["DualPipe"] == parts["DualPipeV"],
        "v_ratio": rows["DualPipeV"]["bubble"] / rows["DualPipe"]["bubble"],
        "fields": [f.name for f in dataclasses.fields(ref.ScheduleStats)],
        "assigned": isinstance(ref.summarize(STAGES, MICRO)[0], tuple),
    }


def verify(result):
    rows, parts = result["rows"], result["parts"]
    return [
        practice.Check(
            "ANSWER: one of the two candidate fields is Chimera's own trick",
            result["two_copies"] == ["DualPipe"] and result["full_overlap"] == ["DualPipe"],
            "the table is "
            + ", ".join(f"{name} {row['bubble']:.2%} copies={row['copies']} "
                        f"overlap={row['overlap']}" for name, row in rows.items())
            + ". param_copies=2 is what Chimera did in 2021, so it cannot be a property DualPipe "
            "added; comm_overlap='full' is the only field that separates DualPipe from everything "
            "else here, and it is a string. The fine-grained forward/backward chunk split that "
            "makes the overlap possible has no field at all",
        ),
        practice.Check(
            "FINDING: DualPipeV is the honest test of the comparison, and it fails it",
            rows["DualPipeV"]["copies"] == 1 and result["v_ratio"] > 1,
            f"DualPipeV has param_copies={rows['DualPipeV']['copies']} -- it drops the very thing "
            f"Chimera and DualPipe share -- and the lesson still calls its overlap "
            f"'{rows['DualPipeV']['overlap']}' and its bubble {result['v_ratio']:.1f}x DualPipe's. "
            f"If dropping the second copy costs {result['v_ratio'] - 1:.0%} of the bubble then the "
            "second copy was worth that much, and the table says a schedule with Chimera's memory "
            "profile loses a fifth of DualPipe's advantage",
        ),
        practice.Check(
            "MECHANISM: DualPipe and DualPipeV share a formula up to a stipulated multiplier",
            result["same_formula"],
            "every schedule's bubble is numerator(P) / denominator(P, M), and at P=8, M=16 those "
            "are "
            + ", ".join(f"{name} {num}/{den}" for name, (num, den) in parts.items())
            + f". DualPipe and DualPipeV are the same fraction, and differ only by the 1.2 in "
            "bubble_dualpipev, so the comparison the exercise asks for reduces to one multiplier "
            "and two labels",
        ),
        practice.Check(
            "FINDING: the field that would have carried the distinction is never set",
            "scales_with_micro_batches" in result["fields"] and result["assigned"],
            f"ScheduleStats declares {result['fields']} and summarize returns tuples, so "
            "scales_with_micro_batches is assigned nowhere. It is the field that would "
            "distinguish a schedule whose bubble is constant in M from one whose bubble shrinks "
            "-- which, as Exercise 1 measures, is none of them: all four shrink, because every "
            "numerator is constant in P and every denominator is linear in M",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
