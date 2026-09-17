"""Exercise 2 — the speedup collapses 1.99x to 1.27x, and part of the collapse is the baseline improving.

    Reduce the coordination heuristic's strength (set `coordination_weight=0.1`).
    Re-run. Show that speedup collapses. Explain why: the workers duplicate effort
    when they cannot coordinate.

Reading of the exercise: speedup is unique progress at N workers over unique
progress at one, both measured at the same coordination weight, because the
weight is a property of the workers and a single-worker run has it too. The sweep
covers weights 1.0, 0.5, 0.1 and 0.0, since the exercise names two points and the
claim is about a trend.

**ANSWER: at N=2 the speedup falls 1.99x to 1.27x, and at 0.0 to 1.15x.**

    weight   N=1 progress   N=2      N=4      N=8
     1.0         193       1.99x    2.07x    2.07x
     0.5         180       1.61x    1.97x    2.18x
     0.1         172       1.27x    1.53x    1.81x
     0.0         171       1.15x    1.17x    1.17x

**MECHANISM: the duplication is exactly what the trace records.** At weight 0
every worker stays on its intended category `A`, so every step produces one
unique `A` and `N-1` tokens tagged `redundant` -- `unique_progress` is 197, 200 and 200
at N=2, 4 and 8 -- within three of each other -- against 1,352 work tokens at
N=8.

**FINDING: part of the collapse is the single-worker baseline getting better.**
One worker's unique progress rises from **171 to 193** as the weight goes 0 to 1
-- a 13% gain from a heuristic that exists to prevent collisions with workers
that are not there. At weight 1.0 the `coordination_weight` branch fires before
the `coord`-token branch, so a coordinating worker never spends a step on
coordination; that is where the extra 22 tokens come from.

**FINDING: at weight 0.5 more workers beat perfect coordination.** N=8 scores
**2.18x** at weight 0.5 against **2.07x** at weight 1.0, because the denominator
moved: 180 against 193. The ratio the exercise asks for is sensitive to a
baseline that the same knob is changing, so "speedup" and "progress" do not order
the configurations the same way.

Structure: `arm` runs the lesson's own `run_hogwild` at one (workers, weight)
pair; `speedup` divides by the single-worker run at the same weight.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "22-async-hogwild-inference"
STEPS, TARGET = 200, 100
WEIGHTS = (1.0, 0.5, 0.1, 0.0)
WORKERS = (2, 4, 8)


def arm(ref, workers, weight):
    return ref.run_hogwild(workers, STEPS, TARGET, weight)


def speedups(ref, weight):
    base = arm(ref, 1, weight)
    return {"baseline": base["unique_progress"],
            "coord_tokens": base["coord_tokens"],
            **{n: arm(ref, n, weight)["unique_progress"] / base["unique_progress"]
               for n in WORKERS}}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = {weight: speedups(ref, weight) for weight in WEIGHTS}
    zero = {n: arm(ref, n, 0.0) for n in (2, 4, 8)}
    return {
        "rows": rows,
        "zero_progress": {n: row["unique_progress"] for n, row in zero.items()},
        "zero_work": {n: row["work_tokens"] for n, row in zero.items()},
        "baseline_gain": rows[1.0]["baseline"] / rows[0.0]["baseline"],
        "collapse": rows[1.0][2] / rows[0.1][2],
    }


def column(rows, key, fmt):
    return ", ".join(f"w={weight} {format(row[key], fmt)}" for weight, row in rows.items())


def verify(result):
    rows = result["rows"]
    full, low, none = rows[1.0], rows[0.1], rows[0.0]
    return [
        practice.Check(
            "ANSWER: at N=2 the speedup falls 1.99x to 1.27x, and to 1.15x with no coordination",
            full[2] > 1.9 and low[2] < 1.4 and none[2] < 1.2,
            "the N=2 speedup over a single worker at the same weight is "
            + column(rows, 2, ".2f")
            + "x, on single-worker baselines of " + column(rows, "baseline", "d")
            + f" unique tokens. Dropping the weight from 1.0 to 0.1 costs "
            f"{result['collapse']:.2f}x of the N=2 speedup, and turning it off entirely leaves "
            f"{none[2]:.2f}x",
        ),
        practice.Check(
            "MECHANISM: the duplication is exactly what the trace records",
            max(result["zero_progress"].values())
            - min(result["zero_progress"].values()) <= 5,
            f"at weight 0 every worker stays on its intended category A, so each step produces "
            f"one unique A and N-1 tokens tagged redundant: unique_progress is "
            + ", ".join(f"N={n} {v}" for n, v in result["zero_progress"].items())
            + " -- within three of each other at every worker count, and identical from 4 "
            "onward -- against work-token counts of "
            + ", ".join(f"{v}" for v in result["zero_work"].values()),
        ),
        practice.Check(
            "FINDING: part of the collapse is the single-worker baseline getting better",
            result["baseline_gain"] > 1.1 and full["coord_tokens"] == 0,
            f"one worker's unique progress rises from {none['baseline']} to {full['baseline']} as "
            f"the weight goes 0 to 1 -- {result['baseline_gain']:.2f}x from a heuristic that "
            f"exists to prevent collisions with workers that are not there. At weight 1.0 the "
            f"coordination branch fires before the coord-token branch, so a coordinating worker "
            f"emits {full['coord_tokens']} coordination tokens against {none['coord_tokens']}, "
            "and that is where the extra tokens come from",
        ),
        practice.Check(
            "FINDING: at weight 0.5 more workers beat perfect coordination",
            rows[0.5][8] > full[8],
            f"N=8 scores {rows[0.5][8]:.2f}x at weight 0.5 against {full[8]:.2f}x at weight 1.0, "
            f"because the denominator moved: {rows[0.5]['baseline']} against "
            f"{full['baseline']}. The ratio the exercise asks for is sensitive to a baseline the "
            "same knob is changing, so speedup and progress do not order the configurations the "
            "same way",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
