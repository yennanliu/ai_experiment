"""Exercise 2 — the 1F1B formula reproduces a simulated schedule exactly; DualPipe's has no schedule to check against.

    Sketch the schedule table for `(P=4, micro_batches=8, schedule=dualpipe)` by
    hand. Mark each time slot with the micro-batch ID and direction. Identify the
    first time slot where bubbles are absent.

Reading of the exercise: the lesson ships four bubble *fractions* and no
schedule, so the table has to be built. It is built for 1F1B first -- warmup of
`P-1-p` idle slots per rank, then alternating forward and backward, then a
mirrored cooldown -- because that is the one schedule whose formula can be
checked against it. The exercise's own question, which slot is the first with no
bubble, is then answered from the table.

**ANSWER: the first bubble-free slot is t = 3, which is `P - 1`.**

    rank 0   .  .  . F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7  .  .  .
    rank 1   .  . F0 F1 F2 B0 F3 B1 F4 B2 F5 B3 F6 B4 F7 B5 B6 B7  .  .  .  .
    rank 2   . F0 F1 B0 F2 B1 F3 B2 F4 B3 F5 B4 F6 B5 F7 B6 B7  .  .  .  .  .
    rank 3  F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7  .  .  .  .  .  .

Every rank is busy from t=3 to t=15 -- **13 of 22 slots**, 59.1% of the wall
clock -- and the warmup that precedes it is exactly the pipeline depth.

**FINDING: the 1F1B formula is exact, to twelve decimal places.** The simulated
bubble is **27.27273%** and `bubble_1f1b(4, 8)` is 27.27273%, and they agree on
`(4,8)`, `(8,16)`, `(4,16)`, `(8,64)`, `(16,128)` and `(2,8)`. `2(P-1)` idle
slots out of `2M + 2(P-1)` is not an approximation -- it is a count of the table
above.

**FINDING: nothing in the lesson lets the same check be run on DualPipe.**
`bubble_dualpipe` is `(P-1)//2` over `3M + (P-1)`, and its docstring asserts
"stable phase bubble is zero. Warmup/cooldown has fixed bubble independent of M"
without a schedule to count. The exercise asks for a DualPipe table and the
lesson's answer key is a formula that cannot be derived from one.

**MECHANISM: integer division makes the bubble a step function of P.**
`(P-1)//2` is **0 at P=2**, so `bubble_dualpipe(2, 64)` is exactly 0.000% -- a
two-stage pipeline with no bubble at all -- and it is **1** at both P=3 and P=4,
giving 0.515% and 0.513%, where 1F1B moves 3.030% to 4.478% across the same
step.

Structure: `schedule` builds the 1F1B table rank by rank; `bubble` counts idle
slots; `first_full` answers the exercise's question.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "19-dualpipe-parallelism"
STAGES, MICRO = 4, 8
SHAPES = ((4, 8), (8, 16), (4, 16), (8, 64), (16, 128), (2, 8))
DEPTHS = (2, 3, 4, 5, 8, 16)
IDLE = "."


def schedule(stages, micro):
    """1F1B, rank by rank: warmup, steady alternation, mirrored cooldown."""
    rows = {}
    for rank in range(stages):
        warm = stages - 1 - rank
        head = min(warm, micro)
        row = [IDLE] * warm + [("F", i) for i in range(head)]
        for i in range(micro - head):
            row += [("F", head + i), ("B", i)]
        row += [("B", i) for i in range(micro - head, micro)]
        row += [IDLE] * warm
        rows[rank] = row
    width = max(len(row) for row in rows.values())
    return {rank: row + [IDLE] * (width - len(row)) for rank, row in rows.items()}


def bubble(rows):
    slots = sum(len(row) for row in rows.values())
    return sum(row.count(IDLE) for row in rows.values()) / slots


def busy_window(rows):
    """First and last time slot at which every rank has work."""
    width = len(next(iter(rows.values())))
    full = [t for t in range(width) if all(rows[rank][t] != IDLE for rank in rows)]
    return (full[0], full[-1], width) if full else (None, None, width)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = schedule(STAGES, MICRO)
    first, last, width = busy_window(rows)
    agreement = {shape: (bubble(schedule(*shape)), ref.bubble_1f1b(*shape)) for shape in SHAPES}
    return {
        "first": first, "last": last, "width": width,
        "busy_share": (last - first + 1) / width,
        "simulated": bubble(rows),
        "closed": ref.bubble_1f1b(STAGES, MICRO),
        "agrees": all(abs(sim - closed) < 1e-12 for sim, closed in agreement.values()),
        "shapes": len(agreement),
        "dual_steps": {depth: ((depth - 1) // 2, ref.bubble_dualpipe(depth, 64),
                               ref.bubble_1f1b(depth, 64)) for depth in DEPTHS},
        "stages": STAGES,
    }


def verify(result):
    steps = result["dual_steps"]
    return [
        practice.Check(
            f"ANSWER: the first bubble-free slot is t = {result['first']}, which is P - 1",
            result["first"] == result["stages"] - 1,
            f"the simulated table is {result['width']} slots wide and every rank is busy from "
            f"t={result['first']} to t={result['last']} -- {result['last'] - result['first'] + 1} "
            f"slots, {result['busy_share']:.1%} of the wall clock. The warmup that precedes it is "
            f"exactly the pipeline depth P - 1 = {result['stages'] - 1}, which is how long the "
            "first micro-batch takes to reach the last rank",
        ),
        practice.Check(
            "FINDING: the 1F1B formula is exact, to twelve decimal places, on six shapes",
            result["agrees"] and abs(result["simulated"] - result["closed"]) < 1e-12,
            f"the simulated bubble at P={STAGES}, M={MICRO} is {result['simulated']:.5%} and "
            f"bubble_1f1b returns {result['closed']:.5%}. They agree on all {result['shapes']} "
            "shapes tried. 2(P-1) idle slots out of 2M + 2(P-1) is not an approximation -- it is "
            "a count of the table",
        ),
        practice.Check(
            "FINDING: nothing in the lesson lets the same check run on DualPipe",
            steps[4][1] < result["closed"] / 5,
            "bubble_dualpipe is (P-1)//2 over 3M + (P-1), and its docstring asserts that the "
            "stable-phase bubble is zero and the warmup bubble is fixed, without a schedule to "
            f"count. At P={STAGES} it returns {steps[4][1]:.3%} against the simulated "
            f"{result['simulated']:.3%} for 1F1B on the same shape. The exercise asks for a "
            "DualPipe table and the lesson's answer key is a formula no table was derived from",
        ),
        practice.Check(
            "MECHANISM: integer division makes the bubble a step function of P",
            steps[2][0] == 0 and steps[3][0] == steps[4][0],
            "(P-1)//2 is "
            + ", ".join(f"P={p} -> {num}" for p, (num, _, _) in steps.items())
            + f", so bubble_dualpipe(2, 64) is exactly {steps[2][1]:.3%} -- a two-stage pipeline "
            f"with no bubble at all -- and P=3 and P=4 share a numerator, giving {steps[3][1]:.3%} "
            f"and {steps[4][1]:.3%} where 1F1B moves {steps[3][2]:.3%} to {steps[4][2]:.3%} "
            "across the same step",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
